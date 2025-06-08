"""
Implementation of APV-MCTS
"""
from typing import NamedTuple, Dict, Union, Sequence, Callable
from time import time

import tensorflow as tf
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory as mp_shm
mp.set_start_method('spawn', force=True)  # use spawn method for multiprocessing
from collections import namedtuple
import contextlib
import tree

from .environment import AssemblyGame, AssemblyGameModel
from .inference_service import InferenceTask, InferenceResult, AlphaDevInferenceService
from .shared_memory import *
from .config import ADConfig
from .device_config import apply_device_config

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
)

logger.setLevel(logging.INFO)

# Shared memory has a fixed size. for each MCTS, we do not need to re-allocate it
# but we do need to re-initialize it.
# so that it is with 100% certainty that addresses we didn't touch are zeroed out.
# hence no need to worry about whether a node is being created or is already there.

class Node(BlockLayout):
    """
    Node in the search tree, residing directly in the shared memory.
    Each node corresponds to a game state and contains edges to its successor states,
    each corresponding to an action that can be taken from this state.
    There are always num_actions such edges, and successor nodes may or may not be expanded.
    Nodes that are not have not been expanded yet have all their attributes set 
    to zero, at the initialization of the shared memory.

    Each edge contains the following attributes:
    - terminal: whether the destination node is terminal
    - expanded: whether the node has been expanded
    - prior: prior probability of the action leading to this node
    - action_mask: mask of valid actions leading to this node
    - N: number of visits to this node
    - W: total value of the node, which is the sum of the value estimates and the empirical discounted return
    - value: value estimate of the node, which is total_value / visit_count
    - virtual_loss: virtual loss to discourage other workers from visiting the same node. 
        implemented as a counter of processes that are currently visiting the branch starting at this node.
        the value of this node is calculated as 
        (total_value + virtual_loss * W_vl)/ (visit_count + virtual_loss)
    All of these attributes are stored in a congiguous block of shared memory in the form of numpy arrays.
    Additionally, each node has a header, which contains three scalars:
    - num_actions: number of actions available at this node
    - parent_offset: offset in the shared memory where the parent node's block starts
    - action_id: integer index corresponding to the action leading to this node.
    
    Each node class and inherited classes have a node_size attribute which is the size of the node in bytes
    and calculated from the node spefification defined at the class level.
    
    Completely lock-free.
    """
    _num_actions = ADConfig.task_spec.num_actions       # number of actions depends on the task specification
    hdr_parent = 0; hdr_action = 1; hdr_expanded = 2; hdr_terminal = 3
    _elements = {
        'header':   ArrayElement(np.int32,   (4,)          ), # parent_offset, action_id, terminal, expanded
        'prior':    ArrayElement(np.float32, (_num_actions,)), # prior probabilities of actions
        'R':        ArrayElement(np.float32, (_num_actions,)), # One-time reward observed at this node
        'W':        ArrayElement(np.float32, (_num_actions,)), # predicted value + empirical discounted return
        'N':        ArrayElement(np.int32,   (_num_actions,)), # number of visits to the node
        'mask':     ArrayElement(np.bool_,   (_num_actions,)), # mask of valid actions leading to this node
        'children': ArrayElement(np.uint32,  (_num_actions,)), # offsets to the children nodes
    }
    # virtual loss constant
    const_vl = -1.0
    
    def __init__(self, shm, offset):
        super().__init__(shm, offset)
        self._parent = None
    
    def expand(self, prior):
        self.prior = prior
        self.header[self.__class__.hdr_expanded] = True
    
    def select(self, action_id):
        """Increment W by const_vl and N by 1 for the given action_id."""
        logger.debug('%s.select called with action_id %s.', repr(self), action_id)
        self.W[action_id] += self.const_vl
        self.N[action_id] += 1
        child_offset = self.children[action_id]
        if child_offset == 0:
            logger.debug('%s.select: child offset is 0, returning None.', repr(self))
            return None
        return self.__class__(self.shm, child_offset)
    
    def deselect(self, action_id, reward=0.0):
        """Inverse operation of select."""
        logger.debug('%s.deselect called with action_id %s and reward %s.', repr(self), action_id, reward)
        self.W[action_id] += -self.const_vl + reward
        self.N[action_id] -= 1
    def visit_child(self, action_id, value):
        """W = W - const_vl + value; N = N - 1 + 1 for the given action_id."""
        logger.debug('%s.visit_child called with action_id %s and value %s.', repr(self), action_id, value)
        self.W[action_id] += -self.const_vl + value
        # N = N - 1 + 1 -> N = N, so this is a no-op
    def update_child(self, action_id, value):
        """Update the child without touching the virtual loss."""
        logger.debug('%s.update_child called with action_id %s and value %s.', repr(self), action_id, value)
        self.W[action_id] += value
        self.N[action_id] += 1
    
    def is_consistent(self):
        logger.debug('%s.is_consistent called.', repr(self))
        if self.is_root:
            return True
        return self.parent.children[self.action_id] == self.offset
    
    @property
    def children_values(self):
        """To be called during search for selecting children. considers the virtual loss."""
        logger.debug('%s.children_values called.', repr(self))
        return np.where(
            self.mask & (self.N != 0),
            np.divide(self.W, self.N),
            0.0
        )
    
    @property
    def children_visits(self):
        """To be called during search for selecting children. considers the virtual loss."""
        logger.debug('%s.children_visits called.', repr(self))
        return self.N
    
    @property
    def children_priors(self):
        """To be called during search for selecting children. considers the virtual loss."""
        logger.debug('%s.children_priors called.', repr(self))
        return self.prior

    @property
    def action_mask(self):
        """To be called during search for selecting children. considers the virtual loss."""
        logger.debug('%s.action_mask called.', repr(self))
        return self.mask
    
    @property
    def value(self):
        """To be called during backprop, ignores the virtual loss."""
        assert False, '%s.value called.' % (repr(self),)
        parent = self.parent # instantiate once
        return (
            parent.W[self.action_id] / parent.N[self.action_id]
            ) if parent.N[self.action_id] > 0 else 0.0
    @property
    def visit_count(self):
        """To be called during backprop, ignores the virtual loss."""
        logger.debug('%s.visit_count called', repr(self))
        if self.is_root:
            return np.sum(self.N)
        return self.parent.N[self.action_id]
    @property
    def parent_offset(self): return self.header[self.__class__.hdr_parent]
    @property
    def action_id(self):     return self.header[self.__class__.hdr_action]
    @property
    def expanded(self):      return self.header[self.__class__.hdr_expanded] != 0
    @property
    def terminal(self):      return self.header[self.__class__.hdr_terminal] != 0
    def set_terminal(self, terminal): self.header[self.__class__.hdr_terminal] = terminal
    @property
    def is_root(self):       return self.header[self.__class__.hdr_parent] == -1

    def set_root(self):
        """Set this node as the root node."""
        self.header[self.__class__.hdr_parent] = -1
        self.header[self.__class__.hdr_action] = -1
        self._parent = None # make sure parent is not incorrectly set.
    
    def _get_parent(self):
        if self.is_root:
            return None
        if self._parent is None: self._parent = self.__class__(self.shm, self.parent_offset)
        return self._parent
    
    def set_parent(self, parent_offset, action):
        self.header[self.__class__.hdr_parent] = parent_offset
        self.header[self.__class__.hdr_action] = action
        self._parent = None # make sure parent is not incorrectly set.
        logger.debug('%s.set_parent parent_offset %s action %s.', repr(self), parent_offset, action)
    
    @property
    def parent(self) -> 'Node':
        assert not self.is_root, "Root has no parent."
        return self._get_parent()
    
    def set_child(self, action_id, child_offset):
        """Set the child offset for the given action_id."""
        logger.debug('%s.set_child action_id %s child_offset %s. current child offset: %s', repr(self), action_id, child_offset, self.children[action_id])
        self.children[action_id] = child_offset
    
    def set_legal_actions(self, legal_actions):
        self.mask[:] = legal_actions
    
    def set_reward(self, reward):
        logger.debug('%s.set_reward called with reward %s.', repr(self), reward)
        self.parent.R[self.action_id] = reward
    
    def __repr__(self):
        return f'Node(offset={self.offset}, action_id={self.action_id})'


class SharedTreeHeader(BlockLayout):
    _elements = {
        'index': AtomicCounterElement(),  # index of the next block to be used
    }

class TreeFull(Exception): pass

class SharedTree(BaseMemoryManager):
    """
    Shared Memory which stores a tree structure.
    
    Maintains two shared memory blocks:
    1. header block stores the index of the next block to be used.
    2. data block stores the nodes of the tree
    """
    def __init__(self, num_blocks, node_cls: Node, name: str = 'SharedTree'):
        self.num_blocks = num_blocks + 1 # 1 for the root.
        self._node_cls = node_cls
        # declare shared memory regions
        self._header_size = SharedTreeHeader.get_block_size(length=num_blocks)
        self._header_name = f'{name}_header'
        
        self._node_size = self._node_cls.get_block_size()
        
        self._data_size = self.num_blocks * self._node_size
        self._data_name = f'{name}_data'
        
        # process-local
        self._is_main = False
        self._local_write_head = None
        logger.debug('SharedTree initialized with %d blocks, node size %d, header size %d, data size %d.', self.num_blocks, self._node_size, self._header_size, self._data_size)
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=True, size=self._data_size)
        
        self.header = SharedTreeHeader(self._header_shm, 0)
        with self.header.index() as index_counter:
            self._local_write_head = index_counter.load() # get the current write head
        
        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        logger.debug('SharedTree.reset called.')
        SharedTreeHeader.clear_block(self._header_shm, 0, length=self.num_blocks)
        with self.header.index() as idx:
            idx.store(1) # do not write to the root.
        for i in range(self.num_blocks):
            self._node_cls.clear_block(self._data_shm, self.i2off(i))
        # initialize the root node
        root: Node = self.get_root()
        root.set_root()
    
    def attach(self):
        logger.debug('SharedTree.attach')
        self._is_main = False
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=False, size=self._data_size)

        self.header = SharedTreeHeader(self._header_shm, 0)
        self._update_index() # obtain a write location
    
    def __del__(self):
        logger.debug('SharedTree.__del__ called. is main: %s', self._is_main)
        del self.header
        self._header_shm.close()
        self._data_shm.close()
        if self._is_main:
            # only the main process should delete the shared memory
            self._header_shm.unlink()
            self._data_shm.unlink()
    
    def _update_index(self):
        with self.header.index() as index_counter:
            index = index_counter.fetch_inc()
            logger.debug('SharedTree._update_index (nb %d) prev: %d crnt: %d, next: %d', self.num_blocks, self._local_write_head, index, index_counter.load())
        if index >= self.num_blocks:
            self._local_write_head = None # we are done
        else:
            self._local_write_head = index
        return index
    
    def fail(self):
        """We didn't manage to write, try again next time to the same block."""
        pass
    
    def succeed(self):
        """We successfully wrote to the block, we can find the next block."""
        self._local_write_head = self._update_index()
    
    def append_child(self, parent: Node, action: int) -> Union[Node, None]:
        """
        Attempt to append a child node to the parent node by
        first writing the offset corresponding to the 
            current write head to the parent node's children array
        then writing the parent's offset and action id to the child node's header
        finally checking if the write to the parent was successful.
        if yes, succeed() else fail().
        
        Multiple processes may attempt to write to the same parent node at 
        the same time and we do not lock the tree for this operation.
        This way, we be almost certain that the tree will stay consistent.
        Losing one or two children due to race conditions is acceptable.
        """
        logger.debug('SharedTree.append_child called with parent_offset %s and action %s.', parent, action)
        if self.is_full():
            raise TreeFull() # signal the caller that no more blocks are available.
        if parent.children[action] != 0:
            # another process already appended this action.
            self.fail()
            return None
        
        child_offset = self.i2off(self._local_write_head)
        parent.set_child(action, child_offset)
        logger.debug('SharedTree.append_child: parent %s children[%d] set to %d.', repr(parent), action, child_offset)
        # now write the parent offset and action id to the child node's header
        child: Node = self._node_cls(self._data_shm, child_offset)
        child.set_parent(parent.offset, action)
        # now check if the write was successful
        if parent.children[action] == child_offset:
            logger.debug('SharedTree.append_child: append successful. index %d new node: %s.', self._local_write_head, repr(child))
            self.succeed()
            return child
        else:
            logger.debug('SharedTree.append_child: append failed, this node: %s, parent: %s.', repr(child), repr(parent))
            self.fail()
            return None
    
    def i2off(self, index: int) -> int:
        return index * self._node_size
    
    def get_node(self, index) -> Node:
        """Return the root node, at offset 0."""
        if index == 0:
            return self.get_root()
        return self._node_cls(self._data_shm, self.i2off(index))
    
    def get_root(self):
        return self._node_cls(self._data_shm, 0)
    
    def get_by_offset(self, offset: int) -> Node:
        return self._node_cls(self._data_shm, offset)

    def is_full(self) -> bool:
        """
        Check if the tree is full.
        The tree is full if the index is equal to the number of blocks.
        """
        return self.header.index.peek() >= self.num_blocks

class TaskAllocatorHeader(BlockLayout):
    _elements = {
        'process_task': ArrayElement(np.int32, (ADConfig.num_actors,)),  # list of tasks assigned to each process
    }
class SharedTaskAllocator(BaseMemoryManager):
    def __init__(self, name: str = 'SharedTaskAllocator'):
        self._header_size = TaskAllocatorHeader.get_block_size()
        self._header_name = f'{name}_header'
        self._header_shm = None
        self._header = None
        self._is_main = False
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        
        self.header = TaskAllocatorHeader(self._header_shm, 0)
        
        self.reset()
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        TaskAllocatorHeader.clear_block(self._header_shm, 0)

    def attach(self):
        self._is_main = False
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self.header = TaskAllocatorHeader(self._header_shm, 0)
    
    def __del__(self):
        logger.debug('SharedTaskAllocator.__del__ called. is main: %s', self._is_main)
        del self.header
        self._header_shm.close()
        if self._is_main:
            # only the main process should delete the shared memory
            self._header_shm.unlink()
    
    def allocate(self, task_ids: Sequence[int]) -> Dict[int, int]:
        """
        Allocate tasks to processes.
        Returns a dictionary mapping process id to task id.
        """
        assert self._is_main, "allocate() should only be called in the main process."
        assert len(task_ids) == ADConfig.num_actors, "Number of task ids must match the number of actors."
        
        # reset the header
        self.reset()
        self.header.process_task[:] = task_ids
    
    def get_task(self, process_id: int) -> int:
        return self.header.process_task[process_id]

class APV_MCTS(object):
    """
    Asynchronous Policy Value Monte Carlo Tree Search (APV-MCTS).
    
    Based on Silver, D. et al. (2016) "Mastering the game of Go with deep neural networks and tree search".
    
    This class works similarly to naive PV-MCTS but it executes rollouts asynchronously.
    
    APV-MCTS consists of a master process, which initiates the search and spans both CPU and GPU worker processes.
    CPU workers are responsible for executing rollouts and 
    GPU workers are used to evaluate new states discovered during rollouts.
    
    Rollouts are executed individually on the shared tree structure. There are the usual four phases:
    - Selection: traverse existing nodes in the tree until we find a leaf node. 
        following Silver, D. et al. (2016), add a virtual loss to each node visited to discourage other workers from visiting the same node.
    - Simulation: when a leaf node is reached, 
        Execute the sequence of actions corr. to visited edges in the tree and observe the new state
        enqueue a new inference task to the GPU worker
    - Evaluation:
        Once the GPU worker returns the value estimate, update the node with the value estimate and the observed reward.
    - Backpropagation:
        Backpropagate the combined value estimate and replace the virtual loss with the actual visit count and total value. 
    Selection and simulation are executed in a single task but the result of the evaluation is not awaited.
    Once the GPU worker returns, a new task is created to backpropagate the value estimate.
    in the meantime, other workers can continue executing rollouts.
    The GPU worker applies continuous batching and also updates its parameters from the parameter server.
    
    The worker processes work on a shared tree structure which consists of nodes and edges.
    Nodes correspond to game states and edges are tuples
    (prior, visit_count, total_value, value)
    with value = total_value / visit_count. 
    and total_value is value_estimate + empirical discounted return obtained during rollouts.
    """
    # task 0 means no task.
    SEARCH_SIM_TASK = 1
    BACKUP_TASK = 2
    
    def __init__(self,
            model,
            search_policy,
            network_factory,
            num_simulations,
            num_actions,
            discount,
            dirichlet_alpha,
            exploration_fraction,
            node_class: Node = Node,
            batch_size: int = 1,
            inference_buffer_size: int = None,
            network_factory_args=(),
            network_factory_kwargs={},
            inference_device_config: Union[Dict, None] = None,
            observers: Sequence = [],
            name:str='apv-mcts' # needs to be specified is multiple instances are used.
        ):
        self.model = model # a (learned) model of the environment.
        self.search_policy = search_policy # a function that takes a node and returns an action to take.
        self.num_simulations = num_simulations  # number of simulations to run per search
        self.num_actions = num_actions  # number of actions in the environment [a noop, we know this from ADConfig already]
        assert num_actions == ADConfig.task_spec.num_actions, "num_actions doesn't match the task specification. You did somethin dumb miki"
        self.discount = discount  # discount factor for the value estimate
        self.inference_buffer_size = inference_buffer_size or num_simulations  # size of the inference buffer
        self.observers = observers  # list of observers that evaluate search statistics.

        # applying Dirichlet noise to the prior probabilities at the root.
        self.dirichlet_alpha = dirichlet_alpha  # alpha parameter for the Dirichlet noise
        self.exploration_fraction = exploration_fraction  # fraction of the prior to add Dirichlet noise to
        
        # declare shared memory ( no init )
        self.tree_factory = functools.partial(
            SharedTree,
            num_simulations, node_class, f'{name}.tree')
        # TODO: this could be optional; and each process can have its own network instance instead.
        self.inference_buffer_factory = functools.partial(
            AlphaDevInferenceService,
            self.inference_buffer_size,
            network_factory,
            batch_size,
            network_factory_args,
            network_factory_kwargs,
            f'{name}.inference'
            )
        self.task_allocator_factory = functools.partial(
            SharedTaskAllocator, f'{name}.task_allocator')
        
        self._init_task_allocation = \
            [APV_MCTS.SEARCH_SIM_TASK] * (ADConfig.num_actors - 1) + [APV_MCTS.BACKUP_TASK] # last actor is the one listening to the evaluator
        
        # construct local shared memory managers
        self.tree = self.tree_factory()
        self.inference_buffer = self.inference_buffer_factory()
        self.task_allocator = self.task_allocator_factory()
        
        # configure the shared memory managers
        self.tree.configure()
        self.inference_buffer.configure()
        self.task_allocator.configure()
        
        # set initial task allocation
        self.task_allocator.allocate(self._init_task_allocation)
        
        # declare the inference process and actor pool
        # TODO: make optional
        self.inference_process = mp.Process(
            target=_run_inference,
            args=(
                self.inference_buffer_factory,
                inference_device_config,
            ),
            name=f'{name}.inf_proc')
        
        self.actor_pool = mp.Pool(processes=ADConfig.num_actors)
        
        # start the inference processs
        self.inference_process.start()
        logger.debug(f"APV_MCTS[main]: Inference process started with PID {self.inference_process.pid}.")
        
        logger.debug("APV_MCTS[main]: Finished initialization.")
    
    def search(self, observation):
        """
        Perform APV-MCTS search on the given observation.
        This method initializes the search tree with root corresponding to the given observation,
        and runs num_simulations rollouts from the root node using a pool of worker processes.
        Finally, it returns a pointer to the root node, which can be used to perform post-mcts action selection.
        """
        # TODO: support only partially resetting the tree.
        logger.debug('APV_MCTS[main process] Starting search simulation phase.')
        self.tree.reset()
        
        self.inference_buffer.submit(node_offset=0, observation=observation)
        
        logger.debug('APV_MCTS[main process] Waiting for inference service to return prior and value estimate.')
        with self.inference_buffer.poll_ready() as inference_result:
            _, prior, _ = inference_result
        logger.debug('APV_MCTS[main process] Received prior and value estimate from inference service.')
        # Add exploration noise to the prior.
        noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
        prior = prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
        
        legal_actions = self.model.legal_actions()
        
        root: Node = self.tree.get_root()
        root.set_legal_actions(legal_actions)
        
        root.expand(prior)
        logger.debug('APV_MCTS[main process] Root node initialized with prior and legal actions. Starting pool')
        
        # all that is left is to start the pool.
        logger.debug('APV_MCTS[main process] Starting actor pool with %d actors.', ADConfig.num_actors)
        statistics = self.actor_pool.map(
            _run_task,
            [(i,
              self.tree_factory, self.inference_buffer_factory, self.task_allocator_factory,
              self.model, self.search_policy, self.discount
              ) for i in range(ADConfig.num_actors)
            ]
        )
        
        for o in self.observers:
            o.update(statistics, root, self.tree, self.inference_buffer)
        
        logger.debug('APV_MCTS[main process] search done. Root (W/N):\n %s', list(zip(root.W, root.N)))
        
        return root  # return the root node, which now contains the search results

def _run_inference(
    inference_factory: Callable[[], AlphaDevInferenceService],
    device_config: Union[Dict, None] = None
    ):
    """To be called from a subprocess"""
    if device_config is not None:
        logger.debug("APV_MCTS[inference process] Applying device configuration")
        apply_device_config(device_config)
    # initialize the inference service
    logger.debug("APV_MCTS[inference process] Initializing inference service.")
    # TODO: setup tensorflow device config.
    inference_buffer = inference_factory()
    inference_buffer.attach()
    # run indefinitely
    inference_buffer.run()

def _phase_1(root: Node, tree: SharedTree, model: AssemblyGameModel,
             search_policy: callable, inference_buffer: AlphaDevInferenceService,
             discount: float):
    """
    Starting from the root (s_0), traverse the existing nodes in the tree following a search policy
    a_L = argmax_a(Q + u); with u=puct(s_L) = c_puct * prior * sqrt(N) / (1 + N) and Q = W / N
    until a leaf node is reached.
    In each node (s_L) we visit, we increment the virtual loss as W(s_L,a_L) += vl_const; N(s_L,a_L) += 1
    Once a leaf node is reached, we 
    - obtain reward and legal actions from the model
        r_L, mask_L = model.step(a[0..L])
    - submit a new inference task to the inference service.
    - backpropagate the reward for each t <= L as
        W(s_t,a_t) = W(s_t,a_t) - vl_const + r_L*discount^(L-t); N(s_t,a_t) = N(s_t,a_t) - 1 + 1 [i.e. no actual update]
    
    Once the inference task is done, we will asynchronously make a second backward pass (and check tree consistency),
    see `_phase_2` method.
    """
    logger.debug('APV_MCTS[phase 1] Starting search simulation phase.')
    node = root
    actions = []; trajectory = []
    while node is not None and node.expanded: # if node was never visited, the parent with return None.
        action = search_policy(node)
        actions.append(action)   # keep track of actions taken.
        trajectory.append(node)  # keep track of parents.

        node = node.select(action) # increment virtual loss
        logger.debug('APV_MCTS[phase 1] Selected action %s and got node %s.', action, repr(node))
    
    logger.debug('APV_MCTS[phase 1] Reached leaf via %s.', list(zip(trajectory, actions)))
    # before simulating, initialize the new node.
    if node is not None:
        logger.debug('APV_MCTS[phase 1] Node %s is not None and not expanded', repr(node))
        # someone else is already evaluating this node.
        return False
    # otherwise, append new child and check if it worked.
    node = tree.append_child(trajectory[-1], actions[-1]) # pass the path to make sure only one block is reserved for this node.
    if node is None:
        # if it didn't work, clean up.
        logger.debug('APV_MCTS[phase 1] Failed to append child node.')
        # this node is already being evaluated by some other process.
        for n, a in zip(trajectory, actions):
            n.deselect(a)
        return False  # we need to try again, the node was overwritten by another process
    
    logger.debug('APV_MCTS[phase 1] Appended new node %s with offset %s.', repr(node), node.offset)
    logger.debug('APV_MCTS[phase 1] Calling model with actions %s.', actions)
    # then run the simulation update the node with the results. This delay also enables other processes to write stuff
    timestep = model.step(actions)
    legal_actions = model.legal_actions()
    
    logger.debug('APV_MCTS[phase 1] Simulation returned with reward %s.', timestep.reward)
    # now we can check for tree consistency (whether any other process has overwritten the node)
    if not node.is_consistent():
        logger.debug('APV_MCTS[phase 1] Node %s is not consistent.', repr(node))
        for n, a in zip(trajectory, actions):
            n.deselect(a)
        return False # we need to try again, the node was overwritten by another process
    
    # otherwise, submit a new inference task to the inference service.
    observation = timestep.observation
    inference_buffer.submit(
        node_offset=node.offset,
        observation=observation
    )
    logger.debug('APV_MCTS[phase 1] Submitted inference task for node %s', repr(node))
    # perform a backward pass with the reward and legal actions
    reward = timestep.reward[0] # there might be other outputs but they are irrelevant for mcts
    terminal = timestep.last()
    
    node.set_terminal(terminal)
    node.set_legal_actions(legal_actions)
    node.set_reward(reward)

    # backpropagate the reward for each t < L and remove the virtual loss.
    for n, a in zip(trajectory, actions):
        reward *= discount
        n.visit_child(a, reward)
    logger.debug('APV_MCTS[phase 1] Backpropagated reward %s for actions %s.', reward, actions)
    return True  # we successfully submitted the task for evaluation

def _phase_2(tree: SharedTree, inference_buffer: AlphaDevInferenceService, discount: float):
    """
    Process the predictions of the network.
    - Poll the inference buffer for a result.
    - Update the prior and value of the node with the result.
    - Backpropagate the value estimate to the root node.
    """
    with inference_buffer.poll_ready(timeout=10) as result:
        if result is None: 
            return False  # no result available, nothing to do here.
        node_offset = result.node_offset.item()
        logger.debug('APV_MCTS[phase 2] result before context exit: %s', node_offset)
        prior = result.prior
        value = result.value
    # obtain the node from the shared tree and check for consistency.
    logger.debug('APV_MCTS[phase 2] result obtained offset %s (type %s)', node_offset, type(node_offset))
    node = tree.get_by_offset(node_offset)
    logger.debug('APV_MCTS[phase 2] result with offset %s (corr. node %s).', node_offset, repr(node))
    if not node.is_consistent():
        logger.debug('APV_MCTS[phase 2] Node %s is not consistent. parent\'s pointer %s, child %s,',
                    repr(node), repr(node.parent), repr(tree.get_node(node.parent.children[node.action_id]//tree._node_size)))
        # the node was overwritten by another process, nothing to do here.
        return False
    # expand the current node
    logger.debug('APV_MCTS[phase 2] Expanding node %s', repr(node))
    node.expand(prior) # reward, legal_actions and terminal are already set
    # backpropagate without touching the virtual loss.
    logger.debug('APV_MCTS[phase 2] Backpropagating value %s for node %s', value, repr(node))
    while not node.is_root:
        parent = node.parent
        # update the parent with the value estimate and visit count
        parent.update_child(node.action_id, value)
        # then move to the parent node and discount the value
        node = parent
        value *= discount
    return True

import cProfile
import pstats
import subprocess

def _run_task(
    args):
        (
            process_id,
            tree_factory,
            inference_factory,
            task_allocator_factory,
            model,
            search_policy,
            discount
        ) = args
        logging.basicConfig(
            format=f'%(asctime)s - APV_MCTS[process {process_id}] - %(levelname)s - %(message)s',
            level=logging.DEBUG,
        )
        
        tree = tree_factory(); tree.attach()
        inference_buffer = inference_factory(); inference_buffer.attach()
        task_allocator = task_allocator_factory(); task_allocator.attach()
        
        
        my_task_id = task_allocator.get_task(process_id)
        root = tree.get_root()
        stats = {
            'process_id': process_id,
            'task_ids': [my_task_id],
            'num_fails': 0,
            'num_successes': 0,
            'start_time': time(),
        }
        logger.debug(f"APV_MCTS[process {process_id}] Starting task with id {my_task_id}.")
        should_stop = False
        
        if ADConfig.do_profiling:
            profiler = cProfile.Profile()
            profiler.enable()
        
        # num_iterations = 0
        while not should_stop: # iterate indefinitely
            try:
                # query task 
                stats['task_ids'].append(my_task_id)
                if my_task_id == APV_MCTS.SEARCH_SIM_TASK:
                    # run phase 1
                    result = _phase_1(
                        root=root, tree=tree, model=model,
                        search_policy=search_policy,
                        inference_buffer=inference_buffer,
                        discount=discount
                    )
                    if result:
                        logger.debug("APV_MCTS[process %s] Phase 1 done; success: %s full: %s, idle_ %s", process_id, result, tree.is_full(), inference_buffer.is_idle())
                    # # for debugging.
                    # if num_iterations == 5:
                    #     should_stop = True
                    # num_iterations += 1
                elif my_task_id == APV_MCTS.BACKUP_TASK:
                    # run phase 2
                    result = _phase_2(tree=tree, inference_buffer=inference_buffer, discount=discount)
                    if result:
                        logger.debug("APV_MCTS[process %s] Phase 2 done; success: %s full: %s, idle_ %s", process_id, result, tree.is_full(), inference_buffer.is_idle())
                    if tree.is_full() and inference_buffer.is_idle():
                        # if we didn't manage to process the result and the inference buffer is done,
                        # we can stop the process.
                        should_stop = True
                        logger.debug(f"APV_MCTS[process {process_id}] Inference buffer is done, stopping.")
                        break
                else:
                    logger.debug(f"Unknown task id {my_task_id} for process {process_id}.")
                    break
                if result: stats['num_successes'] += 1
                else:      stats['num_fails']     += 1
                my_task_id = task_allocator.get_task(process_id)
            except TreeFull:
                logger.debug(f"APV_MCTS[process {process_id}] Tree is full, exiting.")
                break
            except Exception as e:
                logger.error(f"APV_MCTS[process {process_id}] Error during task execution: {e}")
                raise e
        stats['end_time'] = time()
        stats['duration'] = stats['end_time'] - stats['start_time']
        logger.info("APV_MCTS[process %s] done. duration: %s; successes: %d, fails: %d", process_id, stats['duration'], stats['num_successes'], stats['num_fails'])

        if ADConfig.do_profiling:
            profiler.disable()
            prof_stats = pstats.Stats(profiler)
            prof_stats.sort_stats('cumulative')
            # print_mask_stats(actor._model._environment._action_space_storage)
            prof_stats.dump_stats(f'profile/apv_mcts_profile_{process_id}.prof')
            subprocess.run(['flameprof', '-i', f'profile/apv_mcts_profile_{process_id}.prof', '-o', f'profile/apv_mcts_flamegraph_{process_id}.svg'])
            logger.debug(f"APV_MCTS[process {process_id}] Profiling done, results saved.")
        
        return stats
