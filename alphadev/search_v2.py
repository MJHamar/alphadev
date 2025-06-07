"""
Implementation of APV-MCTS
"""
from typing import NamedTuple, Dict, Union, Sequence
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

import logging
logger = logging.getLogger(__name__)

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
        'header': ArrayElement(np.int32,   (4,)          ), # parent_offset, action_id, terminal, expanded
        'prior':  ArrayElement(np.float32, (_num_actions,)), # prior probabilities of actions
        'R':      ArrayElement(np.float32, (_num_actions,)), # One-time reward observed at this node
        'W':      ArrayElement(np.float32, (_num_actions,)), # predicted value + empirical discounted return
        'N':      ArrayElement(np.int32,   (_num_actions,)), # number of visits to the node
        'mask':   ArrayElement(np.bool_,   (_num_actions,)), # mask of valid actions leading to this node
    }
    # virtual loss constant
    const_vl = -1.0
    
    def __init__(self, shm, offset):
        super().__init__(shm, offset)
    
    def expand(self, prior):
        self.prior = prior
        self.header[self.__class__.hdr_expanded] = True
    
    def select(self, action_id):
        """Increment W by const_vl and N by 1 for the given action_id."""
        self.W[action_id] += self.const_vl
        self.N[action_id] += 1
    def deselect(self, action_id, reward=0.0):
        """Inverse operation of select."""
        self.W[action_id] += -self.const_vl + reward
        self.N[action_id] -= 1
    def visit_child(self, action_id, value):
        """W = W - const_vl + value; N = N - 1 + 1 for the given action_id."""
        self.W[action_id] += -self.const_vl + value
        # N = N - 1 + 1 -> N = N, so this is a no-op
    def update_child(self, action_id, value):
        """Update the child without touching the virtual loss."""
        self.W[action_id] += value
        self.N[action_id] += 1
    
    def is_consistent(self):
        return self.parent.children[self.action_id] == self.offset
    
    def children_values(self):
        """To be called during search for selecting children. considers the virtual loss."""
        return np.where(
            self.mask & (self.N != 0),
            np.divide(self.W, self.N),
            0.0
        )
    
    def children_visits(self):
        """To be called during search for selecting children. considers the virtual loss."""
        return self.N
    
    @property
    def parent(self) -> 'Node':
        assert not self.is_root, "Trying to access the parent of the Root!"
        return self.__class__(self.shm, self.parent_offset)
    @property
    def value(self):
        """To be called during backprop, ignores the virtual loss."""
        parent = self.parent # instantiate once
        return (
            parent.W[self.action_id] / parent.N[self.action_id]
            ) if parent.N[self.action_id] > 0 else 0.0
    @property
    def visit_count(self):
        """To be called during backprop, ignores the virtual loss."""
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
    def is_root(self):       return self.header[self.__class__.hdr_action] == -1

    def set_root(self):
        """Set this node as the root node."""
        self.header[self.__class__.hdr_parent] = -1
        self.header[self.__class__.hdr_action] = -1
    
    def set_parent(self, parent_offset, action):
        self.header[self.__class__.hdr_parent] = parent_offset
        self.header[self.__class__.hdr_action] = action
    
    def set_legal_actions(self, legal_actions):
        self.mask[:] = legal_actions
    
    def set_reward(self, reward):
        self.parent.R[self.action_id] = reward


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
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._header = SharedTreeHeader(self._header_shm, 0)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=True, size=self._data_size)
        
        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        SharedTreeHeader.clear_block(self._header_shm, 0, length=self.num_blocks)
        with self._header.index() as idx:
            idx.store(1) # do not write to the root.
        for i in range(self.num_blocks):
            self._node_cls.clear_block(self._data_shm, self.i2off(i))
        # initialize the root node
        root: Node = self.get_root()
        root.set_root()
    
    def attach(self):
        self._is_main = False
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=False, size=self._data_size)
    
    def __del__(self):
        if self._is_main:
            # only the main process should delete the shared memory
            self._header_shm.close()
            self._header_shm.unlink()
            self._data_shm.close()
            self._data_shm.unlink()
    
    def _update_index(self):
        with self._header.index() as index_counter:
            index = index_counter.fetch_inc()
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
    
    def append_child(self, parent_offset: int, action: int) -> Union[Node, None]:
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
        if self.is_full():
            raise TreeFull() # signal the caller that no more blocks are available.
        parent = self.get_by_offset(parent_offset)
        if parent.children[action] != 0:
            # another process already appended this action.
            self.fail()
            return None
        
        child_offset = self.i2off(self._local_write_head)
        parent.children[action] = child_offset
        # now write the parent offset and action id to the child node's header
        child: Node = self._node_cls(self._data_shm, child_offset)
        child.set_parent(parent_offset, action)
        # now check if the write was successful
        if parent.children[action] == child_offset:
            self.succeed()
            return child
        else:
            self.fail()
            return None
    
    def i2off(self, index: int) -> int:
        return index * self._node_size
    
    def get_node(self, index) -> Node:
        """Return the root node, at offset 0."""
        if index == 0:
            return self.root
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
        with self._header.index() as idx:
            return idx.load() >= self.num_blocks

class TaskAllocatorHeader(BlockLayout):
    _elements = {
        'process_task': ArrayElement(np.int32, (ADConfig.num_actors,)),  # list of tasks assigned to each process
    }
class SharedTaskAllocator(BaseMemoryManager):
    def __init__(self):
        self._header_size = TaskAllocatorHeader.get_block_size()
        self._header_name = 'task_allocator_header'
        self._header_shm = None
        self._header = None
        self._is_main = False
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._header = TaskAllocatorHeader(self._header_shm, 0)
        self.reset()
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        TaskAllocatorHeader.clear_block(self._header_shm, 0)

    def attach(self):
        self._is_main = False
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._header = TaskAllocatorHeader(self._header_shm, 0)
    
    def __del__(self):
        if self._is_main:
            # only the main process should delete the shared memory
            self._header_shm.close()
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
        self._header.process_task[:] = task_ids
    
    def get_task(self, process_id: int) -> int:
        return self._header.process_task[process_id]

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
        self.tree = SharedTree(num_blocks=num_simulations, node_cls=node_class, name=f'{name}.tree')
        # TODO: this could be optional; and each process can have its own network instance instead.
        self.inference_buffer = AlphaDevInferenceService(
            num_blocks=self.inference_buffer_size,
            network_factory=network_factory,
            batch_size=batch_size,
            factory_args=network_factory_args,
            factory_kwargs=network_factory_kwargs,
            name=f'{name}.inference'
        )
        self.task_allocator = SharedTaskAllocator()
        self._init_task_allocation = \
            [APV_MCTS.SEARCH_SIM_TASK] * (ADConfig.num_actors - 1) + [APV_MCTS.BACKUP_TASK] # last actor is the one listening to the evaluator
        
        # TODO: make optional
        self.initialized = False
        self.inference_process = mp.Process(target=self._init_inference, args=(inference_device_config,), name=f'{name}.inf_proc')
        self._init_main()
        self.inference_process.start()
        self.actor_pool = mp.Pool(processes=ADConfig.num_actors)
        print(f"APV_MCTS: Inference process started with PID {self.inference_process.pid}.")
        print("Finished initialization.")
    
    def search(self, observation):
        """
        Perform APV-MCTS search on the given observation.
        This method initializes the search tree with root corresponding to the given observation,
        and runs num_simulations rollouts from the root node using a pool of worker processes.
        Finally, it returns a pointer to the root node, which can be used to perform post-mcts action selection.
        """
        # TODO: support only partially resetting the tree.
        print('APV_MCTS[main process] Starting search simulation phase.')
        self.tree.reset()
        
        self.inference_buffer.submit(node_offset=0, observation=observation)
        
        print('APV_MCTS[main process] Waiting for inference service to return prior and value estimate.')
        with self.inference_buffer.poll_ready() as inference_result:
            _, prior, _ = inference_result
        print('APV_MCTS[main process] Received prior and value estimate from inference service.')
        # Add exploration noise to the prior.
        noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
        prior = prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
        
        legal_actions = self.model.legal_actions()
        
        self.root: Node = self.tree.get_root()
        self.root.set_legal_actions(legal_actions)
        
        self.root.expand(prior)
        print('APV_MCTS[main process] Root node initialized with prior and legal actions. Starting pool')
        # all that is left is to start the pool.
        statistics = self.actor_pool.map(self._run_task, range(ADConfig.num_actors))
        for o in self.observers:
            o.update(statistics, self.root, self.tree, self.inference_buffer)
        
        return self.root  # return the root node, which now contains the search results

    def _init_main(self):
        """
        Initialize the root node of the search tree.
        The root node is a special node that is not expanded and has no parent.
        It is created in the shared memory and its attributes are set to zero.
        """
        logger.info("APV_MCTS[main process] Initializing the shared tree and inference service.")
        # we can also test the inference service here 
        # TODO: handle case where inference is not a process
        self.task_allocator.configure()
        self.task_allocator.allocate(self._init_task_allocation)
        self.inference_buffer.configure()
        self.tree.configure()
        self.initialized = True
    
    def _init_actor(self):
        """To be called from a subprocess"""
        logger.info("APV_MCTS[actor process] Initializing the actor process.")
        self.task_allocator.attach()
        self.tree.attach()
        self.inference_buffer.attach()
        self.initialized = True
    
    def _init_inference(self, device_config):
        """To be called from a subprocess"""
        logger.info("APV_MCTS[inference process] Initializing the inference service.")
        # TODO: setup tensorflow device config.
        self.inference_buffer.attach()
        self.initialized = True
        # run indefinitely
        self.inference_buffer.run()

    def _phase_1(self):
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
        print('APV_MCTS[phase 1] Starting search simulation phase.')
        node = self.root
        actions = []; trajectory = []
        while node.expanded:
            action = self.search_policy(node)
            node = node.select(action) # increment virtual loss
            actions.append(action)
            trajectory.append(node)  # keep track of the trajectory for backpropagation
        
        # before simulating, initialize the new node.
        new_node: Node = tree.append_child(trajectory[-2], actions[-1]) # pass the path to make sure only one block is reserved for this node.
        if new_node is None:
            # this node is already being evaluated by some other process.
            for n, a in zip(trajectory[:-1], actions):
                n.deselect(a)
            return False  # we need to try again, the node was overwritten by another process
        
        # then run the simulation update the node with the results. This delay also enables other processes to write stuff
        timestep = self.model.step(actions, update=False)
        legal_actions = self.model.legal_actions()
        # now we can check for tree consistency (whether any other process has overwritten the node)
        if not new_node.is_consistent():
            for n, a in zip(trajectory[:-1], actions):
                n.deselect(a)
            return False # we need to try again, the node was overwritten by another process
        
        # otherwise, submit a new inference task to the inference service.
        observation = timestep.observation
        self.inference_buffer.submit(
            node_offset=new_node.offset,
            observation=observation
        )
        # perform a backward pass with the reward and legal actions
        reward = timestep.reward[0] # there might be other outputs but they are irrelevant for mcts
        terminal = timestep.final()
        
        new_node.set_terminal(terminal)
        new_node.set_legal_actions(legal_actions)
        new_node.set_reward(reward)

        # backpropagate the reward for each t < L and remove the virtual loss.
        reward = self.discount * reward  # discount the reward for the backpropagation
        for n, a in zip(trajectory[:-1], actions):
            n.visit_child(a, reward)
            reward *= self.discount
        
        return True  # we successfully submitted the task for evaluation

    def _phase_2(self):
        """
        Process the predictions of the network.
        - Poll the inference buffer for a result.
        - Update the prior and value of the node with the result.
        - Backpropagate the value estimate to the root node.
        """
        print('APV_MCTS[phase 2] Starting backup phase.')
        with self.inference_buffer.poll_ready(timeout=10) as result:
            if result is None: 
                if self.tree.is_full():
                    raise TreeFull()
                else:
                    raise RuntimeError("No inference results for 10 seconds and tree is not full. Check.")
            node_offset = result.node_offset
            prior = result.prior
            value = result.value
        # obtain the node from the shared tree and check for consistency.
        node = self.tree.get_by_offset(node_offset)
        if not node.is_consistent():
            # the node was overwritten by another process, nothing to do here.
            return False
        # expand the current node
        node.expand(prior) # reward, legal_actions and terminal are already set
        # backpropagate without touching the virtual loss.
        while not node.is_root:
            parent = node.parent
            # update the parent with the value estimate and visit count
            parent.update_child(node.action_id, value)
            # then move to the parent node and discount the value
            node = parent
            value *= self.discount
        return True

    def _run_task(self, process_id: int):
        if not self.initialized:
            self._init_actor()
        my_task_id = self.task_allocator.get_task(process_id)
        print(f'APV_MCTS[actor {process_id}] Starting task execution my_task_id({my_task_id})')
        my_task = self._phase_1 if my_task_id == APV_MCTS.SEARCH_SIM_TASK else self._phase_2
        
        try:
            my_task()
        except TreeFull:
            pass # if tree is full do not schedule anything.
