"""
Implementation of APV-MCTS
"""
from typing import NamedTuple, Dict, Union, Sequence
from time import time

import tensorflow as tf
import numpy as np
import multiprocessing as mp
from collections import namedtuple
import contextlib
import tree

from .environment import AssemblyGame, AssemblyGameModel
from .inference_service import InferenceTask, InferenceResult, AlphaDevInferenceService
from .shared_memory import *
from .config import ADConfig


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
        'vl':     ArrayElement(np.int32,   (_num_actions,)), # virtual loss multiplier to discourage visiting this node
        'mask':   ArrayElement(np.bool_,   (_num_actions,)), # mask of valid actions leading to this node
    }
    # virtual loss constant
    const_vl = -1.0
    
    def __init__(self, shm, offset):
        super().__init__(shm, offset)
    
    def expand(self, prior, value):
        self.parent.W[self.action_id] = value
        self.prior = prior
        self.header[self.__class__.hdr_expanded] = True
    
    def visit(self, value):
        parent = self.parent; action_id = self.action_id
        parent.W[action_id] += value
        parent.N += 1
        parent.deselect(action_id)
    
    def select(self, action_id):
        self.vl[action_id] += 1 # if self.vl[action_id] > 0 else 1
    def deselect(self, action_id):
        self.vl[action_id] -= 1 if self.vl[action_id] > 0 else 0
    
    def children_values(self):
        """To be called during search for selecting children. considers the virtual loss."""
        denominator = self.N + self.vl
        return np.where(
            self.mask & (denominator != 0),
            np.divide(self.W + self.const_vl * self.vl, denominator),
            0.0
        )
    
    def children_visits(self):
        """To be called during search for selecting children. considers the virtual loss."""
        return np.where(self.mask, self.N + self.vl, 0)
    
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
        self.header[self.__class__.hdr_parent] = -1
        self.header[self.__class__.hdr_action] = -1
        self.header[self.__class__.hdr_terminal] = False
    
    def set_parent(self, parent_offset, action):
        self.header[self.__class__.hdr_parent] = parent_offset
        self.header[self.__class__.hdr_action] = action
    
    def set_legal_actions(self, legal_actions):
        self.mask[:] = legal_actions
    
    def set_reward(self, reward):
        self.parent.R[self.action_id] = reward

class SharedTreeHeader(BlockLayout):
    _elements = {
        'index': ArrayElement(np.int32, ()),  # index of the next block to be used
    }

class SharedTree:
    """
    Shared Memory which stores a tree structure.
    
    Maintains two shared memory blocks:
    1. data block stores the nodes of the tree
    2. paths block mirrors the tree structure and is used to allocate new nodes to requesting processes
        without locking the entire tree.
        The process is as follows:
        - requesting process calls `get_next_block(path)` with the path to the node (sequence of actions)
        - the process is given a block index, and the goal is to take this block index and write it as a child to the parent.
        - check if the write was successful. if yes, write the process's id to the new index in the paths block.
            the process who wins this two-stage game wins the right to write to the new node. all other processes fail and have to restart.
            this way we both eliminate the issue with two different nodes being written to the same location
            and vice versa, we eliminate the issue with having the same node duplicated in different memory locations.
            we still can have the issue that two competing processes assign the same index to different nodes.
            for now, we overcome this by maintaining a lock for the indexing pointer.
            NOTE: We can also overcome this by writing the process' id to an index in a separate num_blocks sized array.
    """
    def __init__(self, num_blocks, node_cls: Node, name: str = 'SharedTree'):
        self.num_blocks = num_blocks
        self._node_cls = node_cls
        # declare shared memory regions
        self._header_size = SharedTreeHeader.get_block_size(length=num_blocks)
        self._header_name = f'{name}_header'
        
        self._data_size = num_blocks * node_cls.get_block_size()
        self._data_name = f'{name}_data'
        
        # shared between processes
        self._lock = mp.Lock()  # to ensure atomicity of index updates
        # process-local
        self._is_main = False
        self._local_write_head = None
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp.shared_memory.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._header = SharedTreeHeader(self._header_shm, 0)
        self._data_shm = mp.shared_memory.SharedMemory(name=self._data_name, create=True, size=self._data_size)
        
        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        SharedTreeHeader.clear_block(self._header_shm, 0, length=self.num_blocks)
        for i in range(self.num_blocks):
            self._node_cls.clear_block(self._data_shm, self._header_size + i * self._node_cls.get_block_size())
    
    def attach(self):
        self._header_shm = mp.shared_memory.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._data_shm = mp.shared_memory.SharedMemory(name=self._data_name, create=False, size=self._data_size)
    
    def _update_index(self):
        with self._lock:
            index = self._header.index[0]
            if index >= self.num_blocks:
                self._local_write_head = None # we are done
            else:
                self._local_write_head = index
                self._header.index[0] += 1
        return self._local_write_head
    
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
        child_offset = self._header_size + self._local_write_head * self._data_size
        parent = self._node_cls(self._data_shm, parent_offset)
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
        
    def get_root(self) -> Node:
        """Return the root node, at offset 0."""
        return self._node_cls(self._data_shm, 0)

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
        
        # TODO: make optional
        self.actor_pool = mp.Pool(processes=ADConfig.num_actors, initializer=self._init_actor)
        self._init_main()
        self.inference_process = mp.Process(target=self._init_inference, args=(inference_device_config,), name=f'{name}.inf_proc')
        self.inference_process.run()
    
    def _init_main(self):
        """
        Initialize the root node of the search tree.
        The root node is a special node that is not expanded and has no parent.
        It is created in the shared memory and its attributes are set to zero.
        """
        # we can also test the inference service here 
        # TODO: handle case where inference is not a process
        self.inference_buffer.configure()
        self.tree.configure()
        self.inference_process.run() # start the inference process here.
    
    def search(self, observation):
        """
        Perform APV-MCTS search on the given observation.
        This method initializes the search tree with root corresponding to the given observation,
        and runs num_simulations rollouts from the root node using a pool of worker processes.
        Finally, it returns a pointer to the root node, which can be used to perform post-mcts action selection.
        """
        # TODO: support only partially resetting the tree.
        self.tree.reset()
        
        self.inference_buffer.submit(observation)
        
        with self.inference_buffer.poll_ready() as inference_result:
            prior, value = inference_result
        
        # Add exploration noise to the prior.
        noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
        prior = prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
        
        legal_actions = self.model.legal_actions()
        
        self.root: Node = self.tree.get_root()
        self.root.set_root()
        self.root.set_legal_actions(legal_actions)

        self.root.expand(prior, value)
        
        # all that is left is to start the pool.
        statistics = self.actor_pool.map(self._rollout, range(self.num_simulations))
        for o in self.observers:
            o.update(statistics, self.root, self.tree, self.inference_buffer)
        
        return self.root  # return the root node, which now contains the search results

    def _init_actor(self):
        """To be called from a subprocess"""
        self.tree.attach()
        self.inference_buffer.attach()
    
    def _init_inference(self):
        """To be called from a subprocess"""
        self.inference_buffer.attach()
        # run indefinitely
        self.inference_buffer.run()

    def _select_and_simulate(self):
        node = self.root; history = []  # to keep track of the path taken
        actions = []
        while node.expanded:
            action = self.search_policy(node)
            actions.append(action)
            history.append(node)
            node = node.select(action) # increment virtual loss
        # before simulating, initialize the new node.
        new_node: Node = tree.append_child(history[-1].offset, actions[-1]) # pass the path to make sure only one block is reserved for this node.
        if new_node is None:
            # decrement the virtual losses appended to the path
            for node, action in zip(history, actions):
                node.deselect(action)
            return False # Try again.
        # otherwise, run the simulation and schedule an evaluation
        # first set parent
        new_node.set_parent(history[-1].offset, actions[-1])
        # then run the simulation update the node with the results
        timestep = self.model.step(actions, update=False)
        new_node.set_legal_actions(self.model.legal_actions())
        new_node.set_reward(timestep.reward)
        new_node.set_terminal(timestep.final())
        
        observation = timestep.observation
        self.inference_buffer.submit(
            node_offset=new_node.offset,
            observation=observation
        )
        return True  # we successfully submitted the task for evaluation

    def _backpropagate(
        inference_buffer: IOBuffer,
        tree: SharedTree,
        discount: float = 1.0,
        timeout: float = None
        ):
        result: InferenceResult = inference_buffer.poll_ready(timeout=timeout)
        if not result:
            return False # timeout happenned, we need to try again.
        node_offset = result.node_offset
        prior = result.prior
        value = result.value
        
        node = tree.get_node(node_offset)
        node.expand(prior, value) # reward, legal_actions and terminal are already set

        ret = value
        while not node.is_root:
            node = node.parent
            ret *= discount
            ret += node.reward

            node.visit(ret)
        return True

    def _rollout(self, rollout_nr: int):
        """
        Run a single rollout from the root node.
        with evaluation phase done in a separate process.
        Return search statistics.
        """
        if self.local_write_head is None:
            return { # tree buffer is full.
                'rollout': rollout_nr,
                'selection_tries':0, 'backprop_tries': 0} 
        selection_tries = backprop_tries = 1
        while not self._select_and_simulate():
            selection_tries += 1
        while not self._backpropagate():
            backprop_tries += 1
        return {
            'rollout': rollout_nr,
            'selection_tries': selection_tries,
            'backprop_tries': backprop_tries,
        }
