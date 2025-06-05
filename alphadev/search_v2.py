"""
Implementation of APV-MCTS
"""
from typing import NamedTuple, Dict, Union, List
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
        parent.vl[action_id] -= 1 if parent.vl[action_id] > 0 else 0
        parent.W[action_id] += value
        parent.N += 1
    
    def children_values(self):
        """To be called during search for selecting children. considers the virtual loss."""
        virt_W = np.where(self.mask, self.W + self.const_vl * self.vl, 0.0)
        virt_N = np.where(self.mask, self.N + self.vl, 1)  # avoid division by zero
        return np.where(self.mask, virt_W / virt_N, 0.0)
    
    def children_visits(self):
        """To be called during search for selecting children. considers the virtual loss."""
        return np.where(self.mask, self.N + self.vl, 0)
    
    @property
    def parent(self) -> 'Node':
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
    def set_root(self):      self.header[self.__class__.hdr_action] = -1
    
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
        'index_pid': ArrayElement(np.int32, (ADConfig.num_actors)),  # buffer of failed attempts to write to the tree
        'active_indices': VarLenArrayElement(np.bool_, ()),  # indices of active processes
    }

class SharedTreePathsNode(BlockLayout):
    _elements = {
        'pid': ArrayElement(np.int32, ()),  # process id of the process that owns this node
        'children_offsets': ArrayElement(np.int32, (ADConfig.task_spec.num_actions,)),  # offsets of the children nodes in the shared memory
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
        self._header_size = SharedTreeHeader.get_block_size(length=num_blocks)
        self._header_name = f'{name}_header'
        
        self._paths_size = num_blocks * SharedTreePathsNode.get_block_size()
        self._paths_name = f'{name}_paths'
        
        self._data_size = num_blocks * node_cls.get_block_size()
        self._data_name = f'{name}_data'
        self._header = SharedTreeHeader(self._header_shm, 0, length=num_blocks)
        self._paths_root = SharedTreePathsNode(self._paths_shm, 0)
        self._data_root = node_cls(self._data_shm, 0)
        self._node_cls = node_cls
        self._is_main = False
        
        self._lock = mp.Lock()  # to ensure atomicity of index updates
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp.shared_memory.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._paths_shm = mp.shared_memory.SharedMemory(name=self._paths_name, create=True, size=self._paths_size)
        self._data_shm = mp.shared_memory.SharedMemory(name=self._data_name, create=True, size=self._data_size)
        
        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        SharedTreeHeader.clear_block(self._header_shm, 0, length=self.num_blocks)
        for i in range(self.num_blocks):
            SharedTreePathsNode.clear_block(self._paths_shm, self._header_size + i * SharedTreePathsNode.get_block_size())
            self._node_cls.clear_block(self._data_shm, self._header_size + i * self._node_cls.get_block_size())
    
    def attach(self):
        self._header_shm = mp.shared_memory.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._paths_shm = mp.shared_memory.SharedMemory(name=self._paths_name, create=False, size=self._paths_size)
        self._data_shm = mp.shared_memory.SharedMemory(name=self._data_name, create=False, size=self._data_size)
    
    def _update_index(self):
        pid = mp.current_process().pid
        if self.header.active_indices[pid]:
            return self.header.active_indices[pid]
        # if index is no longer active (it was used to write a node)
        # find the next available index
        with self._lock:
            index = self._header.index
            if index >= self.num_blocks:
                return -1  # no more blocks available
            self._header.index += 1
        self._header.index_pid[pid] = index
        return index
    
    def get_next_block(self, path:List[int]):
        index = self._update_index()
        crnt_node = self._paths_root
        for action in path[:-1]:
            crnt_node = SharedTreePathsNode(self._paths_shm, crnt_node.children_offsets[action])
        # now we are at the parent node, try to write the new node
        # first check if it exists
        if crnt_node.children_offsets[path[-1]] != 0:
            # node already exists, this attempt failed.
            return None
        # otherwise, reserve a new block for this node
        new_node_offset = self._header_size + index * SharedTreePathsNode.get_block_size()
        crnt_node.children_offsets[path[-1]] = new_node_offset
        # now write the process id to the new node
        new_node = SharedTreePathsNode(self._paths_shm, new_node_offset)
        pid = current_process().pid
        if crnt_node.children_offsets[path[-1]] != index:
            # this atttempt failed and we need to restart
            return None # TODO: distinguish
        new_node.pid = pid
        # check parent-node alignment
        if new_node_offset != crnt_node.children_offsets[path[-1]] or new_node.pid != pid:
            # this atttempt failed and we need to restart
            return None
        # otherwise, we successfully reserved a new block for this node
        # now we can return the new node offset
        return index

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
            name:str='apv-mcts' # needs to be specified is multiple instances are used.
        ):
        self.model = model # a (learned) model of the environment.
        self.search_policy = search_policy # a function that takes a node and returns an action to take.
        self.num_simulations = num_simulations  # number of simulations to run per search
        self.num_actions = num_actions  # number of actions in the environment [a noop, we know this from ADConfig already]
        self.discount = discount  # discount factor for the value estimate
        self.inference_buffer_size = inference_buffer_size or num_simulations  # size of the inference buffer

        # applying Dirichlet noise to the prior probabilities at the root.
        self.dirichlet_alpha = dirichlet_alpha  # alpha parameter for the Dirichlet noise
        self.exploration_fraction = exploration_fraction  # fraction of the prior to add Dirichlet noise to
        
        # declare shared memory ( no init )
        self.tree = SharedTree(num_blocks=num_simulations, node_cls=node_class)
        # TODO: this could be optional; and each process can have its own network instance instead.
        self.inference_buffer = AlphaDevInferenceService(
            num_blocks=self.inference_buffer_size,
            network_factory=network_factory,
            batch_size=batch_size,
            factory_args=network_factory_args,
            factory_kwargs=network_factory_kwargs,
            name=f'{name}.inference'
        )
        
        self._init_root()
        self.pool = mp.Pool(processes=ADConfig.num_actors)
    
    def __call__(observation):
        with contextlib.ExitStack() as stack:
            pass
    
    def _init_root(self):
        """
        Initialize the root node of the search tree.
        The root node is a special node that is not expanded and has no parent.
        It is created in the shared memory and its attributes are set to zero.
        """
        prior, value = self.evaluation_fn(self.observation)
        legal_actions = self.model.legal_actions()
        
        self.root = self.node_cls(self.shm, 0, self.num_actions)
        self.root.set_root()
        
        # Add exploration noise to the prior.
        noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
        prior = prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction

        self.root.expand(prior, value, legal_actions)

def _select_and_simulate(
    root: Node, search_policy, tree: SharedTree,
    inference_buffer: IOBuffer,
    model: AssemblyGameModel
    ):
    node = root; parent_node = None
    actions = []
    while node.expanded:
        action = search_policy(node)
        actions.append(action)
        parent_node = node
        node = node.select(action) # increment virtual loss
    # before simulating, reserve a new block for this node.
    new_node: Node = tree.get_next_block(actions) # pass the path to make sure only one block is reserved for this node.
    if not new_node:
        return # nothing to do.
    # otherwise, run the simulation and schedule an evaluation
    # first set parent
    new_node.set_parent(parent_node.offset, actions[-1])
    # then run the simulation update the node with the results
    timestep = model.step(actions, update=False)
    new_node.set_legal_actions(model.legal_actions())
    new_node.set_reward(timestep.reward)
    new_node.set_terminal(timestep.final())
    
    observation = timestep.observation
    inference_buffer.submit(
        node_offset=new_node.offset,
        observation=observation
    )

def _backpropagate(
    inference_buffer: IOBuffer,
    tree: SharedTree,
    discount: float = 1.0,
    timeout: float = None
    ):
    result: InferenceResult = inference_buffer.poll_ready(timeout=timeout)
    if not result:
        return
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

def _rollout(
    root: Node, search_policy, tree: SharedTree,
    inference_buffer: IOBuffer, model: AssemblyGameModel,
):
    """Run a single rollout from the root node.
    with evaluation phase done in a separate process.
    Return search statistics.
    """
    selection_tries = 0
    while not _select_and_simulate(
        root, search_policy, tree, inference_buffer, model):
        selection_tries += 1
    while not _backpropagate(
        inference_buffer, tree, discount=model.discount, timeout=0.1):
        backprop_tries += 1
    return {
        'selection_tries': selection_tries,
        'backprop_tries': backprop_tries,
    }

