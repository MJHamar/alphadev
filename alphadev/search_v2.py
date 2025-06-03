"""
Implementation of APV-MCTS
"""
from typing import NamedTuple, Dict, Union
import numpy as np
from multiprocessing import shared_memory, Lock, Process
from collections import namedtuple
import contextlib
import tree

from .environment import AssemblyGame
from .config import ADConfig


# Shared memory has a fixed size. for each MCTS, we do not need to re-allocate it
# but we do need to re-initialize it.
# so that it is with 100% certainty that addresses we didn't touch are zeroed out.
# hence no need to worry about whether a node is being created or is already there.

class ArrayElement(NamedTuple):
    dtype: np.dtype
    shape: tuple
    def size(self, *args, **kwargs):
        return np.dtype(self.dtype).itemsize * np.prod(self.shape)
    def create(self, shm, offset, *args, **kwargs):
        return np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf, offset=offset)

class VarLenArrayElement(ArrayElement):
    def size(self, length, *args, **kwargs):
        return np.dtype(self.dtype).itemsize * np.prod((length, *self.shape))
    def create(self, shm, offset, length, *args, **kwargs):
        return np.ndarray((length, *self.shape), dtype=self.dtype, buffer=shm.buf, offset=offset)

class NestedArrayElement(ArrayElement):
    model: Union[Dict, NamedTuple]
    def size(self, *args, **kwargs):
        return sum(element.nbytes for element in self.model.values())
    def create(self, shm, offset, *args, **kwargs):
        elements = {}
        crnt_offset = offset
        for name, element in self.model.items():
            elements[name] = np.ndarray(element.shape, dtype=element.dtype, buffer=shm.buf, offset=crnt_offset)
            crnt_offset += element.nbytes
        if isinstance(self.model, dict):
            return elements
        elif isinstance(self.model, NamedTuple):
            return self.model._make(**elements)

class BlockLayout:
    _elements: Dict[ArrayElement] = {}
    
    def __init__(self, shm, offset, *args, **kwargs):
        self.shm = shm
        self.offset = offset
        self._create_elements(*args, **kwargs)
    
    def _create_elements(self, *args, **kwargs):
        crnt_offset = self.offset
        for name, element_spec in self.__class__._elements.items():
            setattr(self, name, element_spec.create(self.shm, crnt_offset, *args, **kwargs))
            crnt_offset += element_spec.size()
    
    @classmethod
    def clear_block(cls, shm, offset):
        """
        Initialize a block of shared memory at the given offset.
        """
        for element_spec in cls._elements.values():
            element = element_spec.create(shm, offset)
            tree.map_structure(lambda x: x.fill(0), element)

    @classmethod
    def get_block_size(cls, *args, **kwargs):
        """
        Get the size of the block in bytes.
        """
        return sum(element.size(*args, **kwargs) for element in cls._elements.values())

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

class InferenceTask(BlockLayout):
    _elements = {
        'node_offset': ArrayElement(np.int32, ()),  # offset of the node in the shared memory, needed for expansion/backpropagation
        'observation': NestedArrayElement(model=AssemblyGame(ADConfig.task_spec).observation_spec()),
    }

class InferenceResult(BlockLayout):
    _elements = {
        'node_offset': ArrayElement(np.int32, ()),  # offset of the node in the shared memory, needed for expansion/backpropagation
        'prior': ArrayElement(np.float32, (ADConfig.task_spec.num_actions,)),  # prior probabilities of actions
        'value': ArrayElement(np.float32, ()),  # **SCALAR** value estimate of the node
    }

class InferenceBuffer:
    pass



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
        self.root.expand(prior, value, legal_actions)
    
    def _select(self):
        node = self.root
        actions = []
        while node.expanded:
            action = self.search_policy(node)
            actions.append(action)
            node = node.select(action) # increment virtual loss
        # before simulating, reserve a new block for this node.
        block_offset = self._get_next_block(actions) # pass the path to make sure only one block is reserved for this node.
        if not block_offset:
            return # nothing to do.
        # otherwise, run the simulation and schedule an evaluation
        timestep = self.model.step(actions, update=False)
        legal_actions = self.model.legal_actions()
        observation = timestep.observation
        reward = timestep.reward
        terminal = timestep.final()
        with self.inference_buffer.next() as inference_offset:
            # TODO.
            inference_task = InferenceTask(
                self.inference_buffer, inference_offset, block_offset,
                observation, reward, terminal, actions, legal_actions
            )
        
