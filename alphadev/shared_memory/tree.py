from typing import Union, Callable, Dict, List, Optional
import multiprocessing.shared_memory as mp_shm
import numpy as np

from .base import BlockLayout, ArrayElement, AtomicCounterElement, BaseMemoryManager

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NodeBase(BlockLayout):
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
    _required_attributes = [
        'header', 'prior', 'R', 'W', 'N', 'mask', 'children', 'const_vl']
    hdr_parent = 0; hdr_action = 1; hdr_expanded = 2; hdr_terminal = 3
    _elements = {
        'header':   ArrayElement(np.int32,   (4,)), # parent_offset, action_id, terminal, expanded
        # other elements need to be defined in the subclass.
    }

    @classmethod
    def define(cls, width: int, vl_constant: float = 1.0):
        """
        Specify the elements of the node class.
        This method should be called in the subclass to define the node's attributes.
        """
        class Node(cls):
            _elements = cls._elements.copy()
            _elements.update({
                'prior':    ArrayElement(np.float32, (width,)),  # prior probabilities of actions
                'R':        ArrayElement(np.float32, (width,)),  # rewards for each action
                'W':        ArrayElement(np.float32, (width,)),  # total value for each action
                'N':        ArrayElement(np.int32,   (width,)),  # visit count for each action
                'mask':     ArrayElement(np.bool_,   (width,)),  # mask of valid actions
                'children': ArrayElement(np.int32,   (width,)),  # offsets of child nodes in the shared memory
            })
            const_vl = vl_constant  # constant virtual loss to apply during rollouts

        return Node

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
    def parent(self) -> 'NodeBase':
        assert not self.is_root, "Root has no parent."
        return self._get_parent()

    def set_child(self, action_id, child_offset):
        """Set the child offset for the given action_id."""
        logger.debug('%s.set_child action_id %s child_offset %s. current child offset: %s', repr(self), action_id, child_offset, self.children[action_id])
        self.children[action_id] = child_offset

    def set_legal_actions(self, legal_actions):
        self.mask[:] = legal_actions

    def set_prior(self, prior):
        self.prior[:] = prior

    def set_reward(self, reward):
        logger.debug('%s.set_reward called with reward %s.', repr(self), reward)
        self.parent.R[self.action_id] = reward

    def __repr__(self):
        return f'{self.__class__.__name__}(offset={self.offset}, action_id={self.action_id})'

class SharedTreeHeaderBase(BlockLayout):
    _required_attributes = ['index', 'root_offset', 'available']
    _elements = {
        'index': AtomicCounterElement(),  # index of the next block to be used
        'root_offset': ArrayElement(np.int32, ()),  # offset of the root node in the shared memory
    }
    @classmethod
    def define(cls, length: int):
        class SharedTreeHeader(cls):
            _elements = cls._elements.copy()
            # enumeration of blocks that are available for writing
            # at tree initialization.
            # `index` is used to index this array.
            _elements['available'] = ArrayElement(np.int32, (length,))
        return SharedTreeHeader

class TreeFull(Exception): pass

class SharedTree(BaseMemoryManager):
    """
    Shared Memory which stores a tree structure.

    Maintains two shared memory blocks:
    1. header block stores the index of the next block to be used.
    2. data block stores the nodes of the tree
    """
    def __init__(self, num_nodes:int, node_width:int, vl_constant:float, name: str = 'SharedTree'):
        """
        Initialize the shared tree with the given number of nodes and node width.
        
        :param num_nodes: number of nodes in the tree, including the root.
        :param node_width: maximum number of children per node, i.e. number of actions in the environment.
        :param vl_constant: constant virtual loss to apply during rollouts.
        """
        self.num_nodes = num_nodes
        self.num_blocks = self.num_nodes * 2 # double the size so we can also retain the tree from earlier searches.
        self._header_cls = SharedTreeHeaderBase.define(length=self.num_blocks)
        self._node_cls = NodeBase.define(
            width=node_width,  # number of actions in the environment
            vl_constant=vl_constant  # constant virtual loss to apply during rollouts
        )
        # declare shared memory regions
        self._header_size = self._header_cls.get_block_size()
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

        self.header = self._header_cls(self._header_shm, 0)
        with self.header.index() as index_counter:
            self._local_write_head = index_counter.load() # get the current write head

        self.reset()  # clear the header and input/output blocks

    def reset(self, last_action: Optional[int] = None):
        assert self._is_main, "reset() should only be called in the main process."
        logger.debug('SharedTree.reset called. last_action: %s', last_action)
        if last_action is not None:
            prev_root = self.get_root()
            logger.debug('SharedTree.reset: previous root is %s. children %s', repr(prev_root), [(i,c) for i,c in enumerate(prev_root.children)])
            new_root_offset = prev_root.children[last_action]
            if new_root_offset == 0:
                new_root = None # do not retain the subtree
                logger.debug("SharedTree.reset: last_action %s was never visited. defaulting to empty tree.", last_action)
            else:
                new_root = self.get_by_offset(new_root_offset)
                logger.debug('SharedTree.reset: last_action %s, prev_root %s.', last_action, repr(prev_root))
        else:
            new_root = None
        # reset the header's counter and available array.
        self._header_cls.clear_block(self._header_shm, 0)
        logger.debug('SharedTree.reset: cleared header block. index: %s, root_offset: %s, available: %s.', 
                     self.header.index.peek(), self.header.root_offset, self.header.available)
        
        # helper array to see which indices are available
        available_mask = np.ones(self.num_blocks, dtype=np.bool_)

        if new_root is not None:
            logger.debug('SharedTree.reset: retaining subtree rooted at %s.', repr(new_root))
            # configure the new root node's offset
            self.header.root_offset[...] = new_root.offset
            root = self.get_root()
            assert root.offset == new_root.offset, \
                f"Expected root offset {new_root.offset}, got {root.offset}."
            
            # set the available indices to Falsse for the subtree of the last root.
            frontier = [root]
            while frontier:
                # BFS to traverse the tree
                node = frontier.pop(0)
                # the index corresponding to this node is not available
                available_mask[self.off2i(node.offset)] = False
                logger.debug('SharedTree.reset: node %s at offset %s is not available for writing.', repr(node), node.offset)
                for child_offset in node.children:
                    if child_offset != 0: # i.e. is initialized
                        logger.debug('SharedTree.reset: child offset %s is valid, adding to frontier.', child_offset)
                        child = self.get_by_offset(child_offset)
                        frontier.append(child)
        else:
            logger.debug('SharedTree.reset: no last action provided, setting root to a new node at offset 0.')
            root: NodeBase = self.get_root() # node at offset 0
            available_mask[self.off2i(root.offset)] = False  # root is never available for writing.
        # set the root node's header
        root.set_root()
        logger.debug('SharedTree.reset: available_mask after retaining subtree: %s', [(i,a) for i, a in enumerate(available_mask)])
        # clear all nodes that are available for writing.
        for i in range(self.num_blocks):
            if available_mask[i]:
                logger.debug('SharedTree.reset: clearing block at index %d (offset %d).', i, self.i2off(i))
                self._node_cls.clear_block(self._data_shm, self.i2off(i))
        # set the available array in the header
        available_indices = np.arange(self.num_blocks)[available_mask]
        logger.debug('SharedTree.reset: available indices: %s', available_indices)
        assert available_indices.shape[0] >= self.num_nodes, \
            f"Expected at least {self.num_nodes} available indices, got {available_indices.shape[0]}."
        self.header.available[:available_indices.shape[0]] = available_indices
        logger.debug('SharedTree.reset: header available array set to %s.', [(i,a) for i, a in enumerate(self.header.available)])
        # this array can now be used to allocate new nodes.
        # sanity check
        assert new_root is None or root.offset != 0, \
            "SharedTree.reset: last_action %s was visited before, but root is at offset %s. (should not be 0)" % (last_action, root.offset)

    def attach(self):
        logger.debug('SharedTree.attach')
        self._is_main = False
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=False, size=self._data_size)

        self.header = self._header_cls(self._header_shm, 0)
    
    def init_index(self):
        self._local_write_head = self._update_index() # obtain a write location

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
            logger.debug('SharedTree._update_index (nb %d) prev: %s crnt: %d', self.num_blocks, self._local_write_head, self.header.available[index])
        if index >= self.num_nodes:
            self._local_write_head = None # we are done
            return None
        else:
            self._local_write_head = self.header.available[index]
            return self._local_write_head

    def fail(self):
        """We didn't manage to write, try again next time to the same block."""
        pass

    def succeed(self):
        """We successfully wrote to the block, we can find the next block."""
        self._local_write_head = self._update_index()

    def append_child(self, parent: NodeBase, action: int) -> Optional[NodeBase]:
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
        
        if self._local_write_head is None:
            # we are done, no more blocks available.
            logger.debug('SharedTree.append_child: no more blocks available, returning None.')
            self.fail()
            return None
        
        child_offset = self.i2off(self._local_write_head)
        parent.set_child(action, child_offset)
        logger.debug('SharedTree.append_child: parent %s children[%d] set to %d.', repr(parent), action, child_offset)
        # now write the parent offset and action id to the child node's header
        child: NodeBase = self._node_cls(self._data_shm, child_offset)
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
        logger.debug('SharedTree.i2off called with index %d node size %d.', index, self._node_size)
        return index * self._node_size
    def off2i(self, offset: int) -> int:
        logger.debug('SharedTree.off2i called with offset %d node size %d.', offset, self._node_size)
        if offset % self._node_size != 0:
            raise ValueError(f"Offset {offset} is not a multiple of node size {self._node_size}.")
        return offset // self._node_size

    def get_node(self, index) -> NodeBase:
        """Return the root node, at offset 0."""
        if index == 0:
            return self.get_root()
        return self._node_cls(self._data_shm, self.i2off(index))

    def get_root(self):
        return self._node_cls(self._data_shm, self.header.root_offset.item())

    def get_by_offset(self, offset: int) -> NodeBase:
        return self._node_cls(self._data_shm, offset)

    def is_full(self) -> bool:
        """
        Check if the tree is full.
        The tree is full if the index is equal to the number of blocks.
        """
        return self.header.index.peek() >= self.num_nodes

class SharedTreeFactory:
    """
    Factory pattern for creating SharedTree instances.
    """
    def __init__(self,
        num_nodes,
        width,
        vl_const,
        name,
    ):
        self._num_nodes = num_nodes
        self._width = width
        self._vl_const = vl_const
        self._name = name
    def __call__(self) -> SharedTree:
        """
        Create an instance of the shared tree.
        """
        return SharedTree(
            num_nodes=self._num_nodes,
            node_width=self._width,
            vl_constant=self._vl_const,
            name=self._name
        )

