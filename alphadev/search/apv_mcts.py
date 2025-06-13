from typing import Union, Callable, Dict, List, Optional
import multiprocessing.shared_memory as mp_shm
from time import time, sleep
import numpy as np

from acme.agents.tf.mcts.search import SearchPolicy
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts import models


from ..service.inference_service import AlphaDevInferenceClient
from ..shared_memory.base import BlockLayout, ArrayElement, AtomicCounterElement, BaseMemoryManager
from .mcts import MCTSBase, NodeBase

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SharedNodeBase(NodeBase, BlockLayout):
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
        'visitors', # keep track of the current number of visitors to this node.
        'header', 'mask', 'children', 'const_vl',
        'prior', 'R', 'Nr', 'W', 'Nw']
    hdr_parent = 0; hdr_action = 1; hdr_expanded = 2; hdr_terminal = 3
    _elements = {
        'header':   ArrayElement(np.int32,   (4,)), # parent_offset, action_id, terminal, expanded
        # other elements need to be defined in the subclass.
    }
    
    @classmethod
    def define(cls, width: int, num_workers:int, lambda_ = 0.5, vl_constant: float= -1.0) -> 'SharedNodeBase':
        """
        Specify the elements of the node class.
        This method should be called in the subclass to define the node's attributes.
        """
        node_cls = super().define(width=width, lambda_=lambda_)
        class SharedNode(node_cls):
            _elements = cls._elements.copy()
            _elements.update({
                'visitors': ArrayElement(np.bool_,   (width, num_workers,)),  # number of visitors to this node
                'prior':    ArrayElement(np.float32, (width,      )),  # prior probabilities of actions
                'R':        ArrayElement(np.float32, (width,      )),  # rewards for each action
                'W':        ArrayElement(np.float32, (width,      )),  # total value for each action
                'N':        ArrayElement(np.int32,   (width,      )),  # visit count for each action
                'mask':     ArrayElement(np.bool_,   (width,      )),  # mask of valid actions
                'children': ArrayElement(np.int32,   (width,      )),  # offsets of child nodes in the shared memory
            })
            const_vl = vl_constant  # constant virtual loss to apply during rollouts
        
        return SharedNode

    def __init__(self, shm, offset, parent=None, action=None):
        BlockLayout.__init__(self, shm, offset)
        if parent is not None and action is not None:
            self.set_parent(parent, action)
        self._parent = None
    
    # ---------------------
    # SHM-specific methods
    # ---------------------
    def select(self, action_id:int) -> int:
        """Set the visitor flag for the current process and return the child offset."""
        logger.debug('%s.select called with action_id %s. from proc. %d', repr(self), action_id, _local_id)
        assert _local_id is not None, "select() can only be called from a worker process. make sure to set _local_id before calling this method."
        
        self.visitors[action_id, _local_id] = True
        
        child_offset = self.children[action_id]
        return child_offset

    def deselect(self, action_id:int, recursive:bool=False) -> int:
        """Inverse operation of select."""
        logger.debug('%s.deselect called with action_id %s.', repr(self), action_id)
        self.visitors[action_id, _local_id] = False
        if recursive and not self.is_root():
            self.parent.deselect(self.action_id, recursive=True)
        return self.parent_offset
    
    def is_consistent(self):
        logger.debug('%s.is_consistent called.', repr(self))
        if self.is_root:
            return True
        return self.parent.children[self.action_id] == self.offset
    
    @property
    def parent_offset(self): return self.header[self.__class__.hdr_parent]
    @property
    def action_id(self):     return self.header[self.__class__.hdr_action]

    # -----------------------------
    # NodeBase interface overrides
    # -----------------------------
    
    def expand(self, prior):
        self.prior[...] = prior
        self.header[self.__class__.hdr_expanded] = True
    
    # NOTE: we don't deselect the action in backup_value.
    # it is expected that backup_reward is called first.
    
    def backup_reward(self, action: types.Action, reward: float, discount: float, trajectory: Optional[List['NodeBase']] = []):
        """Backup reward and also deselect."""
        self.deselect(action)
        return super().backup_reward(action, reward, discount, trajectory)
    
    @property
    def children_values(self) -> np.ndarray:
        vl_counts = self.visitors.sum(axis=1)
        Nr_vl = self.Nr + vl_counts
        values_r = np.divide(self.R + self.const_vl * vl_counts, Nr_vl, self.zeros, where=Nr_vl != 0)
        values_w = np.divide(self.W, self.Nw, self.zeros, where=self.Nw != 0)
        return (1 - self._lam) * values_r + self._lam * values_w
    
    @property
    def children_visits(self) -> np.ndarray:
        return self.Nr + self.visitors.sum(axis=1)
    
    @property
    def parent(self) -> 'SharedNodeBase':
        if self.is_root(): return None
        if self._parent is None:  # lazy load the parent node
            self._parent = self.__class__(self.shm, self.parent_offset)
        return self._get_parent()
    @property
    def action(self) -> types.Action:
        if self.is_root(): return None
        return self.header[self.__class__.hdr_action]
    @property
    def expanded(self) -> bool: return self.header[self.__class__.hdr_expanded] != 0
    @property
    def terminal(self) -> bool: return self.header[self.__class__.hdr_terminal] != 0
    
    def is_root(self) -> bool:
        """Check if this node is the root node."""
        return self.header[self.__class__.hdr_parent] == -1
    
    def set_child(self, action: types.Action, child: 'SharedNodeBase'):
        self.children[action] = child.offset
    
    def set_root(self):
        """Set this node as the root node."""
        self.header[self.__class__.hdr_parent] = -1
        self.header[self.__class__.hdr_action] = -1
        self._parent = None
    
    def set_parent(self, parent:'SharedNodeBase', action: types.Action):
        self.header[self.__class__.hdr_parent] = parent.offset
        self.header[self.__class__.hdr_action] = action
        self._parent = parent
    
    def set_terminal(self, terminal): self.header[self.__class__.hdr_terminal] = terminal

    def get_visit_count(self, action: Optional[types.Action]=None) -> int:
        if action is None: self.Nr.sum() + self.visitors.sum()
        return self.Nr[action] + self.visitors[action].sum()
    def get_reward(self, action: types.Action) -> float:
        if self.is_root(): return 0.0
        Nr_vl = self.Nr[action] + self.visitors[action].sum()
        if Nr_vl == 0: return 0.0
        return self.R[action] + self.const_vl * self.visitors[action].sum() / Nr_vl
    
    def __repr__(self):
        return f'{self.__class__.__name__}(offset={self.offset}, action_id={self.action_id})'

class SharedTreeHeaderBase(BlockLayout):
    _required_attributes = ['root_offset', 'available', 'tasks']
    _elements = {
        'root_offset': ArrayElement(np.int32, ()),  # offset of the root node in the shared memory
    }
    @classmethod
    def define(cls, length: int, num_workers: int):
        class SharedTreeHeader(cls):
            _elements = cls._elements.copy()
            # enumeration of blocks that are available for writing
            # at tree initialization.
            # `index` is used to index this array.
            _elements['available'] = ArrayElement(np.int32, (length,))
            # tasks assigned to each worker
            _elements['tasks'] = ArrayElement(np.int32, (num_workers,))
        return SharedTreeHeader

class TreeFull(Exception): pass

class APV_MCTS(MCTSBase, BaseMemoryManager):
    """
    Shared Memory which stores a tree structure.

    Maintains two shared memory blocks:
    1. header block stores the index of the next block to be used.
    2. data block stores the nodes of the tree
    """
    _EXIT = -1 # worker should exit.
    _IDLE = 0 # worker should not do anything.
    _ROLLOUT = 1 # worker should perform a rollout. (streamlined)
    _SEARCH = 2 # worker should perform phase 1 of the search (in_tree ... simulate)
    _BACKTRACK = 3 # worker should perform phase 2 of the search (backtracking)
    def __init__(self,
        # MCTSBase required parameters
        num_simulations: int,
        num_actions: int,
        model: models.Model,
        search_policy: SearchPolicy,
        # number of parallel search actors.
        num_workers: int,
        # inference; is inference_server is provided, evaluation is ignored.
        inference_server: Optional[AlphaDevInferenceClient] = None,
        evaluation: Optional[types.EvaluationFn] = None,
        # MCTSBase optional parameters
        discount: float = 1.,
        dirichlet_alpha: float = 1,
        exploration_fraction: float = 0.,
        vl_constant:float = -1.0,
        lambda_: float = 0.5,
        name: str = 'SharedTree'):
        """
        Initialize the shared tree with the given number of nodes and node width.
        
        :param num_nodes: number of nodes in the tree, including the root.
        :param node_width: maximum number of children per node, i.e. number of actions in the environment.
        :param vl_constant: constant virtual loss to apply during rollouts.
        """
        self.num_simulations = num_simulations
        self.num_actions = num_actions
        self.model = model
        self.search_policy = search_policy
        
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.vl_constant = vl_constant
        self.name = name
        
        assert inference_server is not None or evaluation is not None, \
            "Either inference_server or evaluation must be provided."
        
        self.inference_server = inference_server
        if inference_server is not None:
            self.evaluation = self._eval_await
        else:
            self.evaluation = evaluation
        
        # declare the shared memory but don't allocate it yet.
        self.num_workers = num_workers
        self.num_nodes = self.num_simulations
        # we use 2x the number of rollouts to allow for keeping the tree from earlier searches.
        self.num_blocks = self.num_nodes * 2
        # declare the header
        self._header_cls = SharedTreeHeaderBase.define(
            length=self.num_blocks, num_workers=self.num_workers)
        self._node_cls = SharedNodeBase.define(
            width=self.num_actions,        # number of actions in the environment
            num_workers=self.num_workers,  # number of parallel workers
            lambda_=lambda_,               # lambda for the value backup
            vl_constant=vl_constant        # constant virtual loss to apply during rollouts
        )
        # declare shared memory regions
        # the header for storing tree-specific information
        self._header_size = self._header_cls.get_block_size()
        self._header_name = f'{name}_header'
        # the data block for representing the tree structure in shared memory
        self._node_size = self._node_cls.get_block_size()

        self._data_size = self.num_blocks * self._node_size
        self._data_name = f'{name}_data'

        # process-local
        self._is_main = False
        self._local_write_head = None
        self._write_head_increment = None
        
        # TODO: declare workers and run them.
        
        if hasattr(self, '_root'):
            # we cannot guarantee that the root node is kept up
            # to date in all worker processes so we load it on-demand.
            del self._root
        
        logger.debug('SharedTree initialized with %d blocks, node size %d, header size %d, data size %d.', self.num_blocks, self._node_size, self._header_size, self._data_size)
    
    def _eval_await(self, observation: types.Observation, node: Optional[SharedNodeBase]=None):
        """Used only when an inference server is used.
        Submit a task and wait for the result.
        Can only be run from the main process
        """
        assert self._is_main, "This method can only be called from the main process."
        offset = node.offset if node is not None else -1
        self.inference_server.submit(**{
            'node_offset': offset, 'observation': observation})
        # wait for the result to be ready
        with self.inference_server.read_ready() as ready:
            assert ready.node_offset == offset, "The node offset in the ready object does not match the requested node. The inference server had leftovers."
            prior = ready.prior
            value = ready.value.item()
        return prior, value
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=True, size=self._data_size)

        self.header = self._header_cls(self._header_shm, 0)
        with self.header.index() as index_counter:
            self._local_write_head = index_counter.load() # get the current write head

        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        """
        Reset all shared memory blocks to their initial state.
        Set the children arrays to -1, indicating that they are uninitialized.
        Same with the root_offset and action in the header.
        """
        assert self._is_main, "reset() should only be called in the main process."
        logger.debug('SharedTree.reset called.')
        # 1. reset the header's counter and available array.
        self._header_cls.clear_block(self._header_shm, 0)
        # 2. clear all nodes that are available for writing.
        for i in range(self.num_blocks):
            offset = self.i2off(i)
            self._node_cls.clear_block(self._data_shm, offset)
            node = self._node_cls(self._data_shm, offset)
            node.set_root()  # set the node as the root (mimic the behavior of the superclass)
            node.children[...] = -1
    
    def init_tree(self, observation: types.Observation, last_action: Optional[int] = None):
        """
        Initialize the tree with the root node corresponding to the given observation.
        The root node is set to be the first node in the shared memory.
        """
        assert self._is_main, "init_tree() should only be called in the main process."
        logger.debug('SharedTree.init_tree called with last_action %s.', last_action)
        # 1. get the root node
        if last_action is not None:
            old_root = self.get_root()
            child_offset = old_root.get_child(last_action)
            if child_offset == -1:
                new_root = None # child doesn't exist, no tree to keep
            else:
                new_root = self.get_by_offset(child_offset)
        else: # no last action no tree to keep
            new_root = None
        logger.debug('SharedTree.init_tree: new root node %s.', repr(new_root))
        # 2. reset the tree and its header, optionally keeping the subtree
        self.reset_tree(keep_subtree=new_root)
        logger.debug('SharedTree.init_tree: tree reset, keeping subtree %s.', repr(new_root))
        # 3. create a new root node
        if new_root is None:
            new_root = self._make_node()
            logger.debug('SharedTree.init_tree: no root node to keep, created %s.', repr(new_root))
        
        self.set_root(new_root)  # set the new root node
        # 4. initialize the root node with the observation if not done already
        if not new_root.expanded:
            logger.debug('SharedTree.init_tree: root node %s is not expanded, invoking network')
            prior, _ = self.evaluation(observation)
            assert prior.shape == (self.num_actions,), \
                f"Expected prior shape ({self.num_actions},) but got {prior.shape}."
        else:
            logger.debug('SharedTree.init_tree: root node %s is already expanded, using existing prior.', repr(new_root))
            prior = new_root.prior
        
        # 3. Add exploration noise to the prior.
        noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
        prior = prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
        new_root.expand(prior) # it's fine to re-expand.
        
        # 4. Set legal actions if not already set.
        if not new_root.legal_actions.any():
            logger.debug('SharedTree.init_tree: root node %s has no legal actions set, getting them from the model.', repr(new_root))
            # 4.1 if the legal actions are not set, we need to get them from the model.
            legal_actions = self.model.legal_actions()
            new_root.set_legal_actions(legal_actions)
        
        # 5. return the root node.
        return new_root
    
    def reset_tree(self, keep_subtree: Optional[SharedNodeBase] = None):
        """
        Reset the tree to its initial state, optionally
        keeping the subtree rooted at the given node.
        Up to num_nodes nodes are kept in the shared memory.
        """
        assert self._is_main, "reset_tree() should only be called in the main process."
        logger.debug('SharedTree.reset_tree called.')
        # 1. find out which nodes to keep.
        keep_nodes = np.zeros(self.num_nodes, dtype=np.bool_)
        if keep_subtree is not None:
            queue = [keep_subtree]; num_kept = 0
            while queue and num_kept < self.num_nodes:
                node = queue.pop(0)
                idx = self.off2i(node.offset)
                keep_nodes[idx] = True
                for child_offset in node.children:
                    if child_offset != -1:
                        child_node = self._node_cls(self._data_shm, child_offset)
                        queue.append(child_node)
                num_kept += 1
            logger.debug('SharedTree.reset_tree: keeping %d nodes in the shared memory.', num_kept)
            # clear the children pointers of the nodes at the end of the retained tree.
            for node in queue:
                node.parent.children[node.action_id] = -1
        # 2. clear all other nodes in the shared memory.
        for i in range(self.num_blocks):
            if keep_nodes[i]:
                continue
            offset = self.i2off(i)
            self._node_cls.clear_block(self._data_shm, offset)
            node = self._node_cls(self._data_shm, offset)
            node.set_root()  # set the node as the root (mimic the behavior of the superclass)
            node.children[...] = -1
        # 3. reset the header's counter and available array.
        self._header_cls.clear_block(self._header_shm, 0)
        self.header = self._header_cls(self._header_shm, 0)
        available = np.arange(self.num_blocks, dtype=np.int32)[~keep_nodes]
        self.header.available[:] = available
    
    def attach(self):
        logger.debug('SharedTree.attach')
        self._is_main = False; self._is_attached = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=False, size=self._data_size)

        self.header = self._header_cls(self._header_shm, 0)
    
    def run_worker(self, worker_id:int):
        """
        Insertion point for the workers.
        Should be invoked before shared memory objects are created.
        First, it waits for the shared memory to be configured by the main process.
        Then, it attaches to the shared memory and
        waits to be signaled to start searching.
        """
        # try to attach to the shared memory
        attached = False
        while not attached:
            try:
                self.attach()
                attached = True
            except FileNotFoundError:
                logger.debug('SharedTree._worker_init: waiting for shared memory to be created.')
                # wait for the main process to create the shared memory
                sleep(0.1)
        # loop forever
        while True:
            task_id = self.header.tasks[worker_id]
            while task_id == APV_MCTS._IDLE:
                task_id = self.header.tasks[worker_id]
                sleep(0.001) # stay eager for the next task
                continue
            # perform the task until the tree is full.
            if task_id == APV_MCTS._EXIT:
                logger.debug('SharedTree._worker_init: worker %d received exit signal.', worker_id)
                break
            stats = []
            # reset local write head for this worker if applicable
            self.worker_ready(worker_id)
            while not self.is_full():
                start = time() # start the timer
                if task_id == APV_MCTS._ROLLOUT:
                    # perform a rollout
                    result = self.rollout()
                elif task_id == APV_MCTS._SEARCH:
                    # perform phase 1 of the search
                    result = self.phase_1()
                elif task_id == APV_MCTS._BACKTRACK:
                    # perform phase 2 of the search
                    result = self.phase_2()
                # log the result and the time taken.
                # this can be useful for load-balancing.
                stats.append((result, time() - start))
            # signal the main process that the task is done
            self.header.tasks[worker_id] = APV_MCTS._IDLE
            # log the statistics
            # TODO: pass stats to main process.
            passes = sum([1 for result, _ in stats if result])
            fails = len(stats) - passes
            pass_times = [t for result, t in stats if result]
            logger.debug('SharedTree._worker_init: worker %d finished task %d with pass/fail %d/%d, avg time %.3f, min time %.3f, max time %.3f.',
                worker_id, task_id, passes, fails,
                np.mean(pass_times) if pass_times else 0.0,
                np.min(pass_times) if pass_times else 0.0,
                np.max(pass_times) if pass_times else 0.0
            )
    
    def worker_ready(self, worker_id: int):
        """
        Set the write head in the tree for this worker.
        it corresponds to the worker's index.
        This way we do not need to worry about race conditions
        in indexing and writing to the shared memory.
        """
        # only the ROLLOUT or SEARCH tasks need a write head.
        # count how many there are
        self._write_head_increment = (
            np.sum(self.header.tasks == APV_MCTS._ROLLOUT) +
            np.sum(self.header.tasks == APV_MCTS._SEARCH))
        if self.header.tasks[worker_id] in (APV_MCTS._ROLLOUT, APV_MCTS._SEARCH):
            self._local_write_head = worker_id
        else:
            self._local_write_head = None
    
    def search(self, observation: types.Observation, last_action: Optional[int] = None):
        """
        1. initialize the tree 
        2. signal the worker processes to start searching
        3. wait for the worker processes to finish searching
        4. return the root node of the tree.
        """
        if not self._is_attached:
            self.configure() 
        # 0. save the current model state
        self.model.save_checkpoint()
        # 1. initialize the tree with the given observation
        root = self.init_tree(observation, last_action)
        # 2. signal the worker processes to start searching
        self.allocate_tasks()
        while not np.all(self.header.tasks == APV_MCTS._IDLE):
            # wait for the worker processes to finish searching
            sleep(0.001)
        return root
    
    def rollout(self) -> bool:
        root = self.get_root()
        try:
            super().rollout(root)
        except:
            return False
    
    def phase_1(self) -> bool:
        root = self.get_root()
        try:
            # the in_tree phase cannot break otherwise we are doomed.
            trajectory, actions = self.in_tree(root)
            # check for race conditions 
            #  - another process expanded in the meantime or
            #  - the last node is not consistent with its parent.
            if trajectory[-1].expanded or not trajectory[-1].is_consistent():
                self.fail(trajectory=trajectory)
                return False
            node, observation = self.simulate(trajectory[-1], actions)
            if not self.is_consistent(node):
                self.fail(trajectory=trajectory)
                return False
            if not node.terminal:
                # instead of calling evaluate, we schedule a task for the inference server
                self.inference_server.submit(**{
                    'node_offset': node.offset, 'observation': observation})
            self.succeed()
            return True
        except TreeFull: # can happen, nothing to do
            self.fail(trajectory=trajectory)
            return False
        except Exception as e:
            logger.exception('SharedTree.phase_1: error during phase 1: %s', e)
            if len(trajectory) > 0:
                self.fail(trajectory=trajectory)
            return False
    
    def phase_2(self) -> bool:
        # poll for results
        with self.inference_server.read_ready() as ready:
            node_offset = ready.node_offset.item()
            prior = ready.prior
            value = ready.value.item()
        node = self.get_by_offset(node_offset)
        # check consistency for the final time
        if not node.is_consistent():
            logger.debug('SharedTree.phase_2: node %s is not consistent with its parent %s.', repr(node), repr(node.parent))
            self.fail(node=node)
            return False
        # do the expansion
        node.expand(prior)
        self.backup(node, value) 
        return True
    
    def get_root(self) -> SharedNodeBase:
        return self.get_by_offset(self.header.root_offset.item())
    def set_root(self, node: SharedNodeBase):
        node.set_root()  # set the node as the root
        self.header.root_offset[...] = node.offset
    
    def get_child(self, node, action):
        child_offset = node.children[action]
        if child_offset == -1:
            return self._make_node(parent=node, action=action)
        return self.get_by_offset(child_offset)
    
    def select_child(self, node, action):
        assert node.expanded, "Node must be expanded to select a child."
        child_offset = node.select(action)
        if child_offset == -1:
            # make a new node and ignore race conditions
            return self._make_node(parent=node, action=action)
        return self._node_cls(self._data_shm, child_offset)
    
    def _make_node(self, parent: Optional[SharedNodeBase] = None, action: Optional[int] = None) -> SharedNodeBase:
        if parent is None and action is None:
            # create a new root node
            return self.get_by_offset(self.header.root_offset.item())
        if self._local_write_head is None:
            assert not self._is_main, "_make_node should only be called from a worker process."
            raise TreeFull()
        
        node = self.get_node(self._local_write_head)
        # override the parent's child pointer regardless of race conditions
        node.set_parent(parent.offset, action)
        parent.set_child(action, node)  # set the child pointer in the parent node
        return node
    
    # ----------------------
    # SHM-specific methods
    # ----------------------
    
    def __del__(self):
        logger.debug('SharedTree.__del__ called. is main: %s', self._is_main)
        del self.header
        self._header_shm.close()
        self._data_shm.close()
        if self._is_main:
            # only the main process should delete the shared memory
            self._header_shm.unlink()
            self._data_shm.unlink()
            # TODO: also stop and kill the workers.
    
    def _update_index(self):
        # increment the write head by the number of workers.
        # this way we are guaranteed that no two workers will write to the same block
        # while also avoiding having to synchronize the write head.
        index = self._local_write_head + self._write_head_increment
        if index >= self.num_nodes:
            self._local_write_head = None # we are done
            return None
        else:
            self._local_write_head = self.header.available[index]
            return self._local_write_head
    
    def fail(self, trajectory: Optional[List[SharedNodeBase]] = None, node: Optional[SharedNodeBase] = None):
        if trajectory is not None:
            # it is a little faster to iterate over the trajectory than having the nodes to re-create the path
            for node in reversed(trajectory):
                node.deselect(node.action_id)
        elif node is not None:
            # if we have a single node, we can just deselect it
            node.parent.deselect(node.action_id, recursive=True)
        else:
            raise ValueError("Either trajectory or node must be provided to fail().")
    
    def succeed(self):
        """We successfully wrote to the block, we can find the next block."""
        self._local_write_head = self._update_index()
    
    def i2off(self, index: int) -> int:
        logger.debug('SharedTree.i2off called with index %d node size %d.', index, self._node_size)
        return index * self._node_size
    def off2i(self, offset: int) -> int:
        logger.debug('SharedTree.off2i called with offset %d node size %d.', offset, self._node_size)
        if offset % self._node_size != 0:
            raise ValueError(f"Offset {offset} is not a multiple of node size {self._node_size}.")
        return offset // self._node_size
    
    def get_node(self, index) -> SharedNodeBase:
        """Return the root node, at offset 0."""
        return self._node_cls(self._data_shm, self.i2off(index))
    
    def get_by_offset(self, offset: int) -> SharedNodeBase:
        return self._node_cls(self._data_shm, offset)
    
    def is_full(self) -> bool:
        """
        Check if the tree is full.
        The tree is full if the index is equal to the number of blocks.
        """
        return self._local_write_head >= self.num_nodes
    
    def allocate_tasks(self):
        if self.streamlined:
            # signal all workers to perform rollouts
            self.header.tasks[:] = APV_MCTS._ROLLOUT
        else:
            # all but one worker searches, the last one backtracks
            tasks = np.full(self.header.tasks.shape, APV_MCTS._SEARCH, dtype=np.int32)
            # NOTE: the memory indexing logic assumes that the last worker is the one the backtracks.
            # if this is not the case, the memory will have holes in it.
            tasks[-1] = APV_MCTS._BACKTRACK
            self.header.tasks[:] = tasks
        logger.debug('SharedTree.allocate_tasks: tasks allocated: %s', self.header.tasks)