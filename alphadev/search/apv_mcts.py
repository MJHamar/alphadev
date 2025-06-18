from typing import Union, Callable, Dict, List, Optional
import multiprocessing.shared_memory as mp_shm
from time import time, sleep
import numpy as np
import functools
import pickle

import os
import subprocess
import traceback

from acme.agents.tf.mcts.search import SearchPolicy
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts import models

from ..service import service
from ..service.inference_service import AlphaDevInferenceClient, InferenceNetworkFactory
from ..device_config import DeviceConfig, ACTOR, CONTROLLER
from ..shared_memory.base import BlockLayout, ArrayElement, BinaryArrayElement, BaseMemoryManager
from ..environment import AssemblyGame, AssemblyGameModel
from .mcts import MCTSBase, NodeBase

_local_id = 'main'
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format=f'process {_local_id}: %(levelname)s: %(message)s',
)

# process-local variable to keep track of the worker ID

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
        
        class SharedNode(cls):
            # NOTE: python's mro kind of breaks here so we can't use super().define() for width and lam
            _width = width
            _lam = lambda_
            _elements = cls._elements.copy()
            _elements.update({
                'visitors': ArrayElement(np.bool_,   (width, num_workers,)),  # number of visitors to this node
                'prior':    ArrayElement(np.float32, (width,      )),  # prior probabilities of actions
                'W':        ArrayElement(np.float32, (width,      )),  # total value for each action
                'Nw':       ArrayElement(np.int32,   (width,      )),  # visit count for each action
                'R':        ArrayElement(np.float32, (width,      )),  # rewards for each action
                'Nr':       ArrayElement(np.int32,   (width,      )),  # visit count for each action
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
        # logger.debug('%s.select called with action_id %s. from proc. %d', repr(self), action_id, _local_id)
        assert _local_id is not None, "select() can only be called from a worker process. make sure to set _local_id before calling this method."
        
        self.visitors[action_id, _local_id] = True
        
        child_offset = self.children[action_id]
        return child_offset

    def deselect(self, action_id:int, recursive:bool=False) -> int:
        """Inverse operation of select."""
        # logger.debug('%s.deselect called with action_id %s.', repr(self), action_id)
        self.visitors[action_id, _local_id] = False
        if recursive and not self.is_root():
            self.parent.deselect(self.action_id, recursive=True)
        return self.parent_offset
    
    def is_consistent(self):
        # logger.debug('%s.is_consistent called.', repr(self))
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
        # logger.debug('%s.parent called.', repr(self))
        if self.is_root(): return None
        if self._parent is None:  # lazy load the parent node
            self._parent = self.__class__(self.shm, self.parent_offset)
        return self._parent
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
        # logger.debug('%s.is_root called. p.off: %s', repr(self), self.header[self.__class__.hdr_parent])
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
    
    def set_terminal(self, terminal): self.header[self.__class__.hdr_terminal] = terminal

    def get_visit_count(self, action: Optional[types.Action]=None) -> int:
        if action is None: return self.Nr.sum() + self.visitors.sum()
        return self.Nr[action] + self.visitors[action].sum()
    
    @property
    def visit_count(self):
        """Return the visit count of this node."""
        if self.is_root(): 
            return np.sum(self.Nr) + np.sum(self.visitors)
        return self.parent.get_visit_count(self.action_id)
    
    def get_reward(self, action: types.Action) -> float:
        if self.is_root(): return 0.0
        Nr_vl = self.Nr[action] + self.visitors[action].sum()
        if Nr_vl == 0: return 0.0
        return self.R[action] + self.const_vl * self.visitors[action].sum() / Nr_vl
    
    def __repr__(self):
        return f'{self.__class__.__name__}(offset={self.offset}, action_id={self.action_id})'
    
    def get_child(self, action: types.Action) -> 'SharedNodeBase':
        """
        Get the child node corresponding to the given action.
        If the child node does not exist, create a new node and return it.
        """
        return self.children[action]

    def __eq__(self, other):
        # same parent and action
        return self.header[self.__class__.hdr_parent] == other.header[self.__class__.hdr_parent] and \
            self.header[self.__class__.hdr_action] == other.header[self.__class__.hdr_action]

class SharedTreeHeaderBase(BlockLayout):
    _required_attributes = ['root_offset', 'available', 'tasks']
    _elements = {
        'root_offset': ArrayElement(np.int32, ()),  # offset of the root node in the shared memory
        'num_writers': ArrayElement(np.int32, ()),  # number of writer processes.
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

class SharedCheckpointBase(BlockLayout):
    """
    Shared memory block for communicating the model's checkpoint
    to worker processes.
    has a size and a data field
    """
    _required_attributes = ['size', 'data']
    _elements = {
        'size': ArrayElement(np.int32, ()),  # size of the checkpoint data
    }
    @classmethod
    def define(cls, size: int):
        class SharedCheckpoint(cls):
            _elements = cls._elements.copy()
            # defined as a data element but in reality it is a byte array.
            # it is recommended to set the size to 2x the measured model checkpoint
            # FIXME: this should be resolved by making the environment immutable
            # and sharing a model state instead of a pickled checkpoint.
            _elements['data'] = BinaryArrayElement(size)
        return SharedCheckpoint

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
        device_config: DeviceConfig,
        # MCTSBase required parameters
        num_simulations: int,
        num_actions: int,
        model: AssemblyGameModel,
        search_policy: SearchPolicy,
        # number of parallel search actors.
        num_workers: int,
        # inference; is inference_server is provided, evaluation is ignored.
        inference_server: Optional[AlphaDevInferenceClient] = None,
        evaluation_factory: Optional[InferenceNetworkFactory] = None, # for constructing a local copy of network
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
        self._device_config = device_config
        
        self.num_simulations = num_simulations
        self.num_actions = num_actions
        self.model = model
        self.search_policy = search_policy
        
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.vl_constant = vl_constant
        self.name = name
        
        assert inference_server is not None or evaluation_factory is not None, \
            "Either inference_server or evaluation factory must be provided."
        
        if inference_server is not None:
            self.inference_server = inference_server
            self.evaluation = self._eval_await
            self.evaluation_factory = None
        else:
            self.inference_server = None
            self.evaluation_factory = evaluation_factory
            self.evaluation = None
        
        # declare the shared memory but don't allocate it yet.
        self.num_workers = num_workers
        self.num_nodes = self.num_simulations
        # we use 2x the number of rollouts to allow for keeping the tree from earlier searches.
        self.num_blocks = self.num_nodes * 2
        # declare the header
        self._header_cls = SharedTreeHeaderBase.define(
            length=self.num_blocks, num_workers=self.num_workers)
        # declare the node class
        self._node_cls = SharedNodeBase.define(
            width=self.num_actions,        # number of actions in the environment
            num_workers=self.num_workers,  # number of parallel workers
            lambda_=lambda_,               # lambda for the value backup
            vl_constant=vl_constant        # constant virtual loss to apply during rollouts
        )
        # declare the checkpoint class
        ckpt_max_size = model.get_checkpoint_size()
        self._checkpoint_cls = SharedCheckpointBase.define(size=ckpt_max_size*1.2)
        # declare shared memory regions
        # the header for storing tree-specific information
        self._header_size = self._header_cls.get_block_size()
        self._header_name = f'{name}_header'
        # the data block for representing the tree structure in shared memory
        self._node_size = self._node_cls.get_block_size()
        self._data_size = self.num_blocks * self._node_size
        self._data_name = f'{name}_data'
        # the checkpoint block for storing the model's state
        self._checkpoint_size = self._checkpoint_cls.get_block_size()
        self._checkpoint_name = f'{name}_checkpoint'
        
        # process-local
        self._is_main = False
        self._local_write_head = None
        self._is_attached = False
        # shared memory objects
        self._header_shm = None
        self._data_shm = None
        self._checkpoint_shm = None
        # declare workers and run them.
        worker_handles = []
        # if there is an inference server, the actors we deploy here play controller roles.
        # otherwise, in 'streamlined' mode, they are actors.
        worker_device_config = device_config.get_config(CONTROLLER if inference_server is not None else ACTOR)
        worker_calls = [functools.partial(self.run_worker, i) for i in range(self.num_workers)]
        for call in worker_calls:
            handle = service.deploy_service(
                executable=call,
                device_config=worker_device_config,
                label=f'{self.name}_worker',
                num_instances=1,  # each worker is a separate process
            )
            worker_handles.extend(handle)
            logger.debug('Worker started with handle %s.', handle)
        # store the worker handles for later use
        self.worker_handles = worker_handles
        # initialize the shared memory
        self.configure()
        logger.debug('SharedTree initialized with %d blocks, node size %d, header size %d, data size %d.', self.num_blocks, self._node_size, self._header_size, self._data_size)
    
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
        # 2. create a new root node
        if new_root is None: # where-ever the current header points to. (0; if retain_actions=True)
            new_root = self._make_node()
        # logger.debug('SharedTree.init_tree: new root node %s.', repr(new_root))
        # 3. reset the tree and its header, optionally keeping the subtree
        #    this also resets the header; populates tha avilable array, clears the root offset etc.
        self.reset_tree(keep_root=new_root)
        # 4. set the new root node
        self.set_root(new_root)
        # 5. initialize the root node with the observation if not done already
        if not new_root.expanded:
            # logger.debug('SharedTree.init_tree: root node %s is not expanded, invoking network')
            prior, _ = self.evaluation(observation)
            assert prior.shape == (self.num_actions,), \
                f"Expected prior shape ({self.num_actions},) but got {prior.shape}."
        else:
            logger.debug('SharedTree.init_tree: root node %s is already expanded, using existing prior.', repr(new_root))
            prior = new_root.prior
        
        # 6. Add exploration noise to the prior.
        noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
        prior = prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
        new_root.expand(prior) # it's fine to re-expand.
        
        # 7. Set legal actions if not already set.
        if not new_root.legal_actions.any():
            # logger.debug('SharedTree.init_tree: root node %s has no legal actions set, getting them from the model.', repr(new_root))
            # 7.1 if the legal actions are not set, we need to get them from the model.
            legal_actions = self.model.legal_actions()
            new_root.set_legal_actions(legal_actions)
        logger.debug('SharedTree.init_tree: root node %s initialized.',
            repr(new_root))
        # 8. return the root node.
        return new_root
    
    def reset_tree(self, keep_root: Optional[SharedNodeBase]):
        """
        Reset the tree to its initial state, optionally
        keeping the subtree rooted at the given node.
        Up to num_nodes nodes are kept in the shared memory.
        """
        assert self._is_main, "reset_tree() should only be called in the main process."
        assert keep_root is not None, "reset_tree() requires a root node to keep"
        # logger.debug('SharedTree.reset_tree called.')
        # 1. find out which nodes to keep.
        keep_nodes = np.zeros(self.num_blocks, dtype=np.bool_)
        # BSF traversal of the children of the given root node.
        queue = [keep_root]; num_kept = 0
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
        # logger.debug('SharedTree.reset_tree: keeping %d nodes in the shared memory.', num_kept)
        # clear the children pointers of the nodes at the end of the retained tree.
        for node in queue:
            node.parent.children[node.action_id] = -1
        # 2. clear all other nodes in the shared memory.
        # logger.debug('numblocks: %d, num_nodes: %d', self.num_blocks, self.num_nodes)
        for i in range(self.num_blocks):
            if keep_nodes[i]:
                continue
            offset = self.i2off(i)
            self._node_cls.clear_block(self._data_shm, offset)
            node = self._node_cls(self._data_shm, offset)
            node.set_root()  # set parent and action pointers to -1 (mimic the behavior of the superclass)
            node.children[...] = -1 # set the children to -1
        # 4. reset the header's counter and available array.
        self._header_cls.clear_block(self._header_shm, 0)
        self.header = self._header_cls(self._header_shm, 0)
        available = np.arange(self.num_blocks, dtype=np.int32)[~keep_nodes]
        self.header.available[:available.shape[0]] = available
    
    def run_worker(self, worker_id:int, *args, **kwargs):
        """
        Insertion point for the workers.
        Should be invoked before shared memory objects are created.
        First, it waits for the shared memory to be configured by the main process.
        Then, it attaches to the shared memory and
        waits to be signaled to start searching.
        """
        logger.debug('SharedTree.run_worker: worker %d started. Args: %s, Kwargs: %s', worker_id, args, kwargs)
        # construct the network if needed
        if self.evaluation_factory is not None:
            logger.debug('SharedTree.run_worker: constructing evaluation network for worker %d.', worker_id)
            self.evaluation = self.evaluation_factory()
        # set the process-local worker ID
        global _local_id
        _local_id = worker_id
        # try to attach to the shared memory
        attached = False
        while not attached:
            try:
                self.attach()
                attached = True
            except FileNotFoundError:
                # logger.debug('SharedTree.run_worker: waiting for shared memory to be created.')
                # wait for the main process to create the shared memory
                sleep(0.1)
        # loop forever
        logger.debug('SharedTree.run_worker: worker %d attached to shared memory.', worker_id)
        # do profiling if enabled
        prof_dir = os.environ.get('PROFILER_OUTPUT_DIR', None)
        if prof_dir is not None:
            import cProfile
            import pstats
            profiler = cProfile.Profile()
            profiler.enable()
            logger.info(f"mcts_worker_{worker_id} profiling enabled. Output directory: {prof_dir}")

        while True:
            task_id = self.header.tasks[worker_id]
            logger.debug('SharedTree.run_worker: worker %d waiting for task, current task id: %d.', worker_id, task_id)
            while task_id == APV_MCTS._IDLE:
                task_id = self.header.tasks[worker_id]
                sleep(0.001) # stay eager for the next task
                continue
            logger.debug('SharedTree.run_worker: worker %d received task %d.', worker_id, task_id)
            # perform the task until the tree is full.
            if task_id == APV_MCTS._EXIT:
                logger.debug('SharedTree.run_worker: worker %d received exit signal.', worker_id)
                break
            stats = []
            logger.debug('SharedTree.run_worker: worker %d starting task %d.', worker_id, task_id)
            # reset local write head for this worker if applicable
            self.worker_ready(worker_id)
            while not self.is_full(task_id):
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
            if logger.isEnabledFor(logging.DEBUG):
                passes = sum([1 for result, _ in stats if result])
                fails = len(stats) - passes
                pass_times = [t for result, t in stats if result]
                logger.debug('SharedTree.run_worker: worker %d finished task %d with pass/fail %d/%d, avg time %.3f, min time %.3f, max time %.3f.',
                    worker_id, task_id, passes, fails,
                    np.mean(pass_times) if pass_times else 0.0,
                    np.min(pass_times) if pass_times else 0.0,
                    np.max(pass_times) if pass_times else 0.0
                )
        # do profiling if enabled
        if prof_dir is not None:
            profiler.disable()
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.dump_stats(os.path.join(prof_dir, f'mcts_worker_{worker_id}_profile.prof'))
            subprocess.run(['flameprof', '-i', os.path.join(prof_dir, f'mcts_worker_{worker_id}_profile.prof'), '-o', os.path.join(prof_dir, f'mcts_worker_{worker_id}_profile.svg')])
    
    def worker_ready(self, worker_id: int):
        """
        Set the write head in the tree for this worker.
        it corresponds to the worker's index.
        This way we do not need to worry about race conditions
        in indexing and writing to the shared memory.
        """
        # attach to inference client if needed.
        if self.inference_server is not None:
            self.inference_server.attach()
        # only the ROLLOUT or SEARCH tasks need a write head.
        # count how many there are
        if self.header.tasks[worker_id] in (APV_MCTS._ROLLOUT, APV_MCTS._SEARCH):
            self._local_write_head = worker_id
            # also load the model's checkpoint.
            ckpt = self.worker_load_checkpoint()
            self.model.set_checkpoint(ckpt)  # override the model's internal checkpoint
            self.model.load_checkpoint()  # load the model's state
        else:
            self._local_write_head = None
        logger.debug('SharedTree.worker_ready: worker %d is ready with write head %s, increment %s.',
            worker_id, self._local_write_head, self.write_head_increment)
    
    def search(self, observation: types.Observation, last_action: Optional[int] = None):
        """
        1. initialize the tree 
        2. signal the worker processes to start searching
        3. wait for the worker processes to finish searching
        4. return the root node of the tree.
        """
        if self.evaluation is None:
            # construct the parent process' evaluation network (used for initializing the tree)
            self.evaluation = self.evaluation_factory()
        # 0. save the current model state
        cktp = self.model.make_checkpoint() # see `alphadev.environment.AssemblyGameModel.get_checkpoint()`
        self.broadcast_checkpoint(cktp)  # broadcast the checkpoint to all workers
        # 1. initialize the tree with the given observation
        root = self.init_tree(observation, last_action)
        # 2. signal the worker processes to start searching
        self.allocate_tasks()
        while not np.all(self.header.tasks == APV_MCTS._IDLE):
            # wait for the worker processes to finish searching
            sleep(0.005)
                # DFS on the tree to see longest path from the root
        self.clear_tasks()
        return root
    
    def rollout(self) -> bool:
        root = self.get_root()
        trajectory = []  
        times = [None] * 4
        try:
            # 1. search for a leaf node in the tree.
            # the in_tree phase cannot break otherwise we are doomed.
            # logger.debug(f'SharedTree.rollout_{self._local_write_head}: --- in_tree call')
            times[0] = time()
            trajectory, actions = self.in_tree(root)
            # check for race conditions 
            #  - another process expanded in the meantime or
            #  - the last node is not consistent with its parent.
            if trajectory[-1].expanded or not trajectory[-1].is_consistent():
                # logger.debug(f'SharedTree.rollout_{self._local_write_head}: fail after in_tree exp: %s, consistent: %s', trajectory[-1].expanded, trajectory[-1].is_consistent())
                self.fail(trajectory=trajectory)
                return False
            # 2. simulate the trajectory
            # logger.debug(f'SharedTree.rollout_{self._local_write_head}: --- simulate call')
            times[1] = time()
            node, observation = self.simulate(node=trajectory[-1], ancestors=trajectory[:-2], actions=actions)
            if not node.is_consistent():
                # logger.debug(f'SharedTree.rollout_{self._local_write_head}: fail after simulate: node %s is not consistent with its parent %s.', repr(node), repr(node.parent))
                self.fail(trajectory=trajectory)
                return False
            # 3. evaluate the node
            # logger.debug(f'SharedTree.rollout_{self._local_write_head}: --- evaluate call')
            times[2] = time()
            _, value = self.evaluate(node, observation)
            # 4. backup the value and the node
            # logger.debug(f'SharedTree.rollout_{self._local_write_head}: --- backup call')
            times[3] = time()
            self.backup(node, np.asarray(value), trajectory=trajectory)
            self.succeed()
            # logger.debug(f'SharedTree.rollout_{self._local_write_head}: --- success.')
            return True
        except TreeFull: # can happen, nothing to do
            self.fail(trajectory=trajectory)
            return False
        except Exception as e:
            logger.exception(f'SharedTree.rollout_{self._local_write_head}: error during rollout: %s', e)
            logger.error(f'State during error: tr len {len(trajectory)} trajectory: {trajectory}, terminals: {[n.terminal for n in trajectory]}, expanded: {[n.expanded for n in trajectory]}, header tasks: {self.header.tasks}, write head: {self._local_write_head}, worker id : {_local_id}, increment: {self.write_head_increment}')
            if len(trajectory) > 0:
                self.fail(trajectory=trajectory)
            raise e
        finally:
            # in either case, reset the model to the root state.
            load_start = time()
            self.model.load_checkpoint()
            load_end = time()
            logger.info('SharedTree.rollout_{wh}: total {total:.5f}; tree {tree:.5f}; sim {sim:5f}; eval {eval:5f}; backup {backup:5f}; load {load:.5f}.'.format(
                wh=self._local_write_head,
                total=load_end - times[0],
                tree=times[1] - times[0] if times[1] is not None else 9.9999,
                sim=times[2] - times[1] if times[1] is not None and times[2] is not None else 9.9999,
                eval=times[3] - times[2] if times[2] is not None and times[3] is not None else 9.9999,
                backup=load_start - times[3] if times[3] is not None else 9.9999,
                load=load_end - load_start
            ))
    
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
            node, observation = self.simulate(node=trajectory[-1], ancestors=trajectory[:-2], actions=actions)
            if not node.is_consistent():
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
            raise e
        finally:
            # in either case, reset the model to the root state.
            self.model.load_checkpoint()
    
    def phase_2(self) -> bool:
        # poll for results
        with self.inference_server.read_ready() as ready:
            if len(ready) == 0:
                return False
            assert len(ready) == 1, "Expected exactly one ready object."
            ready = ready[0]
            node_offset = ready.node_offset.item()
            prior = ready.prior
            value = ready.value.item()
        assert node_offset != -1, "Node offset must not be -1."
        node = self.get_by_offset(node_offset)
        # check consistency for the final time
        if not node.is_consistent():
            # logger.debug('SharedTree.phase_2: node %s is not consistent with its parent %s.', repr(node), repr(node.parent))
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
        # logger.debug('%s.select_child called with action %s. offset %s', repr(node), action, child_offset)
        if child_offset == -1:
            # make a new node and ignore race conditions
            return self._make_node(parent=node, action=action)
        return self._node_cls(self._data_shm, child_offset)
    
    def _make_node(self, parent: Optional[SharedNodeBase] = None, action: Optional[int] = None) -> SharedNodeBase:
        if parent is None and action is None:
            # create a new root node
            return self.get_by_offset(self.header.root_offset.item())
        # logger.debug('SharedTree._make_node called with parent %s, action %s.', repr(parent), action)
        # logger.debug('SharedTree._make_node: local write head %s. available %s', self._local_write_head,self.header.available[self._local_write_head])
        logger.debug("SharedTree._make_node: make_node at idx %d, parent %s, action %s.",
                    self.header.available[self._local_write_head], repr(parent), action)
        node = self.get_node(self.header.available[self._local_write_head])
        assert not node.expanded, f"Node {repr(node)} is already expanded, this is unexpected."
        # if node.expanded:
        #     logger.warning('SharedTree._make_node: node %s is already expanded, this is unexpected.', repr(node))
        #     node.header[self._node_cls.hdr_expanded] = False
        # override the parent's child pointer regardless of race conditions
        node.set_parent(parent, action)
        parent.set_child(action, node)  # set the child pointer in the parent node
        return node
    
    # ----------------------
    # SHM-specific methods
    # ----------------------
    
    def _eval_await(self, observation: types.Observation, node: Optional[SharedNodeBase]=None):
        """Used only when an inference server is used.
        Submit a task and wait for the result.
        Can only be run from the main process
        """
        assert self._is_main, "This method can only be called from the main process."
        self.inference_server.attach()
        offset = node.offset if node is not None else -1
        # logger.debug('SharedTree._eval_await called with node offset %s.', offset)
        self.inference_server.submit(**{
            'node_offset': offset, 'observation': observation})
        # wait for the result to be ready
        with self.inference_server.read_ready(max_samples=1) as ready:
            assert len(ready) == 1, "Expected exactly one ready object."
            ready = ready[0]
            assert ready.node_offset == offset, f"The node offset in the ready object {ready.node_offset} does not match the requested node {offset}. The inference server had leftovers."
            prior = ready.prior
            value = ready.value.item()
        # logger.debug('SharedTree._eval_await: returning with prior %s, value %s.', prior.shape, value)
        return prior, value
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        # logger.debug('SharedTree.configure called.')
        self._is_main = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=True, size=self._data_size)
        self._checkpoint_shm = mp_shm.SharedMemory(name=self._checkpoint_name, create=True, size=self._checkpoint_size)
        
        self.header = self._header_cls(self._header_shm, 0)
        self.checkpoint = self._checkpoint_cls(self._checkpoint_shm, 0)
        
        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        """
        Reset all shared memory blocks to their initial state.
        Set the children arrays to -1, indicating that they are uninitialized.
        Same with the root_offset and action in the header.
        """
        assert self._is_main, "reset() should only be called in the main process."
        # logger.debug('SharedTree.reset called.')
        # 1. reset the header's counter and available array.
        self._header_cls.clear_block(self._header_shm, 0)
        # 2. clear all nodes that are available for writing.
        for i in range(self.num_blocks):
            offset = self.i2off(i)
            self._node_cls.clear_block(self._data_shm, offset)
            node = self._node_cls(self._data_shm, offset)
            node.set_root()  # set the node as the root (mimic the behavior of the superclass)
            node.children[...] = -1
        # 3. reset the checkpoint block
        self._checkpoint_cls.clear_block(self._checkpoint_shm, 0)
    
    def attach(self):
        # logger.debug('SharedTree.attach')
        self._is_main = False; self._is_attached = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._data_shm = mp_shm.SharedMemory(name=self._data_name, create=False, size=self._data_size)
        self._checkpoint_shm = mp_shm.SharedMemory(name=self._checkpoint_name, create=False, size=self._checkpoint_size)
        
        self.header = self._header_cls(self._header_shm, 0)
        self.checkpoint = self._checkpoint_cls(self._checkpoint_shm, 0)
    
    def __del__(self):
        global _local_id
        logger.debug('SharedTree.__del__ called for %s; local id %s main %s.', self.name, _local_id, self._is_main)
        if not self._is_main and self._is_attached:
            self._header_shm.close()
            self._data_shm.close()
            self._checkpoint_shm.close()
        elif self._is_main:
            logger.debug('SharedTree.__del__: main process %s is terminating.', self.name)
            # signal worker to exit gracefully
            self.header.tasks[:] = APV_MCTS._EXIT
            # terminate the workers; they should terminate on their own (given the EXIT task)
            num_not_terminated = len(self.worker_handles)
            while num_not_terminated > 0:
                for handle in self.worker_handles:
                    proc: subprocess.Popen = handle[1]
                    if rc := proc.poll() is not None:
                        logger.debug('SharedTree.__del__: worker %s terminated with return code %s.', handle[0], rc)
                        num_not_terminated -= 1
                sleep(0.5) # wait for the workers
            logger.debug('SharedTree.__del__: all workers terminated.')
            # only the main process should unlink the shared memory
            try:
                self._header_shm.unlink()
            except FileNotFoundError:
                logger.warning('SharedTree.__del__: shared memory already unlinked.')
            try:
                self._data_shm.unlink()
            except FileNotFoundError:
                logger.warning('SharedTree.__del__: shared memory already unlinked.')
            try:
                self._checkpoint_shm.unlink()
            except FileNotFoundError:
                logger.warning('SharedTree.__del__: shared memory already unlinked.')
        # logger.debug('SharedTree.__del__: %s terminated', self.name)
    
    def _update_index(self):
        # increment the write head by the number of workers.
        # this way we are guaranteed that no two workers will write to the same block
        # while also avoiding having to synchronize the write head.
        self._local_write_head = self._local_write_head + self.write_head_increment
        # logger.debug('SharedTree._update_index: local write head updated to %s.', self._local_write_head)
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
        # logger.debug('SharedTree.i2off called with index %d node size %d.', index, self._node_size)
        return index * self._node_size
    def off2i(self, offset: int) -> int:
        # logger.debug('SharedTree.off2i called with offset %d node size %d.', offset, self._node_size)
        if offset % self._node_size != 0:
            raise ValueError(f"Offset {offset} is not a multiple of node size {self._node_size}.")
        return offset // self._node_size
    
    def get_node(self, index) -> SharedNodeBase:
        """Return the root node, at offset 0."""
        return self._node_cls(self._data_shm, self.i2off(index))
    
    def get_by_offset(self, offset: int) -> SharedNodeBase:
        return self._node_cls(self._data_shm, offset)
    
    def is_full(self, task_id) -> bool:
        """
        Check if the tree is full. The logic differs depending on the task:
        - For ROLLOUT and SEARCH tasks, the tree is full if the local write head
          is greater than or equal to the number of blocks.
        - For BACKTRACK tasks, the tree is full if the inference server is idle
          and all tasks in the header are set to IDLE or one of them EXITed.
        - For IDLE and EXIT task ids, an exception is raised.
        The tree is full if the index is equal to the number of blocks.
        """
        if task_id == APV_MCTS._ROLLOUT or task_id == APV_MCTS._SEARCH:
            return self._local_write_head >= self.num_nodes
        elif task_id == APV_MCTS._BACKTRACK:
            return self.inference_server.is_idle() and np.all(self.header.tasks[:-1] == APV_MCTS._IDLE)
        else:
            raise ValueError(f"Invalid task_id {task_id}. Cannot check if the tree is full.")
    
    def allocate_tasks(self):
        if self.inference_server is None:
            # signal all workers to perform rollouts
            self.header.tasks[:] = APV_MCTS._ROLLOUT
            self.header.num_writers[...] = self.num_workers
        else:
            # all but one worker searches, the last one backtracks
            tasks = np.full(self.header.tasks.shape, APV_MCTS._SEARCH, dtype=np.int32)
            # NOTE: the memory indexing logic assumes that the last worker is the one the backtracks.
            # if this is not the case, the memory will have holes in it.
            tasks[-1] = APV_MCTS._BACKTRACK
            self.header.tasks[:] = tasks
            assert self.num_workers > 1, "At least two workers are required for APV_MCTS with inference server."
            self.header.num_writers[...] = self.num_workers - 1
        logger.debug('SharedTree.allocate_tasks: tasks allocated: %s', self.header.tasks)
    
    @property
    def write_head_increment(self) -> int:
        while (increment := self.header.num_writers) == 0:
            # wait for the main process to set the write head
            sleep(0.001)
        return increment
    
    def broadcast_checkpoint(self, checkpoint: AssemblyGame):
        # FIXME: env should be immutable and checkpoint np-serializable.
        logger.debug('SharedTree.broadcast_checkpoint called.')
        assert self._is_main, "broadcast_checkpoint() should only be called in the main process."
        ckpt_bin = pickle.dumps(checkpoint)
        self.checkpoint.data[:len(ckpt_bin)] = ckpt_bin
        self.checkpoint.size[...] = len(ckpt_bin)
    
    def worker_load_checkpoint(self) -> AssemblyGame:
        logger.debug('SharedTree.worker_load_checkpoint called.')
        assert not self._is_main, "worker_get_checkpoint() should only be called in worker processes."
        if self.checkpoint.size == 0:
            raise ValueError("Checkpoint is not set. The main process should call broadcast_checkpoint() first.")
        ckpt_bin = self.checkpoint.data[:self.checkpoint.size.item()]
        return pickle.loads(ckpt_bin)
    
    def clear_tasks(self):
        """Clear the tasks in the header."""
        self.header.tasks[:] = APV_MCTS._IDLE
        self.header.num_writers[...] = 0
        # logger.debug('SharedTree.clear_tasks: tasks cleared.')
