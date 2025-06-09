"""
Implementation of APV-MCTS
"""
from typing import NamedTuple, Dict, Optional, Sequence, Callable
from time import time

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory as mp_shm
mp.set_start_method('spawn', force=True)  # use spawn method for multiprocessing
from collections import namedtuple
import functools

from .environment import AssemblyGame, AssemblyGameModel
from .service.inference_service import AlphaDevInferenceService, run_inference, InferenceFactory
from .service.variable_service import VariableService
from .shared_memory.base import BlockLayout, ArrayElement, BaseMemoryManager
from .shared_memory.tree import SharedTree, NodeBase, TreeFull, SharedTreeFactory

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

class TaskAllocatorHeaderBase(BlockLayout):
    _required_attributes = ['process_task']
    _elements = {}
    @classmethod
    def define(cls, num_actors: int):
        class TaskAllocatorHeader(cls):
            _elements = cls._elements.copy()
            _elements['process_task'] = ArrayElement(np.int32, (num_actors,))  # list of tasks assigned to each process
        return TaskAllocatorHeader

class SharedTaskAllocator(BaseMemoryManager):
    def __init__(self, num_actors, name: str = 'SharedTaskAllocator'):
        self.num_actors = num_actors
        self.header_cls = TaskAllocatorHeaderBase.define(num_actors)
        self._header_size = self.header_cls.get_block_size()
        self._header_name = f'{name}_header'
        self._header_shm = None
        self._header = None
        self._is_main = False

    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=True, size=self._header_size)

        self.header = self.header_cls(self._header_shm, 0)

        self.reset()

    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        self.header_cls.clear_block(self._header_shm, 0)

    def attach(self):
        self._is_main = False
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self.header = self.header_cls(self._header_shm, 0)

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
        assert len(task_ids) == self.num_actors, "Number of task ids must match the number of actors."

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
            model: AssemblyGameModel,
            search_policy: Callable[[NodeBase], int],
            num_simulations: int,
            num_actions: int,
            num_actors: int,
            inference_factory: InferenceFactory,
            discount = 1.0,
            dirichlet_alpha = 1.0,
            exploration_fraction = 0.0,
            const_vl: float = 1.0,
            retain_subtree: bool = False, # whether to keep the subtree corr. to the last selected action.
            do_profiling: bool = False,
            observers: Sequence = [],
            name:str='apv-mcts', # needs to be specified is multiple instances are used.
        ):
        self.model = model # a (learned) model of the environment.
        self.search_policy = search_policy # a function that takes a node and returns an action to take.
        self.num_simulations = num_simulations  # number of simulations to run per search
        self.num_actions = num_actions  # number of actions in the environment
        self.num_actors = num_actors  # number of actors to run in parallel
        assert self.num_actors > 1, "APV_MCTS requires at least 2 actors."
        
        self.discount = discount  # discount factor for the value estimate
        self.observers = observers  # list of observers that evaluate search statistics.
        
        # applying Dirichlet noise to the prior probabilities at the root.
        self.dirichlet_alpha = dirichlet_alpha  # alpha parameter for the Dirichlet noise
        self.exploration_fraction = exploration_fraction  # fraction of the prior to add Dirichlet noise to
        
        self.retain_subtree = retain_subtree  # whether to keep the subtree corresponding to the last selected action
        self.last_action = None  # last action taken, used to retain the subtree
        
        self.do_profiling = do_profiling  # whether to enable profiling
        
        # declare shared memory ( no init )
        self.tree_factory = SharedTreeFactory(
            num_nodes=num_simulations,
            width=self.num_actions, 
            vl_const=const_vl,
            name=f'{name}.tree'
        )
        # keep a pointer to the inference factory. will use as a client.
        self.inference_buffer_factory = inference_factory
        self.task_allocator_factory = functools.partial(
            SharedTaskAllocator, self.num_actors, f'{name}.task_allocator')
        
        # construct local shared memory managers
        self.tree = self.tree_factory()
        self.inference_buffer = self.inference_buffer_factory()
        self.task_allocator = self.task_allocator_factory()
        
        # configure the shared memory managers (create shared memory blocks)
        self.tree.configure()
        self.inference_buffer.configure()
        self.task_allocator.configure()
        
        # initialize the task allocation; 
        #   - N-1 actors search the tree and simulate,
        #   - 1 actor expands and backpropagates.
        self._init_task_allocation = \
            [APV_MCTS.SEARCH_SIM_TASK] * (self.num_actors - 1) + [APV_MCTS.BACKUP_TASK] # last actor is the one listening to the evaluator
        # set initial task allocation
        self.task_allocator.allocate(self._init_task_allocation)
        
        # declare the inference process and actor pool
        self.inference_process = mp.Process(
            target=run_inference,
            # TODO: pass device config
            args=(self.inference_buffer_factory,),
            name=f'{name}.inf_proc'
        )
        self.actor_pool = mp.Pool(processes=self.num_actors)
        logger.debug("APV_MCTS[main]: Finished initialization.")

    def search(self, observation, last_action:Optional[int]=None) -> NodeBase:
        """
        Perform APV-MCTS search on the given observation.
        This method initializes the search tree with root corresponding to the given observation,
        and runs num_simulations rollouts from the root node using a pool of worker processes.
        Finally, it returns a pointer to the root node, which can be used to perform post-mcts action selection.
        """
        # start the inference processs is it isn't already running.
        if not self.inference_process.is_alive():
            self.inference_process.start()
            logger.debug(f"APV_MCTS[main]: Inference process started with PID {self.inference_process.pid}.")
        
        logger.debug('APV_MCTS[main process] Starting search simulation phase with last action %s.', last_action)
        if not self.retain_subtree:
            last_action = None # never retain subtree, so reset last_action.
        
        self.tree.reset(last_action=last_action)
        
        root: NodeBase = self.tree.get_root()
        if not root.expanded: # this is guaranteed to be the case if last_action was None or the corr. node was never visited.
            self.inference_buffer.submit(node_offset=0, observation=observation)
            
            logger.debug('APV_MCTS[main process] Waiting for inference service to return prior and value estimate.')
            with self.inference_buffer.poll_ready() as inference_result:
                _, prior, _ = inference_result
            logger.debug('APV_MCTS[main process] Received prior and value estimate from inference service.')
            # Add exploration noise to the prior.
            noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
            prior = prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
            
            root.expand(prior)
            logger.debug('APV_MCTS[main process] Root node initialized with prior and legal actions. Starting pool')
        else:
            noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
            root.set_prior(
                root.prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
            )
        if not root.mask.any():
            legal_actions = self.model.legal_actions()
            root.set_legal_actions(legal_actions)
        
        # all that is left is to start the pool.
        logger.debug('APV_MCTS[main process] Starting actor pool with %d actors.', self.num_actors)
        statistics = self.actor_pool.map(
            _run_task,
            [(i,
              self.tree_factory, self.inference_buffer_factory, self.task_allocator_factory,
              self.model, self.search_policy, self.discount, self.do_profiling
              ) for i in range(self.num_actors)
            ]
        )
        for o in self.observers:
            o.update(statistics, root, self.tree, self.inference_buffer)
        logger.debug('APV_MCTS[main process] search done. Root (W/N):\n %s', list(zip(root.W, root.N)))
        
        return root  # return the root node, which now contains the search results

def _phase_1(root: NodeBase, tree: SharedTree, model: AssemblyGameModel,
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
    # logger.setLevel(logging.INFO)
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

# process-local shared memory managers to save on initialization time.
_task_tree = None
_task_inference_buffer = None
_task_task_allocator = None

def _run_task(args):
        (process_id,
         tree_factory, inference_factory, task_allocator_factory,
         model, search_policy, discount,
         do_profiling) = args
        logging.basicConfig(
            format=f'%(asctime)s - APV_MCTS[process {process_id}] - %(levelname)s - %(message)s',
            level=logging.DEBUG,
        )
        global _task_tree, _task_inference_buffer, _task_task_allocator
        if _task_tree is None:
            logger.debug(f"APV_MCTS[process {process_id}] Initializing shared tree.")
            _task_tree = tree_factory()
            _task_tree.attach()
        if _task_inference_buffer is None:
            logger.debug(f"APV_MCTS[process {process_id}] Initializing inference buffer.")
            _task_inference_buffer = inference_factory()
            _task_inference_buffer.attach()
        if _task_task_allocator is None:
            logger.debug(f"APV_MCTS[process {process_id}] Initializing task allocator.")
            _task_task_allocator = task_allocator_factory()
            _task_task_allocator.attach()
        tree = _task_tree
        inference_buffer = _task_inference_buffer
        task_allocator = _task_task_allocator

        my_task_id = task_allocator.get_task(process_id)
        if my_task_id == APV_MCTS.SEARCH_SIM_TASK:
            # FIXME: this will break everything if task allocation is dynamic.
            tree.init_index()  # initialize the local write head for this process and run.
        
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
        
        if do_profiling:
            profiler = cProfile.Profile()
            profiler.enable()
        
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
        logger.info("APV_MCTS[process %s] done. task %s; duration: %s; successes: %d, fails: %d; index %s", process_id, stats['task_ids'][-1], stats['duration'], stats['num_successes'], stats['num_fails'], tree.header.index.peek())
        
        if do_profiling:
            profiler.disable()
            prof_stats = pstats.Stats(profiler)
            prof_stats.sort_stats('cumulative')
            # print_mask_stats(actor._model._environment._action_space_storage)
            prof_stats.dump_stats(f'profile/apv_mcts_profile_{process_id}.prof')
            subprocess.run(['flameprof', '-i', f'profile/apv_mcts_profile_{process_id}.prof', '-o', f'profile/apv_mcts_flamegraph_{process_id}.svg'])
            logger.debug(f"APV_MCTS[process {process_id}] Profiling done, results saved.")
        
        return stats
