import os
import time
import numpy as np
import pickle
from acme.agents.tf.mcts.search import Node
from acme.utils.counting import Counter

class MCTSObserver:
    """
    Observer for MCTS.
    """
    def _should_log(self):
        return True

    def _on_search(self, node):
        pass
    def _on_backpropagation(self, node):
        pass
    def _on_action_selection(self, node: Node, probs: np.ndarray, action:int, training_steps: int, temperature: float, mcts):
        pass
    def _on_search_end(self, node):
        pass

    def _noop(self, *args, **kwargs):
        pass

    def on_search(self, node):
        """
        Called when a search is performed.
        """
        if self._should_log():
            return self._on_search(node)
        return self._noop(node)

    def on_backpropagation(self, node):
        """
        Called when backpropagation is performed.
        """
        if self._should_log():
            return self._on_backpropagation(node)
        return self._noop(node)

    def on_action_selection(self, node: Node, probs: np.ndarray, action:int, training_steps: int, temperature: float, mcts):
        """
        Called when an action is selected.
        """
        if self._should_log():
            # print(f"Action selected: {action}, probs: ({type(probs)}) {probs.shape}")
            return self._on_action_selection(node, probs, action, training_steps, temperature, mcts)
        return self._noop(node)

    def on_search_end(self, node):
        """
        Called when the search ends.
        """
        if self._should_log():
            return self._on_search_end(node)
        return self._noop(node)

class ProbabilisticObserverMixin(MCTSObserver):
    """
    Observer that only logs with a certain probability.
    """

    def __init__(self, epsilon=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon

    def _should_log(self):
        return np.random.rand() < self._epsilon

class MCTSPolicyObserver(ProbabilisticObserverMixin, MCTSObserver):
    """
    Observer for MCTS that logs the policy.
    """

    def __init__(self, logger, epsilon=0.1):
        super().__init__(epsilon=epsilon)
        self._logger = logger

    def _on_action_selection(self, node: Node, probs: np.ndarray, action:int, training_steps: int, temperature: float, mcts):
        """
        Called when an action is selected.
        """
        self._logger.write({
            'action': action,
            'probs': probs,
            'priors': node.prior,
            # 'expanded': np.asarray([(c.children is not None) for c in node.children.elements()]),
            'values': node.children_values,
            'temperature': temperature,
        })
        from .search.apv_mcts import APV_MCTS # avoid circular import
        if isinstance(mcts, APV_MCTS):
            save_path = os.path.join(os.path.abspath(os.getcwd()), mcts.name + 'saved_tree' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.pkl')
            print(f"Saving MCTS tree to {save_path}")
            with open(save_path, 'wb') as f:
                # Convert memoryview objects to bytes before pickling
                pickle.dump({'data': bytes(mcts._data_shm.buf), 'header': bytes(mcts._header_shm.buf)}, protocol=pickle.HIGHEST_PROTOCOL, file=f)
                print(f"Saved MCTS tree to {f.name}")

# #############
# EnvironmentObservers
# #############
from acme.utils.observers.base import EnvLoopObserver

class CorrectProgramObserver(EnvLoopObserver):
    """
    Observer that logs the correctness of the program.
    """

    def __init__(self):
        super().__init__()
        self._metrics = {}

    def observe_first(self, env, timestep, action=None):
        self._metrics = {
            'is_correct': False,
            'is_invalid': False,
            'invalid_reason': None,
            'max_num_hits': 0,
            'num_hits': 0,
        }

    def observe(self, env, timestep, action=None):
        if timestep.last():
            self._metrics['is_correct'] = env._is_correct
            self._metrics['is_invalid'] = env._is_invalid
            self._metrics['invalid_reason'] = (
                'toolong' if len(env._program) > env._task_spec.max_program_size else
                'invalid' if env._is_invalid else
                'correct'
            )
            self._metrics['num_hits'] = env._num_hits
            self._metrics['max_num_hits'] = max(self._metrics['max_num_hits'], env._num_hits)
        else:
            self._metrics['max_num_hits'] = max(self._metrics['max_num_hits'], env._num_hits)

    def get_metrics(self):
        return self._metrics

class NonZeroRewardObserver(EnvLoopObserver):
    """
    Observer that catches and saves non-zero rewards.
    """
    def __init__(self, experiment_name):
        super().__init__()
        self._trajectory_save_path = os.path.join(
            os.getcwd(),
            'saved_trajectories_' + experiment_name + time.strftime("%Y%m%d", time.localtime() ),
        )
        os.makedirs(self._trajectory_save_path, exist_ok=True)
        self._num_episodes = 0
        self._current_trajectory = []
        self._should_save = False

    def observe_first(self, env, timestep, action=None):
        self._current_trajectory = [timestep]
        self._should_save = False
        self._save_correct = False

    def observe(self, env, timestep, action=None):
        self._current_trajectory.append(timestep)
        if timestep.reward > 0:
            print(f"Non-zero reward observed: {timestep.reward}, program length: {len(env._program)}")
            self._should_save = True
        if timestep.last() and env._is_correct:
            print(f"Correct program found! length: {len(env._program)}, final reward: {timestep.reward}")
            self._save_correct = True
            self._should_save = True

    def get_metrics(self):
        self._num_episodes += 1
        if self._should_save:
            # save the trajectory
            save_path = os.path.join(
                self._trajectory_save_path,
                f'trajectory_{self._num_episodes}_0' + ('_correct' if self._save_correct else '') + '.pkl'
            )
            actor_id = 1
            while os.path.exists(save_path):
                # NOTE: this won't be called too many times. the different actors will agree on an arrangement.
                actor_id += 1
                save_path = os.path.join(
                    self._trajectory_save_path,
                    f'trajectory_{self._num_episodes}_{actor_id}' + ('_correct' if self._save_correct else '') + '.pkl'
                )
            with open(save_path, 'wb') as f:
                pickle.dump(self._current_trajectory, f)
            print(f"Saved trajectory to {save_path}")
            # return the number of saves
            return {
                'trajectory_saved': True,
            }
        else:
            return {
                'trajectory_saved': False,
            }

class TotalRewardObserver(EnvLoopObserver):
    """
    Observer that logs the total reward.
    """

    def __init__(self, counter:Counter):
        super().__init__()
        self._episode_reward = 0.0
        self._counter = counter

    def observe_first(self, env, timestep, action=None):
        self._episode_reward = 0.0

    def observe(self, env, timestep, action=None):
        self._episode_reward += timestep.reward

    def get_metrics(self):
        counts = self._counter.increment(total_reward=self._episode_reward)
        return {
            'total_reward': counts.get('total_reward', 0.0),
        }