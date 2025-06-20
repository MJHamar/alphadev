from typing import Optional
from time import time
import numpy as np

import acme

from .service.variable_service import VariableService

class EvaluationLoop(acme.EnvironmentLoop):
    def __init__(self, 
        environment,
        actor,
        staging_service: VariableService,
        variable_service: VariableService,
        should_update_threshold = 1.0,
        counter = None,
        logger = None,
        label = 'environment_loop',
        observers = ...):
        """
        Environment loop used for network parameter evaluation.
        It overrides the default parameter updating behaviour of the 
        base class; it
        - periodically pulls the latest parameters from the staging service,
        - executes num_evaluation_episodes episodes in the environment,
        - pushes the parameters to the variable service if the average return
            exceeds the should_update_threshold.
        """
        
        super().__init__(
            environment=environment, actor=actor, counter=counter, logger=logger,
            should_update=False,
            label=label, observers=observers)
        
        self.staging_service = staging_service
        self.variable_service = variable_service
        self.should_update_threshold = should_update_threshold
        
        self._echo_parameters()
    
    def run(
        self,
        num_episodes: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> int:
        """Perform the run loop.

        Run the environment loop either for `num_episodes` episodes or for at
        least `num_steps` steps (the last episode is always run until completion,
        so the total number of steps may be slightly more than `num_steps`).
        At least one of these two arguments has to be None.

        Upon termination of an episode a new episode will be started. If the number
        of episodes and the number of steps are not given then this will interact
        with the environment infinitely.

        Args:
            num_episodes: number of episodes to run the loop for.
            num_steps: minimal number of steps to run the loop for.

        Returns:
            Actual number of steps the loop executed.

        Raises:
            ValueError: If both 'num_episodes' and 'num_steps' are not None.
        """
        # NOTE: this is almost identical to the base class' implementation.
        # the only difference is the accumulation of episode returns and
        # the publishing of the parameters.
        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')
        
        def should_terminate(episode_count: int, step_count: int) -> bool:
            return ((num_episodes is not None and episode_count >= num_episodes) or
                    (num_steps is not None and step_count >= num_steps))
        
        episode_count: int = 0
        step_count: int = 0
        
        episode_returns = []
        
        while not should_terminate(episode_count, step_count):
            episode_start = time.time()
            result = self.run_episode()
            result = {**result, **{'episode_duration': time.time() - episode_start}}
            episode_count += 1
            step_count += int(result['episode_length'])
            # Log the given episode results.
            self._logger.write(result)
            # append to the returns liist
            episode_returns.append(result['episode_return'])
            if len(episode_returns) >= self._evaluation_episodes:
                # calculate the average return
                avg_return = np.mean(episode_returns)
                # reset the returns list
                episode_returns.clear()
                # check if we should update the parameters
                if avg_return - self.previous_avg_return >= self.should_update_threshold:
                    # push the current parameters to the variable service
                    self.variable_service.update(
                        self.current_parameters
                    )
                # pull new parameters
                self.current_parameters = self.staging_service.get_variables()
                # update the actor with the new parameters
                # NOTE: in the current implementation this is a noop and the VariableService takes care of it.
                self._actor.update(wait=True)
        
        return step_count