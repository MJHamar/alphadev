from alphadev.search import dv_mcts, dyn_puct
from dm_env import TimeStep, StepType
import numpy as np

num_actions = 100
num_simulations = 1000

dirichet_alpha = 0.3
exploration_fraction = 0.1

def dummy_evaluation_fn(observation):
    """Dummy evaluation function that returns a random prior and value."""
    prior = np.random.rand(num_actions)
    value = 0.0
    return prior, value

class DummyModel:
    """Dummy model that does nothing."""
    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass
    
    def step(self, observation):
        """Dummy step function that returns a random prior and value."""
        return TimeStep(
            step_type=StepType.MID if observation < num_simulations else StepType.LAST,
            observation=observation+1,
            reward=np.zeros((3,), dtype=np.float32),
            discount=1.0
        )

def run_mcts():
    root_node = dv_mcts(
        observation=observation,
        model=model,
        search_policy=dyn_puct,
        evaluation=evaluation_fn,
        num_simulations=num_simulations,
        num_actions=num_actions,
        dirichlet_alpha=dirichet_alpha,
        exploration_fraction=exploration_fraction
    )
    

if __name__ == "__main__":
    # Create a dummy model and evaluation function.
    model = DummyModel()
    evaluation_fn = dummy_evaluation_fn

    # Run the MCTS algorithm.
    observation = 0
    run_mcts()
    print("MCTS completed.")