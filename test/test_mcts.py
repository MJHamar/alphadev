from alphadev.search.mcts import MCTSBase, PUCTSearchPolicy
from dm_env import TimeStep, StepType
import numpy as np

import logging

num_actions = 100000
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
    def __init__(self):
        self.observation = 0
    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        self.observation = 0
    
    def step(self, actions):
        """Dummy step function that returns a random prior and value."""
        self.observation += 1
        return TimeStep(
            step_type=StepType.MID if self.observation < num_simulations else StepType.LAST,
            observation=self.observation,
            reward=np.zeros((3,), dtype=np.float32),
            discount=1.0
        )
        
    def legal_actions(self):
        """Returns a list of legal actions."""
        return np.ones(num_actions, dtype=np.int32)

def run_mcts():
    mcts = MCTSBase(
        num_simulations=num_simulations,
        num_actions=num_actions,
        model=DummyModel(),
        evaluation=dummy_evaluation_fn,
        search_policy=PUCTSearchPolicy(),
        discount=1.0,
        dirichlet_alpha=dirichet_alpha,
        exploration_fraction=exploration_fraction
    )
    root_node = mcts.search(observation)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Create a dummy model and evaluation function.
    model = DummyModel()
    evaluation_fn = dummy_evaluation_fn

    # Run the MCTS algorithm.
    observation = 0
    run_mcts()
    print("MCTS completed.")