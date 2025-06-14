import tqdm
from alphadev.search.apv_mcts import APV_MCTS
from alphadev.service.inference_service import InferenceNetworkFactory

from alphadev.environment import CPUState
from alphadev.search.mcts import PUCTSearchPolicy, visit_count_policy
from alphadev.config import AlphaDevConfig

from dm_env import TimeStep, StepType
import numpy as np
from time import sleep

import logging
logging.basicConfig(level=logging.DEBUG)

import os
ADConfig = AlphaDevConfig.from_yaml(f'{os.path.dirname(__file__)}/apv_mcts_config.yaml')

num_simulations = ADConfig.num_simulations

dirichlet_alpha = 0.3
exploration_fraction = 0.1

def dummy_evaluation_fn(observation):
    """Dummy evaluation function that returns a random prior and value."""
    prior = np.random.rand(ADConfig.task_spec.num_actions)
    value = np.zeros((observation['program_length'].shape[0]), dtype=np.float32)
    # sleep(0.002)  # Simulate some processing delay
    return prior, value

def dummy_eval_factory():
    """Factory function to create a dummy evaluation function."""
    return dummy_evaluation_fn

class DummyModel:
    """Dummy model that does nothing."""
    def __init__(self, timestep):
        self.ts = timestep
    
    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass
    
    def observation_spec(self):
        return CPUState(
            registers=np.zeros((ADConfig.task_spec.num_inputs, ADConfig.task_spec.num_regs), dtype=np.int32),  # Dummy registers
            memory=np.zeros((ADConfig.task_spec.num_inputs, ADConfig.task_spec.num_mem), dtype=np.int32),  # Dummy memory
            program=np.zeros((ADConfig.task_spec.max_program_size,3), dtype=np.int32),  # Dummy program
            program_length=np.zeros((1,), dtype=np.int32),  # Dummy program length
        )._asdict() # Dummy observation

    
    def step(self, actions: np.ndarray):
        """Dummy step function that returns a random prior and value."""
        self.ts = TimeStep(
            step_type=StepType.MID if self.ts.observation['program_length'] < num_simulations else StepType.LAST,
            observation=CPUState(
                registers=self.ts.observation['registers'],
                memory=self.ts.observation['memory'],
                program=self.ts.observation['program'],
                program_length=self.ts.observation['program_length']+1
            )._asdict(),
            reward=np.zeros((3,), dtype=np.float32),
            discount=1.0
        )
        return self.ts
    
    def legal_actions(self):
        """Returns a list of legal actions."""
        return np.array([True]*ADConfig.task_spec.num_actions, dtype=np.bool_)

def select_one(root):
    return 1

def find_longest_path(root, mcts:APV_MCTS):
    frontier = [(root, 0)]
    max_path = 0
    while frontier:
        node, depth = frontier.pop()
        if not node.expanded:
            continue
        for action in range(ADConfig.task_spec.num_actions):
            child_offset = node.get_child(action)
            if child_offset == -1:
                continue
            child_node = mcts._node_cls(mcts._data_shm, child_offset)
            frontier.append((child_node, depth + 1))
            max_path = max(max_path, depth + 1)
    # print(f'SharedTree.search: longest path from root is {max_path}.')

def test_tree_consistent(root, mcts:APV_MCTS):
    """Test if the tree is consistent."""
    frontier = [root]
    while frontier:
        node = frontier.pop()
        if not node.expanded:
            continue
        for action in range(ADConfig.task_spec.num_actions):
            child_offset = node.get_child(action)
            if child_offset == -1:
                continue
            child_node = mcts._node_cls(mcts._data_shm, child_offset)
            if not child_node.is_consistent():
                raise ValueError(f"Child node {child_offset} is not consistent with parent {node.offset}.")
            frontier.append(child_node)

def run_mcts():
    """Run the MCTS algorithm with a dummy model and evaluation function."""
    observation = CPUState(
        registers=np.zeros((ADConfig.task_spec.num_inputs, ADConfig.task_spec.num_regs), dtype=np.int32),  # Dummy registers
        memory=np.zeros((ADConfig.task_spec.num_inputs, ADConfig.task_spec.num_mem), dtype=np.int32),  # Dummy memory
        program=np.zeros((ADConfig.task_spec.max_program_size,3), dtype=np.int32),  # Dummy program
        program_length=np.zeros((1,), dtype=np.int32),  # Dummy program length
    )._asdict() # Dummy observation
    
    timestep = TimeStep(
        step_type=StepType.FIRST,
        observation=observation,
        reward=np.zeros((3,), dtype=np.float32),
        discount=1.0
    )
    
    # test in streamlined mode.
    mcts = APV_MCTS(
        num_simulations=num_simulations,
        num_actions=ADConfig.task_spec.num_actions,
        model=DummyModel(timestep),
        search_policy=PUCTSearchPolicy(),
        # search_policy=select_one,
        num_workers=ADConfig.async_search_processes_per_pool,
        inference_server=None,
        evaluation_factory=dummy_eval_factory,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        exploration_fraction=exploration_fraction,
        vl_constant=-1.0,
    )
    outer_model = DummyModel(timestep)
    action = None
    pbar = tqdm.tqdm(total=num_simulations, desc="MCTS Simulation Progress")
    while timestep.step_type != StepType.LAST:
        root = mcts.search(observation, last_action=action)
        # find_longest_path(root, mcts)
        # test_tree_consistent(root, mcts)
        action_probs = visit_count_policy(root, mask=outer_model.legal_actions())
        action = np.random.choice(ADConfig.task_spec.num_actions, p=action_probs)
        timestep = outer_model.step(action)
        pbar.update(1)
    pbar.close()
    # root = mcts.search(observation)
    # action = visit_count_policy(root, mask=outer_model.legal_actions())
    # timestep = outer_model.step(action)


if __name__ == "__main__":
    run_mcts()
    print("MCTS completed.")