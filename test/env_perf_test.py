"""Performance tests for the environment."""
from time import time
import cProfile
import pstats
import subprocess
from tqdm import tqdm

import tensorflow as tf
import sonnet as snn
import numpy as np

from alphadev.environment import AssemblyGame, EnvironmentFactory, AssemblyGameModel
from alphadev.acting import MCTSActor
from alphadev.search import PUCTSearchPolicy
from alphadev.utils import TaskSpec, generate_sort_inputs
from acme.specs import make_environment_spec
from acme.environment_loop import EnvironmentLoop

class DummyNetwork(snn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.last_action = 0 # cycle through all actions one by one to make this deterministic
        self.value = tf.constant(0.0, dtype=tf.float32)  # dummy value
        
    def __call__(self, observation):
        pi = tf.one_hot(self.last_action)
        self.last_action = (self.last_action + 1) % self.num_actions
        return (pi, self.value)
        
# 10 actions and 10 inputs
def make_ts(num_inputs):
    return TaskSpec(
        # defaults for irrelevant parameters
        max_program_size=100, num_funcs= 10, num_regs=5, num_mem=5, num_locations=10, correct_reward=1.0, correctness_reward_weight=1.0, latency_quantile=0.5, latency_reward_weight=1.0, num_latency_simulations=1,
        observe_reward_components=False,
        # we care about these three.
        num_inputs=num_inputs,
        num_actions=200,
        inputs=generate_sort_inputs(3,5,num_inputs)
    )
    
task_a10_i10 = make_ts(10)

env_10 = AssemblyGame(task_spec=task_a10_i10)

actor_10 = MCTSActor(
    environment_spec=make_environment_spec(env_10),
    network=DummyNetwork(num_actions=task_a10_i10.num_actions),
    model=AssemblyGameModel(task_spec=task_a10_i10),
    search_policy=PUCTSearchPolicy(c_puct_base=19652, c_puct_init=1.25),
    temperature_fn=lambda x: 1.0,  # always deterministic
    num_simulations=100,# make 100 rollouts
    discount=1.0,
)

def select_n_actions(env:AssemblyGame, actor:MCTSActor, n):
    ts = env.reset()
    for _ in tqdm(range(n)):
        action = actor.select_action(ts.observation)
        env.step(action)
        if env._is_invalid:
            ts = env.reset()

def run_env_loop(env, actor, num_steps):
    loop = EnvironmentLoop(env, actor)
    loop.run(num_episodes=num_steps)

def profile_env_loop(env, actor, num_steps):
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_env_loop(env, actor, num_steps)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    return stats

def profile_select_n_actions(env, actor, num_actions):
    profiler = cProfile.Profile()
    profiler.enable()
    
    select_n_actions(env, actor, num_actions)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    return stats

def main(id_):
    num_steps = 100
    num_episodes = 1 # x100
    
    print("Profiling select_action...")
    select_action_stats = profile_select_n_actions(env_10, actor_10, num_steps)
    select_action_stats.dump_stats(f'profile/select_action_profile_{id_}.prof')
    subprocess.run(['flameprof', '-i', f'profile/select_action_profile_{id_}.prof', '-o', f'profile/select_action_flamegraph_{id_}.svg'])
    print("Profiling Environment Loop...")
    env_loop_stats = profile_env_loop(env_10, actor_10, num_episodes)
    env_loop_stats.dump_stats(f'profile/env_loop_profile_{id_}.prof')
    subprocess.run(['flameprof', '-i', f'profile/env_loop_profile_{id_}.prof', '-o', f'profile/env_loop_flamegraph_{id_}.svg'])
    print("Profiling complete.")
    print("Profiles saved to 'profile/select_action_profile.prof' and 'profile/env_loop_profile.prof'.")
    
    
if __name__ == '__main__':
    import sys
    main(id_=sys.argv[1])