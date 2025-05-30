"""Performance tests for the environment."""
from time import time
import cProfile
import pstats
import subprocess
from tqdm import tqdm

import tensorflow as tf
import sonnet as snn
import numpy as np

from alphadev.environment import AssemblyGame, EnvironmentFactory, AssemblyGameModel, ModelFactory
from alphadev.acting import MCTSActor
from alphadev.search import PUCTSearchPolicy
from alphadev.utils import TaskSpec, generate_sort_inputs
from alphadev.config import AlphaDevConfig
from alphadev.network import AlphaDevNetwork, NetworkFactory
from alphadev.alphadev_acme import make_agent
from acme.specs import make_environment_spec
from acme.environment_loop import EnvironmentLoop

class DummyNetwork(snn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.last_action = 0 # cycle through all actions one by one to make this deterministic
        self.value = tf.constant([0.0], dtype=tf.float32)  # dummy value
    
    @tf.function
    def __call__(self, observation):
        pi = tf.one_hot([self.last_action], self.num_actions) #add batch dim
        self.last_action = (self.last_action + 1) % self.num_actions
        return pi, self.value
        
# 10 actions and 10 inputs
def make_ts(num_inputs):
    return TaskSpec(
        # defaults for irrelevant parameters
        max_program_size=100, num_funcs= 10, num_regs=5, num_mem=5, num_locations=10, correct_reward=1.0, correctness_reward_weight=1.0, latency_quantile=0.5, latency_reward_weight=1.0, num_latency_simulations=1,
        observe_reward_components=False,
        # we care about these three.
        num_inputs=num_inputs,
        num_actions=200,
        inputs=generate_sort_inputs(3,5,num_inputs),
        emulator_mode='i32'
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

# realistic scenario
def actor_env_from_config(path) -> EnvironmentLoop:
    """Load the configuration from a file."""
    config = AlphaDevConfig.from_yaml(path)
    assert not config.distributed, "This test is not for distributed agents."
    agent = make_agent(config)
    environment = EnvironmentFactory(config)()
    env_executor = EnvironmentLoop(
        environment=environment,
        actor=agent,
        counter=agent.counter,
        should_update=False,
        logger=agent.logger,
        observers=[]
    )
    return agent._actor, env_executor


def print_mask_stats(action_space_storage):
    mask_stats = action_space_storage._stats
    if not mask_stats:
        print("No mask stats available.")
        return
    # 'hashes' counter
    # 'mask_calls' counter
    # 'mask_history_hitmiss' list of dict 'hit' and 'miss' counts per call
    # 'mask_updates' list of counts of updates per call
    # 'mask_empty' counter 
    # 'mask_nonempty' counter (empty+nonempty = mask_calls)
    print(f"Mask stats:")
    cache_size = len(action_space_storage._mask_cache)
    print(f"    cache size: {cache_size}")
    print(f"    mask calls: {mask_stats['mask_calls']}")
    print(f"    hashes: {mask_stats['hashes']}")
    print(f"    mask empty: {mask_stats['mask_empty']}")
    print(f"    mask nonempty: {mask_stats['mask_nonempty']}")
    hits = np.array([hit['hit'] for hit in mask_stats['mask_history_hitmiss']])
    misses = np.array([hit['miss'] for hit in mask_stats['mask_history_hitmiss']])
    hits_sum = np.sum(hits)
    misses_sum = np.sum(misses)
    hits_ratio = hits_sum / (hits_sum + misses_sum) if (hits_sum + misses_sum) > 0 else 0
    misses_ratio = misses_sum / (hits_sum + misses_sum) if (hits_sum + misses_sum) > 0 else 0
    print(f"    mask cache avg hit/miss: {hits_ratio:.2f}/{misses_ratio:.2f} (hits: {hits_sum}, misses: {misses_sum})")
    print(f"    mask cache median hit/miss: {np.median(hits):.2f}/{np.median(misses):.2f}")
    updates = np.array(mask_stats['mask_updates'])
    print(f"    mask cache updates: {np.sum(updates)} (avg: {np.mean(updates):.2f}, median: {np.median(updates):.2f})")
    empty_count = mask_stats['mask_empty']
    nonempty_count = mask_stats['mask_nonempty']
    if empty_count + nonempty_count > 0:
        empty_ratio = empty_count / (empty_count + nonempty_count)
        nonempty_ratio = nonempty_count / (empty_count + nonempty_count)
    else:
        empty_ratio = 0
        nonempty_ratio = 0
    print(f"    update empty/nonempty ratio: {empty_ratio:.2f}/{nonempty_ratio:.2f} (empty: {empty_count}, nonempty:  {nonempty_count})")


def select_n_actions(env:AssemblyGame, actor:MCTSActor, n):
    print(f"Selecting {n} actions in the environment {env} using actor {actor} and network {actor._network}.")
    ts = env.reset()
    for _ in tqdm(range(n)):
        action = actor.select_action(ts.observation)
        ts = env.step(action)
        if env._is_invalid:
            ts = env.reset()

def run_env_loop(env, actor, num_steps, loop=None):
    if loop is None:
        loop = EnvironmentLoop(env, actor)
    loop.run(num_episodes=num_steps)

def profile_env_loop(env, actor, num_steps):
    print("Profiling environment loop...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    run_env_loop(env, actor, num_steps)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print_mask_stats(actor._model._environment._action_space_storage)
    return stats

def profile_prepared_env_loop(loop: EnvironmentLoop, num_steps):
    print("Profiling prepared environment loop...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    loop.run(num_episodes=num_steps)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    # print_mask_stats(actor._model._environment._action_space_storage)
    return stats

def profile_select_n_actions(env, actor, num_actions):
    profiler = cProfile.Profile()
    profiler.enable()
    
    select_n_actions(env, actor, num_actions)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print_mask_stats(actor._model._environment._action_space_storage)
    return stats

def main(id_, actor=None, env_loop=None):
    if actor is None:
        actor = actor_10
    if env_loop is None:
        env = env_10
    else:
        env = env_loop._environment
    
    num_steps = 100
    num_episodes = 1 # x100
    
    print("Profiling select_action...")
    select_action_stats = profile_select_n_actions(env, actor, num_steps)
    select_action_stats.dump_stats(f'profile/select_action_profile_{id_}.prof')
    subprocess.run(['flameprof', '-i', f'profile/select_action_profile_{id_}.prof', '-o', f'profile/select_action_flamegraph_{id_}.svg'])
    print("Profiling Environment Loop...")
    if env_loop is not None:
        env_loop_stats = profile_prepared_env_loop(env_loop, num_episodes)
    else:
        env_loop_stats = profile_env_loop(env_10, actor_10, num_episodes)
    env_loop_stats.dump_stats(f'profile/env_loop_profile_{id_}.prof')
    subprocess.run(['flameprof', '-i', f'profile/env_loop_profile_{id_}.prof', '-o', f'profile/env_loop_flamegraph_{id_}.svg'])
    print("Profiling complete.")
    print("Profiles saved to 'profile/select_action_profile.prof' and 'profile/env_loop_profile.prof'.")
    
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        sys.argv.append('debug')
    print("Arguments:", sys.argv)
    if len(sys.argv) == 3:
        actor, env_loop = actor_env_from_config(sys.argv[2])
        print(f"Using actor {actor} and environment loop {env_loop}.")
        main(id_=sys.argv[1], actor=actor, env_loop=env_loop)
    else:
        print("No environment loop provided, using default actor and environment.")
        main(id_=sys.argv[1])