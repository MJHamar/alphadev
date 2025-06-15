"""Performance tests for the environment."""
import os
from time import time
import cProfile
import pstats
import subprocess
from tqdm import tqdm

import tensorflow as tf
for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
import sonnet as snn
import numpy as np

from alphadev.environment import AssemblyGame, EnvironmentFactory, AssemblyGameModel, ModelFactory
from alphadev.acting import MCTSActor
from alphadev.agents import MCTS
from alphadev.search.mcts import PUCTSearchPolicy
from alphadev.utils import TaskSpec, generate_sort_inputs
from alphadev.config import AlphaDevConfig
from alphadev.network import AlphaDevNetwork, NetworkFactory, make_input_spec
from alphadev.alphadev_acme import make_agent
from acme.specs import make_environment_spec
from acme.environment_loop import EnvironmentLoop

CFG_PATH = f'{os.path.dirname(__file__)}/apv_mcts_config.yaml'

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

config = AlphaDevConfig.from_yaml(CFG_PATH)

# realistic scenario
def actor_env_from_config(path) -> EnvironmentLoop:
    """Load the configuration from a file."""
    config = AlphaDevConfig.from_yaml(path)
    assert not config.distributed, "This function is meant to be used with a single-threaded configuration."
    agent = make_agent(config)
    environment = EnvironmentFactory(config)()
    env_executor = EnvironmentLoop(
        environment=environment,
        actor=agent._actor,
        counter=agent.counter,
        should_update=False,
        logger=agent.logger,
        observers=[]
    )
    return agent, env_executor


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
    # print_mask_stats(actor._model._environment._action_space_storage)
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
    # print_mask_stats(actor._model._environment._action_space_storage)
    return stats

if __name__ == '__main__':
    import sys
    
    prof_out_dir = os.path.abspath(os.path.join('profile', sys.argv[1]))
    select_prof_out_dir = os.path.join(prof_out_dir, 'select_action')
    env_prof_out_dir = os.path.join(prof_out_dir, 'env_loop')
    
    
    num_steps = 100
    num_episodes = 1 # x100

    
    os.makedirs(select_prof_out_dir, exist_ok=True)
    os.environ['PROFILER_OUTPUT_DIR'] = select_prof_out_dir
    print(f"Profiler output directory: {select_prof_out_dir}")
    
    # agent, env_loop = actor_env_from_config(sys.argv[2])
    
    # print(f"Using actor {agent._actor} and environment loop {env_loop}.")
    # print("Profiling select_action...")
    
    # select_start = time()
    # select_action_stats = profile_select_n_actions(env_loop._environment, agent._actor, num_steps)
    # select_end = time()
    
    # select_action_stats.dump_stats(f'{select_prof_out_dir}/select_action_profile.prof')
    # subprocess.run(['flameprof', '-i', f'{select_prof_out_dir}/select_action_profile.prof', '-o', f'{select_prof_out_dir}/select_action_flamegraph.svg'])
    
    
    os.makedirs(env_prof_out_dir, exist_ok=True)
    os.environ['PROFILER_OUTPUT_DIR'] = env_prof_out_dir
    print(f"Profiler output directory: {env_prof_out_dir}")
    
    agent, env_loop = actor_env_from_config(sys.argv[2])
    print(f"Using actor {agent._actor} and environment loop {env_loop}.")
    print("Profiling Environment Loop...")

    env_start = time()
    env_loop_stats = profile_prepared_env_loop(env_loop, num_episodes)
    env_end = time()

    env_loop_stats.dump_stats(f'{prof_out_dir}/env_loop_profile.prof')
    subprocess.run(['flameprof', '-i', f'{env_prof_out_dir}/env_loop_profile.prof', '-o', f'{env_prof_out_dir}/env_loop_flamegraph.svg'])
    
    print("Profiling complete.")
    # print(f"Select Action Time: {select_end - select_start:.2f} seconds.")
    print(f"Environment Loop Time: {env_end - env_start:.2f} seconds.")
