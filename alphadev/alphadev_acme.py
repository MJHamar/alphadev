"""
Main script for running AlphaDev with an ACME and reverb backend.
"""
import sonnet as snn
import tensorflow as tf

import numpy as np
import ml_collections

from acme.specs import EnvironmentSpec, make_environment_spec, Array, BoundedArray, DiscreteArray
from acme.agents.tf.mcts import models

from tinyfive.multi_machine import multi_machine
from .agents import MCTS, DistributedMCTS # copied from github (not in the dm-acme package)

from .config import AlphaDevConfig
from .search import PUCTSearchPolicy
from .network import AlphaDevNetwork, NetworkFactory
from .environment import AssemblyGame, AssemblyGameModel, EnvironmentFactory, ModelFactory

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# fix for memory problems

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print("Setting memory growth for GPU: ", gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# #################
# Agent definition
# #################
class optimizer_factory:
    def __init__(self, learning_rate: float, momentum: float): self._learning_rate = learning_rate; self._momentum = momentum
    def __call__(self): return snn.optimizers.Momentum(learning_rate=self._learning_rate, momentum=self._momentum)

def make_agent(config: AlphaDevConfig):
    # -- create factories
    env_factory = EnvironmentFactory(config)
    net_factory = NetworkFactory(config)
    mod_factory = ModelFactory(config)
    opt_factory = optimizer_factory(config.lr_init, config.momentum)
    env_spec = make_environment_spec(env_factory())
    # -- search policy
    search_policy = PUCTSearchPolicy(config.pb_c_base, config.pb_c_init)
    
    if config.distributed:
        return DistributedMCTS(
            environment_factory=env_factory,
            network_factory=net_factory,
            model_factory=mod_factory,
            optimizer_factory=opt_factory,
            search_policy=search_policy,
            temperature_fn=config.temperature_fn,
            num_actors=config.num_actors,
            num_simulations=config.num_simulations,
            batch_size=config.batch_size,
            prefetch_size=config.prefetch_size,
            target_update_period=config.target_update_period,
            samples_per_insert=config.samples_per_insert,
            min_replay_size=config.min_replay_size,
            max_replay_size=config.max_replay_size,
            importance_sampling_exponent=config.importance_sampling_exponent,
            priority_exponent=config.priority_exponent,
            n_step=config.n_step,
            learning_rate=config.lr_init,
            discount=config.discount,
            environment_spec=env_spec,
            variable_update_period=config.variable_update_period,
            use_dual_value_network=config.hparams.categorical_value_loss,
            logger_factory=config.logger_factory,
            observers=config.env_observers,
            mcts_observers=config.search_observers,
    )
    else:
        cfg_logger = config.logger_factory()
        return MCTS(
            network=net_factory(None),
            model=mod_factory(None),
            optimizer=opt_factory(),
            n_step=config.n_step,
            discount=config.discount,
            replay_capacity=config.max_replay_size, # TODO
            num_simulations=config.num_simulations,
            environment_spec=env_spec,
            search_policy=search_policy,
            temperature_fn=config.temperature_fn,
            batch_size=config.batch_size,
            use_dual_value_network=config.hparams.categorical_value_loss,
            logger=cfg_logger,
            mcts_observers=config.search_observers,
        )

def run_single_threaded(config: AlphaDevConfig, agent: MCTS):
    environment = AssemblyGame(config.task_spec)
    
    num_episodes = config.training_steps
    for episode in range(num_episodes):
        # a. Reset environment and agent at start of episode
        logger.info("Initializing episode...")
        timestep = environment.reset()
        agent._actor.observe_first(timestep)
        
        # b. Run episode
        while not timestep.last():
            # Agent selects an action
            action = agent.select_action(timestep.observation)
            # logger.info("ed %d: %s len %d act %d", episode, timestep.step_type, timestep.observation['program_length'], action)
            # Environment steps
            new_timestep = environment.step(action)
            # logger.info("New timestep:", new_timestep)
            # Agent observes the result
            agent.observe(action=action, next_timestep=new_timestep)
            # Update timestep
            timestep = new_timestep

        # c. Train the learner
        logger.info("Final timestep reached: %s reward: %s", timestep.step_type, timestep.reward.numpy())
        logger.info("Training agent...")
        agent._learner.step()

        # d. Log training information (optional)
        logger.info(f"Episode {episode + 1}/{num_episodes} completed.")

def run_distributed(config: AlphaDevConfig, agent: DistributedMCTS):
    # build the distributed agent
    program = agent.build(config.distributed_backend_config)
    # run the distributed agent
    program.launch()

def run_alphadev(config: AlphaDevConfig):
    # -- define agent
    agent = make_agent(config)
    # -- run
    if config.distributed:
        # run in distributed mode
        run_distributed(config, agent)
    else:
        # run in single-threaded mode
        run_single_threaded(config, agent)
    
    # -- save

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    
    # -- load config
    import sys
    args = sys.argv[1:]
    try:
        config_path = args[0]
        config = AlphaDevConfig.from_yaml(config_path)
    except Exception as e:
        print("No config file provided. Using default config.", e)
        config = AlphaDevConfig()
    # -- run alphadev
    run_alphadev(config)
