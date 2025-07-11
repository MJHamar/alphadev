"""
Main script for running AlphaDev with an ACME and reverb backend.
"""
import sonnet as snn
from .tf_util import tf


from acme.specs import make_environment_spec
from acme.environment_loop import EnvironmentLoop

from .agents import MCTS, DistributedMCTS # copied from github (not in the dm-acme package)

from .config import AlphaDevConfig
from .search.mcts import PUCTSearchPolicy
from .network import NetworkFactory
from .environment import AssemblyGame, EnvironmentFactory, ModelFactory
from .device_config import DeviceConfig

import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# #################
# Agent definition
# #################
class optimizer_factory:
    def __init__(self, learning_rate: float, momentum: float): self._learning_rate = learning_rate; self._momentum = momentum
    def __call__(self): return snn.optimizers.Momentum(learning_rate=self._learning_rate, momentum=self._momentum)

def make_agent(config: AlphaDevConfig):
    device_config = DeviceConfig(config.device_config_path)
    # -- create factories
    env_factory = EnvironmentFactory(config)
    net_factory = NetworkFactory(config)
    mod_factory = ModelFactory(config)
    opt_factory = optimizer_factory(config.lr_init, config.momentum)
    env_spec = make_environment_spec(env_factory())
    # -- search policy
    search_policy = PUCTSearchPolicy(config.pb_c_base, config.pb_c_init)
    
    if config.distributed:
        # -- distributed MCTS agent
        return DistributedMCTS(
            # device configuration for the different processes.
            device_config=device_config,
            # basic params
            environment_factory=env_factory,
            network_factory=net_factory,
            model_factory=mod_factory,
            optimizer_factory=opt_factory,
            environment_spec=env_spec,
            # search
            search_policy=search_policy,
            temperature_fn=config.temperature_fn,
            num_actors=config.num_actors,
            num_simulations=config.num_simulations,
            discount=config.discount,
            variable_update_period=config.variable_update_period,
            dirichlet_alpha=config.root_dirichlet_alpha,
            exploration_fraction=config.root_exploration_fraction,
            search_retain_subtree=config.search_retain_subtree,
            # training parameters
            do_eval_based_updates=config.do_eval_based_updates,
            evaluation_update_threshold=config.evaluation_update_threshold,
            evaluation_episodes=config.evaluation_episodes,
            
            do_train=config.do_train,
            training_steps=config.training_steps,
            batch_size=config.batch_size,
            prefetch_size=config.prefetch_size,
            use_target_network=config.use_target_network,
            target_update_period=config.target_update_period,
            samples_per_insert=config.samples_per_insert,
            min_replay_size=config.min_replay_size,
            max_replay_size=config.max_replay_size,
            use_prioritized_replay=config.use_prioritized_replay,
            priority_exponent=config.priority_exponent,
            n_step=config.n_step,
            learning_rate=config.lr_init,
            # APV MCTS parameters
            use_apv_mcts=config.use_async_search,
            apv_processes_per_pool=config.async_search_processes_per_pool,
            virtual_loss_const=config.async_seach_virtual_loss,
            # Inference server parameters
            use_inference_server=config.search_use_inference_server,
            search_batch_size=config.search_batch_size,
            search_buffer_size=config.search_buffer_size,
            # Other parameters
            use_dual_value_network=config.hparams.categorical_value_loss,
            logger_factory=config.logger_factory,
            observers=config.env_observers,
            mcts_observers=config.search_observers,
    )
    else:
        cfg_logger = config.logger_factory()
        return MCTS(
            device_config=device_config,
            model=mod_factory(None),
            network_factory=net_factory,
            optimizer=opt_factory(),
            search_policy=search_policy,
            temperature_fn=config.temperature_fn,
            n_step=config.n_step,
            discount=config.discount,
            replay_capacity=config.max_replay_size, # TODO
            environment_spec=env_spec,
            batch_size=config.batch_size,
            
            num_simulations=config.num_simulations,
            dirichlet_alpha=config.root_dirichlet_alpha,
            exploration_fraction=config.root_exploration_fraction,
            search_retain_subtree=config.search_retain_subtree,
            use_apv_mcts=config.use_async_search,
            apv_processes_per_pool=config.async_search_processes_per_pool,
            virtual_loss_const=config.async_seach_virtual_loss,
            
            use_inference_server=config.search_use_inference_server,
            search_batch_size=config.search_batch_size,
            search_buffer_size=config.search_buffer_size,
            
            training_steps=config.training_steps,
            use_dual_value_network=config.hparams.categorical_value_loss,
            use_target_network=config.use_target_network,
            target_update_period=config.target_update_period,
            logger=cfg_logger,
            mcts_observers=config.search_observers,
        )

def run_single_threaded(config: AlphaDevConfig, agent: MCTS):
    environment = AssemblyGame(config.task_spec)
    
    num_episodes = config.episode_accumulation_period
    num_steps = config.training_steps
    
    env_observers = config.env_observers(agent.logger)

    env_executor = EnvironmentLoop(
        environment=environment,
        actor=agent,
        counter=agent.counter,
        should_update=False,
        logger=agent.logger,
        observers=env_observers
    )
    for _ in range(num_steps):
        # generate data
        env_executor.run(num_episodes=num_episodes)
        # update agent
        agent._learner.step()
    
def run_distributed(config: AlphaDevConfig, agent: DistributedMCTS):
    # build the distributed agent
    program = agent.build(config)
    # run the distributed agent
    program.launch()
    program.stop()

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
    multiprocessing.set_start_method("spawn", force=True)
    
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
    config.verify_device_config()
    run_alphadev(config)
