import os
from typing import Callable, Union, Optional, Literal
import functools
import yaml
import dataclasses
import ml_collections
import numpy as np
import portpicker
import tensorflow as tf

from acme.utils.loggers import make_default_logger, Logger


from .utils import IOExample, TaskSpec, generate_sort_inputs, x86_opcode2int
from .observers import MCTSObserver, MCTSPolicyObserver, CorrectProgramObserver, NonZeroRewardObserver
from .loggers import WandbLogger

@dataclasses.dataclass
class AlphaDevConfig(object):
    """AlphaDev configuration."""

    experiment_name: str = 'AlphaDev-test'
    
    # Environment: spec of the Variable Sort 3 task
    num_inputs: int = 17
    num_mem: int = 14
    num_regs: int = 5
    items_to_sort: int = 3
    correct_reward: float = 1.0
    correctness_reward_weight: float = 2.0
    latency_reward_weight: float = 0.5
    latency_quantile: float = 0.05
    num_latency_simulations: int = 1
    emulator_mode: Literal['u8', 'i16', 'i32'] = 'i32'
    ### Self-Play
    num_actors: int = 1 
    max_moves: int = 100
    num_simulations: int = 5
    discount: float = 1.0
    search_retain_subtree: bool = True

    # Root prior exploration noise.
    root_dirichlet_alpha: float = 0.03
    root_exploration_fraction: float = 0.25

    # UCB formula
    pb_c_base: int = 19652
    pb_c_init: float = 1.25
    temperature_fn: Union[str, Callable] = 'visit_softmax_temperature_fn'

    ### Network architecture
    embedding_dim: int = 512
    # representation network
    representation_use_program: bool = True
    representation_use_locations: bool = True
    representation_use_locations_binary: bool = False
    representation_use_permutation_embedding: bool = False
    representation_repr_net_res_blocks: int = 8
    # Multi-Query Attention
    representation_attention_head_depth: int = 128
    representation_attention_num_heads: int = 4
    representation_attention_attention_dropout: bool = False
    representation_attention_position_encoding: str = 'absolute'
    representation_attention_num_layers: int = 6
    # Value head
    value_max: float = 3.0  # These two parameters are task / reward-
    value_num_bins: int = 301  # dependent and need to be adjusted.
    categorical_value_loss: bool = True # wheether to treat the value functions as a distribution

    ### Training
    training_steps: int = 1000 #int(1000e3)
    batch_size: int = 8
    n_step: int = 5 # TD steps
    lr_init: float = 2e-4
    momentum: float = 0.9
    
    ### Single-threaded training
    episode_accumulation_period: int = 2 # how many episodes to accumulate before training
    
    ### Distributed training
    distributed: bool = True # whether to use distributed training
    prefetch_size: int = 4
    variable_update_period: int = 50 # aka checkpoint interval
    target_update_period: int = 10 # aka target interval
    samples_per_insert: int = 1
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    importance_sampling_exponent: float = 0.2
    priority_exponent: float = 0.6
    # Distributed communication backend
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    # Replay server (port)
    replay_server_port: int = None
    # variable service
    variable_service_name: str = 'variable'
    # search single-threaded or asynchronous
    use_async_search: bool = True # if False, all actors will have their own version of the network
    async_search_num_pools: int = 1 # number of parallel search pools to use in async search
    async_search_processes_per_pool: Union[int, str] = 'auto' # number of processes per pool, 'auto' will distribute all available cores evently
    search_batch_size: int = 1
    # device config
    device_config_path: Optional[str] = 'device_config.yaml'
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = 'alphadev'
    wandb_entity: str = "hamar_m"
    wandb_tags: Optional[str] = None
    wandb_notes: Optional[str] = None
    wandb_mode: str = 'online'
    wanbd_run_id: Optional[str] = None
    # Observers
    observe_program_correctness: bool = True
    save_non_zero_reward_trajectories: bool = True
    # MCTS observers
    observe_mcts_policy: bool = True
    mcts_observer_ratio: float = 0.001
    # profiling
    do_mcts_profiling: bool = False

    def __post_init__(self):
        
        self.hparams = ml_collections.ConfigDict()
        self.hparams.embedding_dim = self.embedding_dim
        # representation network
        self.hparams.representation = ml_collections.ConfigDict()
        self.hparams.representation.use_program = self.representation_use_program
        self.hparams.representation.use_locations = self.representation_use_locations
        self.hparams.representation.use_locations_binary = self.representation_use_locations_binary
        self.hparams.representation.use_permutation_embedding = self.representation_use_permutation_embedding
        self.hparams.representation.repr_net_res_blocks = self.representation_repr_net_res_blocks
        # Multi-Query Attention
        self.hparams.representation.attention = ml_collections.ConfigDict()
        self.hparams.representation.attention.head_depth = self.representation_attention_head_depth
        self.hparams.representation.attention.num_heads = self.representation_attention_num_heads
        self.hparams.representation.attention.attention_dropout = self.representation_attention_attention_dropout
        self.hparams.representation.attention.position_encoding = self.representation_attention_position_encoding
        self.hparams.representation.attention.num_layers = self.representation_attention_num_layers
        # Value head
        self.hparams.value_max = self.value_max
        self.hparams.value_num_bins = self.value_num_bins
        self.hparams.categorical_value_loss = self.categorical_value_loss
        
        self.temperature_fn = self.visit_softmax_temperature_fn
        
        self.input_examples = generate_sort_inputs(
                items_to_sort=self.items_to_sort,
                max_len=self.num_mem,
                num_samples=self.num_inputs
            )
        
        self.task_spec = TaskSpec(
            max_program_size=self.max_moves,
            num_inputs=self.num_inputs,
            num_funcs=len(x86_opcode2int),
            num_locations=self.num_regs+self.num_mem,
            num_regs=self.num_regs,
            num_mem=self.num_mem,
            num_actions=252, # original value was 271
            correct_reward=self.correct_reward,
            correctness_reward_weight=self.correctness_reward_weight,
            latency_reward_weight=self.latency_reward_weight,
            latency_quantile=self.latency_quantile,
            num_latency_simulations=self.num_latency_simulations,
            inputs=self.input_examples,
            observe_reward_components=self.hparams.categorical_value_loss,
            emulator_mode=self.emulator_mode,
        )
        
        self.logger_factory = logger_factory(self)
        self.env_observers, self.search_observers = make_observer_factories(self)
        
        self.distributed_backend_config = {
            'type': 'redis',
            'host': self.redis_host,
            'port': self.redis_port,
            'db': self.redis_db
        }
        
        if self.use_async_search:
            if isinstance(self.async_search_processes_per_pool, str) and self.async_search_processes_per_pool == 'auto':
                self.async_search_processes_per_pool = os.cpu_count() // self.async_search_num_pools
            elif isinstance(self.async_search_processes_per_pool, int):
                pass
            else:
                raise ValueError("async_search_processes_per_pool must be 'auto' or an integer.")
        
        if self.replay_server_port is None:
            self.replay_server_port = portpicker.pick_unused_port()
    
    def verify_device_config(self):
        if self.distributed:
            if not os.path.exists(self.device_config_path):
                raise FileNotFoundError(
                    f"Device configuration file not found: {self.device_config_path}. "
                    "Please make sure to run the device configuration script first."
                )
            with open(self.device_config_path, 'r') as f:
                self.device_config = yaml.safe_load(f)
    
    @classmethod
    def from_yaml(cls, path) -> 'AlphaDevConfig':
        """Create a config from a ml_collections.ConfigDict."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            return cls(**config_dict)

    @staticmethod
    def visit_softmax_temperature_fn(steps): 
        return 1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25

class logger_factory:
    def __init__(self, config: AlphaDevConfig):
        self.config = config
    
    def __call__(self) -> Logger:
        if self.config.use_wandb:
            print('Creating wandb logger')
            wandb_config = {
                'project': self.config.wandb_project,
                'entity': self.config.wandb_entity,
                'name': self.config.experiment_name,
                'tags': self.config.wandb_tags,
                'notes': self.config.wandb_notes,
                'mode': self.config.wandb_mode,
            }
            if self.config.wanbd_run_id is not None:
                wandb_config['run_id'] = self.config.wanbd_run_id
            logger = WandbLogger(wandb_config)
        else:
            print('Creating terminal logger')
            logger = make_default_logger(self.config.experiment_name, time_delta=0.0)
        
        return logger

    @property
    def __name__(self):
        return 'logger_factory'

class env_observer_factory:
    def __init__(self, config: AlphaDevConfig):
        self.config = config
    
    def __call__(self, logger: Logger):
        observers = []
        if self.config.observe_program_correctness:
            observers.append(
                CorrectProgramObserver()
            )
        if self.config.save_non_zero_reward_trajectories:
            observers.append(
                NonZeroRewardObserver()
            )
        return observers

class search_observer_factory:
    def __init__(self, config: AlphaDevConfig):
        self.config = config
    
    def __call__(self, logger: Logger) -> MCTSPolicyObserver:
        observers = []
        if self.config.observe_mcts_policy:
            observers.append(
                MCTSPolicyObserver(logger, epsilon=self.config.mcts_observer_ratio)
            )
        return observers
    
def make_observer_factories(config: AlphaDevConfig):
    return env_observer_factory(config), search_observer_factory(config)
