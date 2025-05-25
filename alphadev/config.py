from typing import Callable, Union, Optional, Literal
import functools
import yaml
import dataclasses
import ml_collections
import numpy as np
import portpicker
import tensorflow as tf

from acme.utils.loggers import make_default_logger, Logger
import acme.tf.utils as tf2_utils
from acme.specs import make_environment_spec

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
    num_latency_simulations: int = 10
    ### Self-Play
    num_actors: int = 1 
    max_moves: int = 100
    num_simulations: int = 5
    discount: float = 1.0

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
    # inference service
    make_inference_service: bool = False # if False, all actors will have their own version of the network
    inference_service_backend: Literal['redis', 'shm'] = 'redis'
    inference_service_name: str = 'inference'
    inference_accumulation_period: int = 1.0
    
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
        )
        
        self.logger_factory = logger_factory(self)
        self.env_observers, self.search_observers = make_observer_factories(self)
        
        self.distributed_backend_config = {
            'type': 'redis',
            'host': self.redis_host,
            'port': self.redis_port,
            'db': self.redis_db
        }
        
        if self.replay_server_port is None:
            self.replay_server_port = portpicker.pick_unused_port()
        
        if self.distributed:
            self.device_config = DeviceAllocationConfig(self)
    
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

class DeviceAllocationConfig:
    """
    When using distributed training, this class is used to pre-compute the device allocation for each concurrent component.
    
    What we expect:
    - 1 learner process.
    - N+1 actor processes (N for experience replay, 1 for evaluation).
    - OR, a single inference service that does inference for all actors.
    
    Before distributed training is launched, we perform the following steps:
    1. Construct an instance of the network on CPU to determine its size.
    2. Pre-compute device allocation for each component based on the number of components, number of available GPUs and the size of the network.
        - For the learner process, we assume network_size * 3 memory demand.
        - For each actor process, we assume network_size * 1.2 memory demand.
    3. Prepare callbacks to be called by each component when their corresponding process is launched.
    If no GPU is available, this is a NoOp.
    """
    ACTOR_PROCESS = 'actor'
    LEARNER_PROCESS = 'learner'
    
    def __init__(self, config: AlphaDevConfig):
        self.config = config
        self.num_actors = config.num_actors
        self.inference_mode = config.make_inference_service
        self.gpus = tf.config.list_physical_devices('GPU')
        # determine the network size in a new process so that we don't interfere with the main process
        self.device_allocations = self.compute_device_allocations()
        self.callbacks = self.make_callbacks()
    
    def _make_process_key(self, process_type: str, index: int = 0) -> str:
        return f"{process_type}_{index}"
    
    @staticmethod
    def device_config_callback(process_type, gpu, memory):
        print(f"Setting device for {process_type} to {gpu} with {memory} bytes.")
        tf.config.set_visible_devices(gpu, 'GPU')
        tf.config.experimental.set_memory_growth(gpu, True)
        
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)]
        )
    
    def _compute_network_size(self, batch_size: int = 1) -> int:
        """Compute the size of the network in bytes."""
        from .network import NetworkFactory
        from .environment import EnvironmentFactory
        print("Computing network size...")
        # create a new tf session temporarily to containerize this computation
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("Using GPU for network size computation.")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            network = NetworkFactory(self.config)(None)
            env = EnvironmentFactory(self.config)()
            env_spec = make_environment_spec(env)
            tf2_utils.create_variables(
                network, env_spec.observations,
                batch_size=batch_size,
            )
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            print("Peak Memory Usage:", memory_info['peak'])
            return memory_info['peak']
        else:
            print("No GPU available, returning 0 for network size.")
            return 0
    
    def compute_network_size(self, batch_size: int = 1) -> int:
        import multiprocessing as mp
        
        with mp.Pool(1) as pool:
            network_size = pool.apply(self._compute_network_size, args=(batch_size,))
        
        print(f"Network size: {network_size} bytes")
        return network_size
    
    def _get_total_memory(self):
        import subprocess
        
        try:
            # nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                encoding='utf-8', stderr=subprocess.DEVNULL
            )
            mem_mb = output.strip().split('\n')
            mem_bytes = [int(mb) * 1024 * 1024 for mb in mem_mb]
            return mem_bytes
        except subprocess.CalledProcessError:
            print("Failed to get GPU memory info, returning empty list.")
            return []
    
    def compute_device_allocations(self):
        """
        Compute the device allocations for each component.
        """
        if len(self.gpus) == 0:
            print("No GPUs available, returning empty device allocations.")
            return {}
        
        device_allocations = {}
        
        gpu_totals = self._get_total_memory()
        print(f"Total GPU memory available: {gpu_totals} bytes")
        if not gpu_totals:
            print("No GPU memory information available, returning empty device allocations.")
            return {}
        assert len(gpu_totals) == len(self.gpus), "Mismatch between number of GPUs and memory totals."
        
        gpu_available = dict(zip(self.gpus, gpu_totals))
        
        # TODO: would make more sense to distribute the memory without leaving any GPU vram empty
        
        learner_memory = self.compute_network_size(self.config.batch_size) * 3
        print(f"Learner process memory allocation: {learner_memory} bytes")
        if self.inference_mode:
            actor_memory = self.compute_network_size(self.config.batch_size) * 1.2
            print(f"Inference service memory allocation: {actor_memory} bytes")
        else:
            actor_memory = self.compute_network_size(1) * 1.2
        print(f"Actor process memory allocation: {actor_memory} bytes")
        # start with the learner process
        learner_gpu = self.gpus[0]
        device_allocations[self._make_process_key(self.LEARNER_PROCESS)] = {
            'gpu': learner_gpu,
            'memory': learner_memory
        }
        gpu_available[learner_gpu] -= learner_memory
        print(f"Remaining memory for actors: {gpu_available}")
        
        if self.inference_mode:
            if gpu_available[learner_gpu] < actor_memory:
                print(f"{learner_gpu} does not have enough memory for inference service, switching to next GPU.")
                gpu_available.pop(learner_gpu)
                if len(gpu_available) == 0:
                    raise RuntimeError("No more GPUs available for inference service, stopping allocation.")
                learner_gpu = next(iter(gpu_available))
            device_allocations[self._make_process_key('inference_service')] = {
                'gpu': learner_gpu,
                'memory': actor_memory
            }
            gpu_available[learner_gpu] -= actor_memory
            print(f"Allocated {actor_memory} bytes to inference service on {learner_gpu}. Remaining memory: {gpu_available[learner_gpu]} bytes")
        
        # allocate memory for each actor process
        crnt_gpu = learner_gpu
        for i in range(self.num_actors+1):
            remaining = gpu_available[crnt_gpu]
            if remaining < actor_memory:
                print(f"{crnt_gpu} does not have any memory left for actor {i}, switching to next GPU.")
                gpu_available.pop(crnt_gpu)
                if len(gpu_available) == 0:
                    raise RuntimeError("No more GPUs available for actors, stopping allocation.")
                crnt_gpu = next(iter(gpu_available))
            device_allocations[self._make_process_key(self.ACTOR_PROCESS, i)] = {
                'gpu': crnt_gpu,
                'memory': actor_memory
            }
            gpu_available[crnt_gpu] -= actor_memory
            print(f"Allocated {actor_memory} bytes to actor {i} on {crnt_gpu}. Remaining memory: {gpu_available[crnt_gpu]} bytes")
        print(f"Final device allocations: {device_allocations}")
        return device_allocations

    def make_callbacks(self):
        """
        Create callbacks for each component to set the device allocation.
        """
        callbacks = {}
        for process_type, allocation in self.device_allocations.items():
            gpu = allocation['gpu']
            memory = allocation['memory']
            callbacks[process_type] = functools.partial(
                self.device_config_callback,
                process_type=process_type,
                gpu=gpu,
                memory=memory
            ) 
        return callbacks

    def get_callback(self, process_type: str, index: int = 0) -> Optional[Callable]:
        """
        Get the callback for a specific process type.
        """
        key = self._make_process_key(process_type, index)
        return self.callbacks.get(key, None)
