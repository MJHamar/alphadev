import sys
import os
from alphadev.config import AlphaDevConfig
import tensorflow as tf
import yaml

import acme.tf.utils as tf2_utils
from acme.specs import make_environment_spec

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
        self.inference_mode = False # TODO: remove
        self.num_actors = config.num_actors if not self.inference_mode else 1
        self.gpus = tf.config.list_physical_devices('GPU')
        # determine the network size in a new process so that we don't interfere with the main process
        self.device_allocations = self.compute_device_allocations()
        
    @staticmethod
    def make_process_key(process_type: str, index: int = 0) -> str:
        return f"{process_type}_{index}"
    
    def _compute_network_size(self, batch_size: int = 1) -> int:
        """Compute the size of the network in bytes."""
        from .network import NetworkFactory, make_input_spec
        from .environment import EnvironmentFactory
        print("Computing network size...")
        # create a new tf session temporarily to containerize this computation
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("Using GPU for network size computation.")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            env = EnvironmentFactory(self.config)()
            env_spec = make_environment_spec(env)
            network = NetworkFactory(self.config)(make_input_spec(env_spec.observations))
            tf2_utils.create_variables(
                network, [env_spec.observations],
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
        gpu_totals = sum(gpu_available.values())
        # device_allocations[self._make_process_key('inference_service')] = {
        #     'gpu': learner_gpu,
        #     'memory': actor_memory
        # }
        learner_memory = self.compute_network_size(self.config.batch_size) * 3
        print(f"Learner process memory allocation: {learner_memory} bytes")
        if self.inference_mode:
            actor_memory = self.compute_network_size(self.config.batch_size) * 2
            print(f"Inference service memory allocation: {actor_memory} bytes")
        else:
            actor_memory = self.compute_network_size(1) * 2
            print(f"Actor process memory allocation: {actor_memory} bytes")
        
        min_slot_sizes = [learner_memory] + [actor_memory] * self.num_actors
        process_types = [self.LEARNER_PROCESS] + [self.ACTOR_PROCESS] * self.num_actors
        # iteratively, we distribute the processes (1 learner, N actors) evenly across the avilable GPUs,
        # keeping track of the total memory demand for each GPU
        gpu_allocations = {gpu: [] for gpu in self.gpus}
        # allocate the learner to the first GPU
        learner_gpu = self.gpus[0]
        gpu_allocations[learner_gpu].append(0)
        gpu_index = 1 if len(self.gpus) > 1 else 0
        for i in range(1, self.num_actors + 1):
            gpu = self.gpus[gpu_index]
            gpu_allocations[gpu].append(i)
            gpu_index = (gpu_index + 1) % len(self.gpus)
        print(f"GPU allocations: {gpu_allocations}")
        # now, gpu_allocations maps process indices to GPUs.
        # we are ready to compute the device allocations
        for gpu, indices in gpu_allocations.items():
            gpu_memory = gpu_available[gpu]
            slot_size = gpu_memory // len(indices)
            for index in indices:
                if slot_size < min_slot_sizes[index]:
                    raise ValueError(
                        f"Not enough memory on GPU {gpu} for process {process_types[index]}: "
                        f"required {min_slot_sizes[index]} bytes, available {slot_size} bytes."
                    )
                process_type = process_types[index]
                # decrement the index by 1 for the actor processes.
                idx = index-1 if index > 0 else 0
                device_allocations[self._make_process_key(process_type, idx)] = {
                    'gpu': gpu,
                    'memory': slot_size
                }
        print(f"Device allocations: {device_allocations}")
        return device_allocations

def apply_device_config(device_allocation: dict):
    """
    Apply the device allocation configuration to the current TensorFlow session.
    
    Args:
        device_allocation: A dictionary with keys 'gpu' and 'memory'
        corresponding to the GPU device and memory allocation for the process.
    """
    tf.config.experimental.set_memory_growth(
        device_allocation['gpu'], True
    )
    tf.config.set_visible_devices(
        device_allocation['gpu'], 'GPU'
    )
    tf.config.set_logical_device_configuration(
        device_allocation['gpu'],
        [tf.config.LogicalDeviceConfiguration(memory_limit=device_allocation['memory'])]
    )

def main():
    """
    Main entry point to configure device allocations prior to launching the distributed program.
    
    Expects a single argument:
    - config_yaml_path: Path to the configuration file
    
    Will save the device allocation config to the path specified in the configuration file under the key 'device_config_path'.
    """
    config = AlphaDevConfig.from_yaml(sys.argv[1])
    device_config = DeviceAllocationConfig(config)
    device_config_path = config.device_config_path
    with open(device_config_path, 'w') as f:
        yaml.dump(device_config.device_allocations, f)
    print(f"Device allocations saved to {device_config_path}")

if __name__ == '__main__':
    main()
    print("Device configuration completed successfully.")