from enum import Enum
import yaml


LEARNER = "learner"
ACTOR = "actor"
CONTROLLER = "controller"

def compute_device_config(ad_config_path: str):
    """
    To be called prior to running the alphadev pipeline.
    
    Computes a device configuration based on the configuration file.
    
    There are three types of configurations:
    - learner: uses the GPU for training.
    - actor: uses the GPU for inference.
    - controller: does not use the GPU but supervises the other processes.
    
    This method finds the expected number of GPU workers and divides the available GPUs evently among them.
    There is always one learner process.
    Depending on the mcts configuration, there is either
    - num_actors actor processes, in case of single-threaded MCTS or alphago-style APV MCTS,
    - num_actors * (async_search_processes_per_pool + 1) actor processes, in case of distributed MCTS.
        the +1 comes from the fact that the controller (APV_MCTS) also needs GPU to initialize the search tree.
    The controller process does not use the GPU.
    """
    # import tensorflow here
    import tensorflow as tf
    from .config import AlphaDevConfig
    import pynvml
    
    ad_config = AlphaDevConfig.from_yaml(ad_config_path)
    
    cpu_device = tf.config.list_physical_devices('CPU')[0]
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    config = {}
    # 1. controller configuration
    config[CONTROLLER] = {
        "device_type": "CPU",
        "device_id": 0,
        "allocation_size": None,
    }
    # 2. find the number of GPU users
    num_gpu_users = 1  # always one learner
    if not ad_config.use_async_search:
        num_gpu_users += ad_config.num_actors
    elif ad_config.search_use_inference_server:
        # alphadev-style APV MCTS. one GPU user for each actor.
        num_gpu_users += ad_config.num_actors
    else:
        # 'streamlined' MCTS. every search actor uses the GPU.
        num_gpu_users += ad_config.num_actors * (ad_config.async_search_processes_per_pool + 1)
    # 3. divide the GPUs among the GPU users
    num_gpus = len(gpu_devices)
    if num_gpus == 0:
        config[LEARNER] = config[ACTOR] = config[CONTROLLER]
        return config
    users_per_gpu = num_gpu_users // num_gpus
    if users_per_gpu == 0: # the unlikely case when there are more GPUs than users
        users_per_gpu = 1
    pynvml.nvmlInit()
    sizes = []
    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        sizes.append(mem_info.total)

    print(f"GPU device sizes:", sizes)
    per_process_alloc = [sizes[i % num_gpus] // users_per_gpu for i in range(num_gpu_users)]
    # 4. assign the GPUs to the processes
    config[LEARNER] = {
        "device_type": "GPU",
        "device_id": 0,
        "allocation_size": per_process_alloc[0],
    }
    # 5. assign the GPUs to the actors
    config[ACTOR] = [{
        "device_type": "GPU",
        "device_id": i % num_gpus,
        "allocation_size": per_process_alloc[i],
    } for i in range(1, num_gpu_users)]
    
    return config

class DeviceConfig:
    def __init__(self, path: str):
        self.num_actors = 0
        self.config = yaml.safe_load(open(path, 'r'))
    
    def get_config(self, process_type: str):
        # TODO: support for multiple devices
        # right now only one will be used even if there are multiple.
        dev_cfg = self.config[process_type]
        if process_type == ACTOR and isinstance(dev_cfg, list):
            # for actor processes, return the first element.
            cfg = dev_cfg[self.num_actors]
            self.num_actors += 1
            return cfg
        return dev_cfg

def apply_device_config(local_tf, config = None):
    print("Applying device configuration", config)
    if config is None:
        # set CPU as the only visible device
        local_tf.config.set_visible_devices([], 'GPU')
        return local_tf
    device = local_tf.config.list_physical_devices(config['device_type'])[config['device_id']]
    allocation_size = config.get('allocation_size', None)
    # set the visible device with a specific allocation size
    local_tf.config.set_visible_devices([device], config['device_type'])
    if allocation_size is not None:
        local_tf.config.experimental.set_memory_growth(device, True)
        local_tf.config.set_logical_device_configuration(
            device, [local_tf.config.LogicalDeviceConfiguration(memory_limit=allocation_size)])
    return local_tf

def get_device_config_from_cli(args):
    """Parse device config-related arguments and remove them"""
    config = {}
    if '--device_type' in args:
        device_type_index = args.index('--device_type') + 1
        config['device_type'] = args[device_type_index]
        args.pop(device_type_index)
        args.pop(device_type_index - 1)
        
        device_id_index = args.index('--device_id') + 1
        config['device_id'] = int(args[device_id_index])
        args.pop(device_id_index)
        args.pop(device_id_index - 1)
        
        if '--allocation_size' in args:
            allocation_size_index = args.index('--allocation_size') + 1
            config['allocation_size'] = int(args[allocation_size_index])
            args.pop(allocation_size_index)
            args.pop(allocation_size_index - 1)
        elif '--allocation_ratio' in args:
            allocation_ratio_index = args.index('--allocation_ratio') + 1
            config['allocation_ratio'] = float(args[allocation_ratio_index])
            args.pop(allocation_ratio_index)
            args.pop(allocation_ratio_index - 1)
    else:
        return None
    return config

def get_cli_args_from_config(config):
    if config is None:
        return []
    cli = ['--device_type', config['device_type']]
    cli += ['--device_id', str(config['device_id'])]
    if 'allocation_size' in config and config['allocation_size'] is not None:
        cli += ['--allocation_size', str(config['allocation_size'])]
    return cli

if __name__ == '__main__':
    import sys
    from .config import AlphaDevConfig
    ad_config_path = sys.argv[1] 
    ad_config = AlphaDevConfig.from_yaml(ad_config_path)
    device_config = compute_device_config(ad_config_path)
    print("Computed device configuration:")
    for process_type, cfg in device_config.items():
        print(f"{process_type}: {cfg}")
    with open(ad_config.device_config_path, 'w') as f:
        yaml.dump(device_config, f, default_flow_style=False)
    print(f"Device configuration saved to {ad_config.device_config_path}")
