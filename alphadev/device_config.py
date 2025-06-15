from enum import Enum
import yaml

class ProcessType(Enum):
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
    
    ad_config = AlphaDevConfig.from_yaml(ad_config_path)
    
    cpu_device = tf.config.list_physical_devices('CPU')[0]
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    config = {}
    # 1. controller configuration
    config[ProcessType.CONTROLLER] = {
        "device_name": cpu_device.name,
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
        config[ProcessType.LEARNER] = config[ProcessType.ACTOR] = config[ProcessType.CONTROLLER]
        return config
    users_per_gpu = num_gpu_users // num_gpus
    if users_per_gpu == 0: # the unlikely case when there are more GPUs than users
        users_per_gpu = 1
    gpu_sizes = [gpu_devices[i % num_gpus].memory_limit // users_per_gpu for i in range(num_gpu_users)]
    # 4. assign the GPUs to the processes
    config[ProcessType.LEARNER] = {
        "device_name": gpu_devices[0].name,
        "allocation_size": gpu_sizes[0],
    }
    # 5. assign the GPUs to the actors
    config[ProcessType.ACTOR] = [{
        "device_name": gpu_devices[i].name,
        "allocation_size": gpu_sizes[i],
    } for i in range(1, num_gpu_users)]
    
    return config

class DeviceConfig:
    def __init__(self, path: str):
        self.config = yaml.safe_load(open(path, 'r'))
    
    def get_config(self, process_type: ProcessType):
        # TODO: support for multiple devices
        # right now only one will be used even if there are multiple.
        dev_cfg = self.config[process_type]
        if isinstance(dev_cfg, list):
            # for actor processes, return the first element.
            return dev_cfg[0]
        return dev_cfg

def apply_device_config(local_tf, device_name=None, allocation_size=None):
    if device_name is None:
        # set CPU as the only visible device
        local_tf.config.set_visible_devices([], 'GPU')
        return local_tf
    if allocation_size is not None:
        # set the visible device with a specific allocation size
        local_tf.config.set_visible_devices([local_tf.config.PhysicalDevice(device_name, memory_limit=allocation_size)], 'GPU')
    else:
        # if no allocation size is specified, set memory growth to True
        local_tf.config.set_visible_devices([local_tf.config.PhysicalDevice(device_name)], 'GPU')
        local_tf.config.experimental.set_memory_growth(local_tf.config.PhysicalDevice(device_name) , True)
    
    return local_tf

def get_device_config_from_cli(args):
    """Parse device config-related arguments and remove them"""
    config = {}
    if '--device_name' in args:
        device_name_index = args.index('--device_name') + 1
        config['device_name'] = args[device_name_index]
        args.pop(device_name_index)
        args.pop(device_name_index - 1)
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
        config['device_name'] = None
        config['allocation_size'] = None
        config['allocation_ratio'] = None
    return config
