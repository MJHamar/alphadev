"""
Entrypoint for service processes.
"""

import os
import sys
from absl import app
import argparse

import logging

def parse_args():
    """
    Parse command line arguments.
    
    Three arguments are expected:
    - `executable_path`: Path to the executable file.
    - `--label`: string identifier of the process.
    - `--device_name`: Name of the GPU device to use (e.g., 'cuda:0').
    - `--allocation_size`: Size of the memory allocation in bytes (e.g., '1024').
    """
    parser = argparse.ArgumentParser(description="Service process entry point.")
    parser.add_argument('executable_path', type=str, help='Path to the executable file.')
    parser.add_argument('--label', type=str, required=True, help='String identifier of the process.')
    parser.add_argument('--device_name', type=str, required=False, help='Name of the device to use (e.g., cuda:0).')
    parser.add_argument('--allocation_size', type=str, required=False, help='Size of the memory allocation in bytes (e.g., 1024).')
    args = parser.parse_args()
    assert ((args.device_name is not None and args.allocation_size is not None)
            or
            (args.device_name is None and args.allocation_size is None)), \
                "Both --device_name and --allocation_size must be provided together or not at all."
    return args

# parse arguments
config = parse_args()
print('arguments parsed:', config)
print(f"{config.label} starting with executable {config.executable_path}.")
logger = logging.getLogger(f'proc_{config.label}')

logging.basicConfig(
    level=logging.INFO,
)

# configure GPU runtime
import tensorflow as tf
if config.device_name is not None:
    tf.config.set_visible_devices([config.device_name], 'GPU')
    tf.config.experimental.set_memory_growth(config.device_name, True)

    tf.config.experimental.set_virtual_device_configuration(
        config.device_name,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=config.allocation_size)]
    )
    logger.info(f"{config.label} configured to use device {config.device_name} with allocation size {config.allocation_size} bytes.")
else:
    logger.info(f"{config.label} running without device configuration.")

# load the executable
import cloudpickle
with open(config.executable_path, 'rb') as f:
    executable = cloudpickle.load(f)

# clean stdin to avoid parsing unknown flags
sys.argv = sys.argv[:1]

# run the executable
app.run(executable,None)

logger.info(f"{config.label} exited.")
