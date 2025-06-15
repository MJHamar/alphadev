"""
Entrypoint for service processes.
"""

# import the tf_util module to make sure the device is configured properly
from ..tf_util import tf

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
    args = parser.parse_args()
    return args

# parse arguments
config = parse_args()
print('arguments parsed:', config)
print(f"{config.label} starting with executable {config.executable_path}.")
logger = logging.getLogger(f'proc_{config.label}')

logging.basicConfig(
    level=logging.INFO,
)

# load the executable
import cloudpickle
with open(config.executable_path, 'rb') as f:
    executable = cloudpickle.load(f)

# clean stdin to avoid parsing unknown flags
sys.argv = sys.argv[:1]

# run the executable
app.run(executable)

logger.info(f"{config.label} exited.")
