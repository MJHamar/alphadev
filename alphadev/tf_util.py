"""Module to import tensorflow with a specific configuration."""

import sys
from .device_config import apply_device_config, get_device_config_from_cli

config = get_device_config_from_cli(sys.argv)

import tensorflow as tf

# tf = apply_device_config(tf, config)

# Tensorflow can now be imported in all the other modules.
