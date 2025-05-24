"""Variable service for persistent storage of variables.
This service will be used by the learner and inference server to store
and retrieve variables.
"""
import redis
import pickle
import time
import contextlib

from .config import AlphaDevConfig

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VariableService():
    """Variable service that stores variables in a Redis database.
    
    It defines three methods:
    - `update`: Update the variable storage with the given variables.
    - `has_variables`: Check if the variable storage has variables.
    - `get_variables`: Get the variables from the variable storage.
    """
    def __init__(self, config: AlphaDevConfig):
        self._config = config
        self._variable_key = f'{config.variable_service_name}'
        self._redis_config = config.distributed_backend_config
        assert self._redis_config['type'] == 'redis', "Only redis is supported for variable storage"
    
    @contextlib.contextmanager
    def connection(self):
        """Context manager for the Redis client."""
        try:
            client = redis.Redis(
                host=self._redis_config['host'],
                port=self._redis_config['port'],
                db=self._redis_config['db']
            )
            client.ping()
            yield client
        finally:
            client.close()
            del client
    
    def update(self, variables):
        """Update the variable storage with the given variables."""
        logger.debug(f"Updating variables in {self._variable_key}")
        with self.connection() as conn:
            variables_bin = pickle.dumps(variables)
            conn.set(self._variable_key, variables_bin)
            return self._variable_key
    
    def has_variables(self):
        """Check if the variable storage has variables."""
        logger.debug(f"Checking if variables exist in {self._variable_key}")
        with self.connection() as conn:
            variables_bin = conn.exists(self._variable_key)
            if variables_bin == 0:
                return False
            else:
                return True
    
    def get_variables(self, keys=None):
        # NOTE: keys is unused because it is also unused in AZLearner.
        """Get the variables from the variable storage."""
        logger.debug(f"Getting variables from {self._variable_key}")
        with self.connection() as conn:
            variables_bin = conn.get(self._variable_key)
            if variables_bin is None:
                raise ValueError("No variables found in variable storage")
            else:
                variables = pickle.loads(variables_bin)
                return variables