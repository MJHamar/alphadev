"""Variable service for persistent storage of variables.
This service will be used by the learner and inference server to store
and retrieve variables.
"""
import redis
import pickle
import time
import contextlib
from uuid import uuid4 as uuid

from acme.tf.variable_utils import VariableClient as acmeVariableClient

from ..config import AlphaDevConfig

import logging
base_logger = logging.getLogger(__name__)
base_logger.setLevel(logging.DEBUG)

class VariableService():
    """Variable service that stores variables in a Redis database.
    
    It defines three methods:
    - `update`: Update the variable storage with the given variables.
    - `has_variables`: Check if the variable storage has variables.
    - `get_variables`: Get the variables from the variable storage.
    """
    def __init__(self, config: AlphaDevConfig):
        self._config = config
        self._variable_pointer = f'{config.variable_service_name}_{uuid().hex[:8]}'
        self.current_variable_key = b'' # local pointer to the current variable key. used to avoid fetching unnecessarily.
        self._redis_config = config.distributed_backend_config
        self._checkpoint_dir = config.checkpoint_dir
        self._checkpoint_every = config.checkpoint_every
        self._num_updates = 0
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
        base_logger.debug(f"Updating variables in {self._variable_pointer}")
        self._num_updates += 1
        if self._checkpoint_dir is not None and self._num_updates % self._checkpoint_every == 0:
            base_logger.info(f"Checkpointing variables to {self._checkpoint_dir}")
            with open(f"{self._checkpoint_dir}/variables_{self._num_updates}.pkl", "wb") as f:
                pickle.dump(variables, f)
        base_logger.debug(f"Storing variables in {self._variable_pointer}")
        with self.connection() as conn:
            variables_bin = pickle.dumps(variables)
            new_variable_key = f"{self._variable_pointer}_{self._num_updates}"
            # upload the new variables
            conn.set(new_variable_key, variables_bin)
            # set the pointer to the latest variables
            conn.set(self._variable_pointer, new_variable_key)
            # expire the previous variable key
            if hasattr(self, 'prev_variable_key'):
                conn.expire(self.prev_variable_key, self._config.variable_expiration_time)
            # set the previous variable key to the new one
            self.prev_variable_key = new_variable_key
            return self._variable_pointer
    
    def has_variables(self):
        """Check if the variable storage has variables."""
        base_logger.debug(f"Checking if variables exist in {self._variable_pointer}")
        with self.connection() as conn:
            variables_exist = conn.exists(self._variable_pointer)
            if variables_exist == 0:
                return False
            else:
                return True
    
    def get_variables(self, keys=None):
        # NOTE: keys is unused because it is also unused in AZLearner.
        """Get the variables from the variable storage."""
        if not self.has_variables():
            start = time.time()
            while not self.has_variables() and time.time() - start < 90: # wait for variables to be available
                time.sleep(1)
            if not self.has_variables():
                base_logger.error("Variables not found in variable storage after waiting for 90 seconds")
                raise ValueError("No variables found in variable storage")
        with self.connection() as conn:
            variables_key = conn.get(self._variable_pointer)
            if variables_key == self.current_variable_key:
                return None  # No new variables, return None
            self.current_variable_key = variables_key
            base_logger.debug(f"Fetching variables from key: {variables_key}")
            variables_bin = conn.get(variables_key)
            if variables_bin is None:
                raise ValueError("No variables found in variable storage")
            else:
                variables = pickle.loads(variables_bin)
                return variables

class VariableClient(acmeVariableClient):
    """Small extension of the acme VariableClient to allow for None variables received from the service."""

    def _copy(self, new_variables):
        if new_variables is None:
            # If new_variables is None, we do not update the variables.
            return 
        return super()._copy(new_variables)