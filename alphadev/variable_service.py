"""Variable service for persistent storage of variables.
This service will be used by the learner and inference server to store
and retrieve variables.
"""
import redis
import pickle
import time

from .service import RPCService, RPCClient, _RedisRPCClient
from .config import AlphaDevConfig

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VariableService():
    def __init__(self, config: AlphaDevConfig):
        self._config = config
        self._variable_key = f'{config.variable_service_name}_variables'
        self._redis_config = config.connection_config
        assert self._redis_config['type'] == 'redis', "Only redis is supported for variable storage"
    
    def connect(self):
        """Connect to the redis server."""
        self._redis = redis.StrictRedis(
            host=self._redis_config['host'],
            port=self._redis_config['port'],
            db=self._redis_config['db'],
        )
    
    def close(self):
        if hasattr(self, '_redis'):
            self._redis.close()
            del self._redis
    
    def update(self, variables):
        """Update the variable storage with the given variables."""
        logger.debug(f"Updating variables in {self._variable_key}")
        self.connect()
        variables_bin = pickle.dumps(variables)
        self._redis.set(self._variable_key, variables_bin)
        self.close()
        return self._variable_key
    
    def has_variables(self):
        """Check if the variable storage has variables."""
        logger.debug(f"Checking if variables exist in {self._variable_key}")
        self.connect()
        has_vars = self._redis.exists(self._variable_key)
        self.close()
        return has_vars
    
    def get_variables(self):
        """Get the variables from the variable storage."""
        logger.debug(f"Getting variables from {self._variable_key}")
        self.connect()
        variables_bin = self._redis.get(self._variable_key)
        if variables_bin is None:
            return None
        variables = pickle.loads(variables_bin)
        self.close()
        return variables

class VariableServiceClient(RPCClient):
    def __init__(self, config: AlphaDevConfig):
        self._config = config
        self._service_name = config.variable_service_name
        self._redis_config = config.connection_config
        assert self._redis_config['type'] == 'redis', "Only redis is supported for variable storage"
        super().__init__(
            client=_RedisRPCClient(
                self._redis_config, f'{self._service_name}_queue', logger.getChild('client.redis')
            ), methods=['get_variables', 'has_variables', 'update'],
            logger=logger.getChild('client')
        )
        setattr(self, 'update', self._UpdateHandler('update', self._client, self.logger.getChild('update')))
    
    class _UpdateHandler(RPCClient._Handler):
        def __call__(self, *args, **kwargs):
            key = super().__call__(*args, **kwargs)
            if key is None:
                return None
            # get the variables from redis via the given key
            with self._client.connection() as conn:
                variables_bin = conn.get(key)
                if variables_bin is None:
                    return None
                variables = pickle.loads(variables_bin)
                return variables

def make_variable_service(config: AlphaDevConfig):
    return RPCService(
        conn_config=config.connection_config,
        instance_factory=VariableService,
        args=(config, ),
        instance_cls=VariableService,
        is_persistent=True,
        logger=logger,
    )
def make_variable_client(config: AlphaDevConfig):
    return VariableServiceClient(config)