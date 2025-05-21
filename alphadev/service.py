"""
Lightweight implementation of Service objects that wrap different components
of the distributed training pipeline.
"""
import reverb
import redis
import signal
import pickle
import threading
import contextlib
import multiprocessing
from uuid import uuid4 as uuid

import logging
base_logger = logging.getLogger(__name__)

class _RedisRPCClient:
    """Redis implementation of the RPC client."""
    def __init__(self, redis_config, server_queue, logger=None):
        self._redis_config = redis_config
        self._server_queue = server_queue
        self._logger = logger or base_logger.getChild(f'{self.__class__.__name__}.{server_queue}')
    
    def connect(self):
        self._client = redis.Redis(
            host=self._redis_config['host'],
            port=self._redis_config['port'],
            db=self._redis_config['db']
        )
        self._client.ping()
        self._logger.debug('Connected to Redis server at %s:%s', self._redis_config['host'], self._redis_config['port'])
    
    def close(self):
        if hasattr(self, '_client'):
            self._client.close()
            self._logger.debug('Closed connection to Redis server at %s:%s', self._redis_config['host'], self._redis_config['port'])
    
    def rpc(self, payload):
        self.connect()
        try:
            # Send the payload to the Redis server
            call_id = uuid()
            rpc_call = {
                'id': call_id,
                'payload': payload
            }
            self._client.rpush(self._server_queue, rpc_call)
            self._logger.debug('Sent RPC call to Redis server: %s', rpc_call)
        except redis.exceptions.ConnectionError as e:
            self._logger.error('Failed to send RPC call to Redis server: %s', e)
            raise RuntimeError('Failed to send RPC call to Redis server') from e
        finally:
            self.close()
        try:
            # Wait for the response
            response = self._client.blpop(call_id, timeout=5)
            if response:
                self._logger.debug('Received response from Redis server: %s', response)
                return response[0]
            else:
                self._logger.error('No response received from Redis server')
                raise RuntimeError('No response received from Redis server')
        except redis.exceptions.TimeoutError as e:
            self._logger.error('Timeout while waiting for response from Redis server: %s', e)
            raise RuntimeError('Timeout while waiting for response from Redis server') from e
        finally:
            self.close()

class _RedisRPCService:
    """Redis implementation of the RPC service."""
    def __init__(self, redis_config, label, logger=None):
        self._redis_config = redis_config
        self._label = label
        self._server_queue = f'{label}_queue'
        self._close_key = f'{label}_close'
        self.connect()
        self._client.set(self._close_key, 0)
        self._logger = logger or base_logger.getChild(f'{self.__class__.__name__}.{label}')
    
    def connect(self):
        self._client = redis.Redis(
            host=self._redis_config['host'],
            port=self._redis_config['port'],
            db=self._redis_config['db']
        )
        self._client.ping()
    
    def close(self):
        if hasattr(self, '_client'):
            self._client.close()
    
    def set_close(self):
        # Set the close key to 1 to signal the service to stop
        self._client.set(self._close_key, 1)
        self._logger.debug('Set close key for Redis service: %s', self._close_key)
    
    @property
    def should_close(self):
        # Check if the close key is set to 1
        return self._client.get(self._close_key) != 0
    
    def get_client(self):
        return _RedisRPCClient(self._redis_config, self._server_queue)
    
    def return_rpc(self, key, payload):
        # Return the payload to the Redis server
        try:
            self._client.rpush(key, payload)
            self._logger.debug('Returned RPC call to Redis server: %s', payload)
        except redis.exceptions.ConnectionError as e:
            self._logger.error('Failed to return RPC call to Redis server: %s', e)
            raise RuntimeError('Failed to return RPC call to Redis server') from e
    
    def next(self, timeout=5):
        # wait for the next request from the Redis server
        # or return None if timeout is reached
        try:
            msg = self._client.blpop(self._server_queue, timeout=timeout)
            if msg:
                self._logger.debug('Received request from Redis server: %s', msg)
                msg = msg[0]
                return_id = msg['id']
                payload = msg['payload']
                return lambda p: self._return_rpc(return_id, p), payload
            else:
                return None
        except redis.exceptions.TimeoutError as e:
            return None

class RPCClient:
    """
    Exposes a handle for each public method of registered at the corresponding service.

    The handle is a callable that takes the same arguments as the
    corresponding method and returns the result of calling that method on the
    underlying component.
    """
    
    def __init__(self, client, methods, logger=None):
        self._client = client
        self._methods = methods
        self._set_handlers(methods)
        self._logger = logger or base_logger.getChild(f'{self.__class__.__name__}')
    
    class _Handler:
        """
        Handle the request by calling the corresponding method on the client.
        """
        def __init__(self, method, client, logger=None):
            self._method = method
            self._client = client
            self._logger = logger or base_logger.getChild(f'{self.__class__.__name__}.{method}')
        
        def __call__(self, *args, **kwargs):
            
            # Call the method on the client
            payload = {
                'method': self._method,
                'arguments': {
                    'args': args,
                    'kwargs': kwargs
                }
            }
            payload_bin = pickle.dumps(payload)
            # Send the payload to the client and block until a response is received.
            result_bin = self._client.rpc(payload_bin)
            assert result_bin is not None and len(result_bin) > 0, 'No result received from client'
            result = pickle.loads(result_bin)
            return result
    
    def _set_handlers(self, methods):
        """
        Sets callable attributes for each method in the methods dictionary.
        """
        for method in methods:
            # Create a callable attribute for each method
            setattr(self, method, self._Handler(method, self._client))

class RPCService:
    """
    Each service object exposes a handle for each public method of the underlying 
    component. 
    
    The handle is a callable that takes the same arguments as the
    corresponding method and returns the result of calling that method on the
    underlying component via http requests.
    
    Each service comes with a run method that starts the service, by spawning a
    thread that runs the service. The run method is called by the main thread
    when the service is started.
    The service is stopped by calling the stop method, which stops the thread
    and cleans up the resources used by the service.
    """
    _registered_methods = {}
    
    def __init__(self, label:str, conn_config:dict, instance_factory:callable,
                 worker_polling_interval:float=1.0,
                 logger=None):
        self._label = label
        self._conn_config = conn_config
        self._worker_polling_interval = worker_polling_interval
        self._service = self._make_service(conn_config)
        self._registered_methods = self._discover_methods(instance_factory)
        self._instance_factory = instance_factory
        self._should_run = self._registered_methods.pop('run', None) is not None
        self._logger = logger or base_logger.getChild(f'{self.__class__.__name__}.{label}')
        
    def _make_service(self, conn_config):
        type_ = conn_config.pop('type', 'redis')
        if type_ == 'redis':
            return _RedisRPCService(conn_config, self._label, self._logger.getChild('service'))
        else:
            raise ValueError(f'Unknown service type: {type_}')
    
    @property
    def _should_stop(self):
        return self._service.should_close
    
    def _worker(self):
        while not self._should_stop:
            # Extract the return function and payload from the incoming request
            return_func, payload = self._service.next(timeout=self._worker_polling_interval)
            method = payload.pop('method')
            arguments = payload.pop('arguments', b'')
            assert len(args) > 0, 'No arguments provided'
            # unpickle the args
            arguments = pickle.loads(arguments)
            args = arguments['args']
            kwargs = arguments['kwargs']
            # call the method
            result = self._registered_methods[method](*args, **kwargs)
            # pickle the result
            result_bin = pickle.dumps(result)
            # Call the return function with the payload
            return_func(result_bin)
    
    def run(self):
        """
        Run the service. This method is called by the main thread when the
        service is started.
        """
        instance = self._instance_factory()
        if self._should_run:
            # span a worker thread to run the instance
            self._runner = threading.Thread(target=instance.run)
            self._runner.daemon = True
            self._runner.start()
        # start the service
        self._service.connect()
        # start the worker thread
        self._worker_thread = threading.Thread(target=self._worker)
        self._worker_thread.daemon = False 
        self._worker_thread.start()
        self._logger.info('Service %s started', self._label)
    
    def stop(self):
        """
        Stop the service. This method is called by the main thread when the
        service is stopped.
        """
        self._service.set_close()
        if self._should_run:
            self._runner.join(timeout=5)
        self._worker_thread.join(timeout=5)
        self._service.close()
        self._logger.info('Service %s stopped', self._label)
    
    def create_handle(self):
        """
        Create a handle for the service.
        """
        return RPCClient(self._service.get_client(), self._registered_methods.keys())
    
    def _discover_methods(self, instance_factory):
        instance = instance_factory()
        methods = {}
        for method_name in dir(instance):
            method = getattr(instance, method_name)
            if callable(method) and not method_name.startswith('_'):
                methods[method_name] = method
                self._logger.info('Exposing method: %s', method_name)
            elif method_name == '__call__':
                # also expose the __call__ method, if any
                methods[method_name] = method
                self._logger.info('Exposing method: %s', method_name)
        return methods

class ReverbService():
    """
    Reverb service that is given a reverb table constructor and creates a reverb server
    
    When create_handle is called, it returns a reverb client that can be used to 
    interact with the reverb server.
    """
    def __init__(self, priority_tables_fn, checkpoint_ctor=None, checkpoint_time_delta_minutes=None):
        """
        Args:
            priority_tables_fn: A mapping from table name to function used to
                compute priorities for said table.
            checkpoint_ctor: Constructor for the checkpointer to be used. Passing None
                uses Reverb's default checkpointer.
            checkpoint_time_delta_minutes: Time between async (non-blocking)
                checkpointing calls.
        """
        # credits to dm-launchpad for the implementation, too bad it is not maintained anymore
        self._priority_tables_fn = priority_tables_fn
        self._checkpoint_ctor = checkpoint_ctor
        self._checkpoint_time_delta_minutes = checkpoint_time_delta_minutes
        # reverb can resolve the address itself.
    
    def run(self):
        """
        Run the reverb server and start the checkpointing thread if needed.
        """
        # credits to dm-launchpad for the implementation
        priority_tables = self._priority_tables_fn()
        if self._checkpoint_ctor is None:
            checkpointer = None
        else:
            checkpointer = self._checkpoint_ctor()
        self._server = reverb.Server(
            tables=priority_tables,
            checkpointer=checkpointer,
        )
    
    def stop(self):
        """
        Stop the reverb server and clean up the resources used by the service.
        """
        self._server.stop()
    
    def create_handle(self):
        return self._server.localhost_client()

class Program(object):
    """
    Program class that manages the services and returns their handles.
    """
    def __init__(self):
        self._services = {}
        self._service_processes = {}
        self._current_group = None
        self._group_members = 0
    
    def add_service(self, service: RPCService):
        """
        Add a service to the program.
        """
        new_label = f'{self._current_group}/{self._group_members}.{service._label}'
        service._label = new_label
        self._services[new_label] = service
        return service.create_handle()
    
    def launch(self):
        """
        Run all the services in the program in a separate process.
        """
        for name, service in self._services.items():
            base_logger.info('Starting service %s', name)
            proc = multiprocessing.Process(target=service.run)
            proc.start()
            self._service_processes[name] = proc
            base_logger.info('Service %s started', name)
    
    def stop(self):
        """
        Stop all the services in the program.
        """
        for service in self._service_processes.values():
            try:
                # send a SIGINT signal to the process
                service.terminate()
                service.join(timeout=5)
            except Exception as e:
                base_logger.error('Failed to stop service %s: %s', service._label, e)
                continue
        self._services.clear()
    
    @contextlib.contextmanager
    def group(self, label: str):
        """
        Creates a group for a collection of homogeneous nodes.
        
        credits to dm-launchpad for the implementation.
        """
        if not label:
            raise ValueError('Label should not be empty.')
        if self._current_group:
            raise ValueError('group() cannot be nested.')
        try:
            self._current_group = label
            self._group_members = 0
            yield
        finally:
            # Try/finally is to make sure that the current_group is correctly
            # reset even if an exception occurs.
            self._current_group = None
            self._group_members = 0
