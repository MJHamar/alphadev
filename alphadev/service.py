"""
Lightweight implementation of Service objects that wrap different components
of the distributed training pipeline.
"""
import reverb
import socket
import redis
import signal
import pickle
import threading
import contextlib
import multiprocessing
import portpicker
from uuid import uuid4 as uuid

import logging
base_logger = logging.getLogger(__name__)
base_logger.setLevel(logging.DEBUG)
no_logger = logging.getLogger('no_logger')
no_logger.setLevel(logging.INFO)

address_table = set()

class MaybeLogger:
    def __init__(self, logger=None):
        self._logger = logger or no_logger
    
    @property
    def logger(self):
        return self._logger
    
    def set_logger(self, logger):
        self._logger = logger

class _RedisRPCClient(MaybeLogger):
    """Redis implementation of the RPC client."""
    def __init__(self, redis_config, server_queue, logger=None):
        self._redis_config = redis_config
        self._server_queue = server_queue
        MaybeLogger.__init__(self, logger)
    
    def connect(self):
        self._client = redis.Redis(
            host=self._redis_config['host'],
            port=self._redis_config['port'],
            db=self._redis_config['db']
        )
        self._client.ping()
        self.logger.debug('Connected to Redis server at %s:%s', self._redis_config['host'], self._redis_config['port'])
    
    def close(self):
        if hasattr(self, '_client'):
            self._client.close()
            self.logger.debug('Closed connection to Redis server at %s:%s', self._redis_config['host'], self._redis_config['port'])
    
    def rpc(self, payload):
        self.connect()
        try:
            # Send the payload to the Redis server
            call_id = uuid().hex
            rpc_call = {
                'id': call_id,
                'payload': payload
            }
            rpc_call_bin = pickle.dumps(rpc_call)
            self._client.rpush(self._server_queue, rpc_call_bin)
            self.logger.debug('Sent RPC call to Redis server: %s', rpc_call)
        except redis.exceptions.ConnectionError as e:
            self.logger.error('Failed to send RPC call to Redis server: %s', e)
            raise RuntimeError('Failed to send RPC call to Redis server') from e
        try:
            # Wait for the response
            response = self._client.blpop(call_id, timeout=5)
            if response:
                self.logger.debug('Received response from Redis server: %s', response)
                return pickle.loads(response[1])
            else:
                self.logger.error('No response received from Redis server')
                raise RuntimeError('No response received from Redis server')
        except redis.exceptions.TimeoutError as e:
            self.logger.error('Timeout while waiting for response from Redis server: %s', e)
            raise RuntimeError('Timeout while waiting for response from Redis server') from e
        finally:
            self.close()

class _RedisRPCService(MaybeLogger):
    """Redis implementation of the RPC service."""
    def __init__(self, redis_config, label, logger=None):
        MaybeLogger.__init__(self, logger)
        self._redis_config = redis_config
        self._label = label
        self._server_queue = f'{label}_queue'
        self._close_key = f'{label}_close'
        self.connect()
        self._client.set(self._close_key, 0)
        self.close()
    
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
            del self._client
    
    def set_close(self):
        # Set the close key to 1 to signal the service to stop
        self._client.set(self._close_key, 1)
        self.logger.debug('Set close key for Redis service: %s', self._close_key)
    
    @property
    def should_close(self):
        # Check if the close key is set to 1
        return self._client.get(self._close_key) == 0
    
    def get_client(self):
        return _RedisRPCClient(self._redis_config, self._server_queue)
    
    def _return_rpc(self, key, payload):
        # Return the payload to the Redis server
        try:
            payload_bin = pickle.dumps(payload)
            self._client.rpush(key, payload_bin)
            self.logger.debug('Returned RPC call to Redis server: %s', payload)
        except redis.exceptions.ConnectionError as e:
            self.logger.error('Failed to return RPC call to Redis server: %s', e)
            raise RuntimeError('Failed to return RPC call to Redis server') from e
    
    def next(self, timeout=5):
        # wait for the next request from the Redis server
        # or return None if timeout is reached
        try:
            msg_bin = self._client.blpop(self._server_queue, timeout=timeout)
            
            if msg_bin:
                msg = pickle.loads(msg_bin[1])
                self.logger.debug('Received request from Redis server: %s', msg)
                return_id = msg['id']
                payload = msg['payload']
                return lambda p: self._return_rpc(return_id, p), payload
            else:
                return None, None
        except redis.exceptions.TimeoutError as e:
            return None, None

class RPCClient(MaybeLogger):
    """
    Exposes a handle for each public method of registered at the corresponding service.

    The handle is a callable that takes the same arguments as the
    corresponding method and returns the result of calling that method on the
    underlying component.
    """
    
    def __init__(self, client, methods, logger=None):
        MaybeLogger.__init__(self, logger)
        self._client = client
        self._methods = methods
        self._set_handlers(methods)
    
    class _Handler(MaybeLogger):
        """
        Handle the request by calling the corresponding method on the client.
        """
        def __init__(self, method, client, logger=None):
            MaybeLogger.__init__(self, logger)
            self._method = method
            self._client = client
        
        def __call__(self, *args, **kwargs):
            
            # Call the method on the client
            payload = {
                'method': self._method,
                'arguments': {
                    'args': args,
                    'kwargs': kwargs
                }
            }
            # Send the payload to the client and block until a response is received.
            result = self._client.rpc(payload)
            return result
    
    def __call__(self, *args, **kwargs):
        """
        Call the method on the client.
        """
        if '__CALL__' not in self._methods:
            raise AttributeError(f'\'RPCClient\' has no __call__ method registered.')
        return getattr(self, '__CALL__')(*args, **kwargs)
    
    def _set_handlers(self, methods):
        """
        Sets callable attributes for each method in the methods dictionary.
        """
        for method in methods:
            # Create a callable attribute for each method
            setattr(self, method, self._Handler(method, self._client))

class RPCService(MaybeLogger):
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
    
    def __init__(self, 
                 conn_config:dict,
                 instance_factory:callable,
                 instance_cls:type=None,
                 args:tuple=(),
                 worker_polling_interval:float=1.0,
                 logger=None):
        MaybeLogger.__init__(self, logger)
        self._conn_config = conn_config
        self._worker_polling_interval = worker_polling_interval
        self._instance_factory = instance_factory
        self._instance_args = args
        self._registered_methods = self._discover_methods(instance_cls)
        self._service = self._make_service(conn_config)
        self._should_run = self._registered_methods.pop('run', None) is not None
        
    def _make_service(self, conn_config):
        type_ = conn_config.pop('type', 'redis')
        label = self._instance_factory.__name__ + uuid().hex[:4]
        if type_ == 'redis':
            return _RedisRPCService(conn_config, label, self.logger.getChild('service'))
        else:
            raise ValueError(f'Unknown service type: {type_}')
    
    @property
    def _should_stop(self):
        return self._service.should_close
    
    def _worker(self):
        while not self._should_stop:
            # Extract the return function and payload from the incoming request
            return_func, payload = self._service.next(timeout=self._worker_polling_interval)
            if payload is None:
                self.logger.debug('No payload received, continuing to poll')
                continue
            method = payload.pop('method')
            arguments = payload.pop('arguments')
            assert isinstance(arguments, dict), 'arugments should be a dict'
            args = arguments['args']
            kwargs = arguments['kwargs']
            # call the method
            result = self._registered_methods[method](*args, **kwargs)
            # Call the return function with the payload
            return_func(result)
    
    def run(self):
        """
        Run the service. This method is called by the main thread when the
        service is started.
        """
        
        instance = self._instance_factory(*self._instance_args)
        if self._should_run:
            # span a worker thread to run the instance
            self._runner = threading.Thread(target=instance.run)
            self._runner.daemon = True
            self._runner.start()
        # start the service
        self._service.connect()
        # start the worker thread
        self._worker_thread = threading.Thread(target=self._worker)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        self.logger.info('Service %s started', self._service._label)
    
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
        self.logger.info('Service %s stopped', self._service._label)
    
    def create_handle(self):
        """
        Create a handle for the service.
        """
        return RPCClient(self._service.get_client(), list(self._registered_methods.keys()))
    
    def _discover_methods(self, instance_cls):
        methods = {}
        for method_name in dir(instance_cls):
            method = getattr(instance_cls, method_name)
            if callable(method) and not method_name.startswith('_'):
                methods[method_name] = method
                self.logger.info('Exposing method: %s', method_name)
            elif method_name == '__call__':
                # also expose the __call__ method, if any
                methods['__CALL__'] = method
                self.logger.info('Exposing method: %s', method_name)
            elif method_name == '__CALL__' and callable(method):
                raise ValueError('__CALL__ is a reserved method name.')
        return methods

class ReverbService(MaybeLogger):
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
        MaybeLogger.__init__(self)
        # credits to dm-launchpad for the implementation, too bad it is not maintained anymore
        self._priority_tables_fn = priority_tables_fn
        self._checkpoint_ctor = checkpoint_ctor
        self._checkpoint_time_delta_minutes = checkpoint_time_delta_minutes
        self._port = portpicker.pick_unused_port()
    
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
            port=self._port,
            checkpointer=checkpointer,
        )
    
    def stop(self):
        """
        Stop the reverb server and clean up the resources used by the service.
        """
        self._server.stop()
    
    def create_handle(self):
        return reverb.Client(f'localhost:{self._port}')

class Program(object):
    """
    Program class that manages the services and returns their handles.
    """
    def __init__(self):
        self._services = []
        self._service_processes = []
        self._current_group = None
        self._group_members = 0
    
    def add_service(self, service: RPCService, label: str = ""):
        """
        Add a service to the program.
        """
        new_label = f'{self._current_group}/{self._group_members}'
        self._group_members += 1
        if label is not None:
            new_label += '.' + label
        service.set_logger(base_logger.getChild(f'{new_label}'))
        self._services.append((new_label, service))
        base_logger.info('Service %s added', new_label)
        return service.create_handle()
    
    def launch(self):
        """
        Run all the services in the program in a separate process.
        """
        for name, service in self._services:
            base_logger.info('Starting Service %s', name)
            proc = multiprocessing.Process(target=service.run)
            proc.start()
            self._service_processes.append((name, proc))
    
    def stop(self):
        """
        Stop all the services in the program.
        """
        for name, service in reversed(self._service_processes):
            try:
                # send a SIGINT signal to the process
                service.terminate()
                service.join(timeout=5)
            except Exception as e:
                base_logger.error('Failed to stop service %s %s', name, e)
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
