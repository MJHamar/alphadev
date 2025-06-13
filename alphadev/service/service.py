"""
Lightweight implementation of Service objects that wrap different components
of the distributed training pipeline.
"""
from typing import Dict, Union
import os
import sys
import abc
import reverb
import redis
import pickle
import threading
import contextlib
import signal
from time import sleep
from uuid import uuid4 as uuid
import subprocess
import cloudpickle
from tempfile import NamedTemporaryFile

import logging
base_logger = logging.getLogger(__name__)
base_logger.setLevel(logging.DEBUG)
no_logger = logging.getLogger('no_logger')
no_logger.setLevel(logging.INFO)

class MaybeLogger:
    def __init__(self, logger=None):
        self._logger = logger or no_logger
    
    @property
    def logger(self):
        return self._logger
    
    def set_logger(self, logger):
        self._logger = logger

class _ClientBackend:
    @abc.abstractmethod
    @contextlib.contextmanager
    def connection(self): pass
    @abc.abstractmethod
    def rpc(self, payload, timeout=5): pass

class _ServiceBackend:
    @abc.abstractmethod
    def configure(self): pass
    @abc.abstractmethod
    @contextlib.contextmanager
    def connection(self): pass
    @abc.abstractmethod
    def set_close(self): pass
    @abc.abstractmethod
    def should_close(self): pass
    @abc.abstractmethod
    def get_client(self): pass
    @abc.abstractmethod
    def _return_rpc(self, key, payload): pass
    @abc.abstractmethod
    def next(self, timeout=5): pass

class _RedisRPCClient(_ClientBackend, MaybeLogger):
    """Redis implementation of the RPC client."""
    def __init__(self, redis_config, server_queue, logger=None):
        self._redis_config = redis_config
        self._server_queue = server_queue
        MaybeLogger.__init__(self, logger)
    
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
    
    def rpc(self, payload, timeout=5):
        with self.connection() as client:
            try:
                # Send the payload to the Redis server
                call_id = uuid().hex
                rpc_call = {
                    'id': call_id,
                    'payload': payload
                }
                rpc_call_bin = pickle.dumps(rpc_call)
                client.rpush(self._server_queue, rpc_call_bin)
                self.logger.debug('Sent RPC call to Redis server: %s', str(rpc_call)[:50])
            except redis.exceptions.ConnectionError as e:
                self.logger.error('Failed to send RPC call to Redis server: %s', e)
                raise RuntimeError('Failed to send RPC call to Redis server') from e
            try:
                # Wait for the response
                self.logger.debug('RedisClient waiting for response at %s', call_id)
                response = client.blpop(call_id, timeout=timeout)
                if response:
                    response = pickle.loads(response[1])
                    self.logger.debug('Received response from Redis server: %s', str(response)[:50])
                    return response
                else:
                    self.logger.error('id: %s No response received from Redis server', call_id)
                    raise RuntimeError('No response received from Redis server')
            except redis.exceptions.TimeoutError as e:
                self.logger.error('id: %s Timeout while waiting for response from Redis server: %s', call_id, e)
                raise RuntimeError('Timeout while waiting for response from Redis server') from e

class _RedisRPCService(_ServiceBackend, MaybeLogger):
    """Redis implementation of the RPC service."""
    def __init__(self, redis_config, label, logger=None):
        MaybeLogger.__init__(self, logger)
        self._redis_config = redis_config
        self._label = label
        self._server_queue = f'{label}_queue'
        self._close_key = f'{label}_close'
        with self.connection() as client:
            client.set(self._close_key, 0)
    
    def configure(self):
        pass # redis should already be configured.
    
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
    
    def set_close(self):
        with self.connection() as client:
            # Set the close key to 1 to signal the service to stop
            client.set(self._close_key, 1)
            self.logger.debug('Set close key for Redis service: %s', self._close_key)
    
    @property
    def should_close(self):
        # Check if the close key is set to 1
        with self.connection() as client:
            return client.get(self._close_key) == 0
    
    def get_client(self):
        return _RedisRPCClient(self._redis_config, self._server_queue)
    
    def _return_rpc(self, key, payload):
        with self.connection() as client:
            # Return the payload to the Redis server
            try:
                self.logger.debug('RedisServer callback writing to %s with %s', key, str(payload)[:50])
                payload_bin = pickle.dumps(payload)
                client.rpush(key, payload_bin)
            except redis.exceptions.ConnectionError as e:
                self.logger.error('Failed to return RPC call to Redis server: %s', e)
                raise RuntimeError('Failed to return RPC call to Redis server') from e
    
    def next(self, timeout=5):
        # wait for the next request from the Redis server
        # or return None if timeout is reached
        with self.connection() as client:
            try:
                msg_bin = client.blpop(self._server_queue, timeout=timeout)
                
                if msg_bin:
                    msg = pickle.loads(msg_bin[1])
                    self.logger.debug('Received request from Redis server: %s', str(msg)[:50])
                    return_id = msg['id']
                    payload = msg['payload']
                    return lambda p: self._return_rpc(return_id, p), payload
                else:
                    return None, None
            except redis.exceptions.TimeoutError as e:
                return None, None

def make_service_backend(conn_config, label, logger) -> _ServiceBackend:
    type_ = conn_config.get('type', 'redis')
    if type_ == 'redis':
        return _RedisRPCService(conn_config, label,
                                logger=logger.getChild('service') if logger is not None else None)
    else:
        raise ValueError(f'Unknown service type: {type_}')

def make_client_backend(conn_config, label, logger=None) -> _ClientBackend:
        type_ = conn_config.get('type', 'redis')
        if type_ == 'redis':
            return _RedisRPCClient(conn_config, label,
                                   logger=logger.getChild('redis') if logger is not None else None)
        else:
            raise ValueError(f"Unknown connection type: {type_}")

class RPCClient(MaybeLogger):
    """
    Exposes a handle for each public method registered at the corresponding service.

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

class Service(MaybeLogger):
    """
    Base class for services that can be run in a separate thread.
    
    This class is used to create a service that can be run in a separate thread
    and can be stopped by calling the stop method.
    """
    def __init__(self, logger=None):
        MaybeLogger.__init__(self, logger)
        # register signal handlers for SIGINT and SIGTERM
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        """
        Signal handler to stop the service.
        """
        self.logger.info('Received signal %s, stopping service', signum)
        if signum in (signal.SIGINT, signal.SIGTERM):
            self.stop()
    
    @abc.abstractmethod
    def run(self, *args, **kwargs): pass
    @abc.abstractmethod
    def stop(self): pass
    @abc.abstractmethod
    def create_handle(self): pass

class RPCService(Service):
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
                 fork_worker:bool=True,
                 worker_polling_interval:float=1.0,
                 logger=None):
        super().__init__(self, logger)
        self._conn_config = conn_config
        self._fork_worker = fork_worker
        self._worker_polling_interval = worker_polling_interval
        self._instance_factory = instance_factory
        self._instance_args = args
        self._registered_methods = self._discover_methods(instance_cls)
        self._service = make_service_backend(
            conn_config=conn_config,
            label=instance_cls.__name__ + '.' + uuid().hex[:4],
            instance_cls=instance_cls,
            logger=logger
        )
        self._should_run = self._registered_methods.pop('run', None) is not None
        assert fork_worker or self._should_run, 'fork_worker should be True if run is not defined'
    
    @property
    def _should_stop(self):
        return self._service.should_close
    
    def _worker(self):
        while not self._should_stop:
            # Extract the return function and payload from the incoming request
            return_func, payload = self._service.next(timeout=self._worker_polling_interval)
            if payload is None:
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
    
    def run(self, *args, **kwargs):
        """
        Run the service. This method is called by the main thread when the
        service is started.
        """
        # TODO: make signal handler to stop the service
        instance = self._instance_factory(*self._instance_args)
        # bind methods to the instance
        for method_name in self._registered_methods.keys():
            self._registered_methods[method_name] = getattr(instance, method_name)
        
        if self._should_run:
            # start the worker thread
            self._worker_thread = threading.Thread(target=self._worker)
            self._worker_thread.daemon = True
            self._worker_thread.start()
            # span a worker thread to run the instance
            self.logger.info('%s running service', self._service._label)
            instance.run()
        else:
            # start the worker in the main thread
            self.logger.info('%s start listening for requests', self._service._label)
            self._worker()
    
    def stop(self):
        """
        Stop the service. This method is called by the main thread when the
        service is stopped.
        """
        self._service.set_close()
        if self._should_run:
            self._runner.join(timeout=5)
        self._worker_thread.join(timeout=5)
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

class ReverbService(Service):
    """
    Reverb service that is given a reverb table constructor and creates a reverb server
    
    When create_handle is called, it returns a reverb client that can be used to 
    interact with the reverb server.
    """
    def __init__(self, priority_tables_fn, port, checkpoint_ctor=None, checkpoint_time_delta_minutes=None):
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
        self._port = port
    
    def run(self, *args, **kwargs):
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
        # keep the server running
        base_logger.info('Reverb server started on port %s', self._port)
        self._server.wait()
        base_logger.info('Reverb server stopped')
    
    def stop(self):
        """
        Stop the reverb server and clean up the resources used by the service.
        """
        self._server.stop()
    
    def create_handle(self):
        return reverb.Client(f'localhost:{self._port}')

def deploy_service(executable: callable, device_config:Dict[str, str] = None, label: str = "", num_instances: int = 1):
    """
    Deploy a service by spawning a new process that runs the executable.
    Done using the subprocess module and the `process_entry.py` script as the entry point.
    
    Args:
        executable: The callable to run in the new process.
        device_config: A dictionary containing the device configuration for the service.
            It should contain 'gpu' and 'memory' keys.
        label: A label for the service, used to identify it in logs.
        num_instances: The number of instances of the service to run.
    Returns:
        A list of handles for the deployed services and corresponding temporary files.
    """
    handles = []
    for i in range(num_instances):
        new_label = f'{label}_{i}' if label else f'service_{i}'
        fl = NamedTemporaryFile(prefix=f'service_{new_label}_', suffix='.pkl', delete=False)
        cloudpickle.dump(executable, fl)
        fl.flush()
        args = ['--label', new_label]
        if device_config is not None:
            args += ['--device_name', device_config['gpu'], '--allocation_size', str(device_config['memory'])]
        base_logger.info('Launching service: %s', new_label)
        proc = subprocess.Popen([
            sys.executable,
            '-m', 'alphadev.service.process_entry',
            fl.name,
            *args
        ])
        handles.append((new_label, proc, fl))
    return handles

class Program(object):
    """
    Program class that manages the services and returns their handles.
    """
    def __init__(self):
        self._services = []
        self._service_processes = []
        self._current_group = None
        self._group_members = 0
    
    def add_service(self, service: Service, label: str = "", device_config:Dict[str, str] = None):
        """
        Add a service to the program.
        """
        new_label = f'{self._current_group}_{self._group_members}'
        self._group_members += 1
        if label is not None:
            new_label += '.' + label
        service.set_logger(base_logger.getChild(f'{new_label}'))
        self._services.append((new_label, service, device_config))
        base_logger.info('Service %s added', new_label)
        return service.create_handle()
    
    def launch(self):
        """
        Run all the services in the program in a separate process.
        """
        
        self._service_processes = []
        for name, service, device_config in self._services:
            base_logger.info('Configuring service %s', name)
            handles = deploy_service(service.run, device_config=device_config, label=name)
            self._service_processes.extend(handles)
            base_logger.info('Service %s launched with PID %s', name, handles[0][1].pid)
        base_logger.info('All services launched.')
    
    def stop(self):
        """
        Stop all the services in the program.
        """
        base_logger.info('Start monitoring services for termination...')
        while True:
            for name, proc, _ in self._service_processes:
                proc: subprocess.Popen = proc
                rc = proc.poll()
                if rc is None:
                    # base_logger.info('Service %s is still running', name)
                    continue
                else:
                    base_logger.info('Service %s has stopped with exit code %s', name, rc)
                    proc.wait()
                    break
            else:
                sleep(1)
                continue
            break
        base_logger.info('A service has stopped, stopping all services')
        for name, proc, _ in self._service_processes:
            proc: subprocess.Popen = proc
            base_logger.info('Stopping service %s', name)
            proc.send_signal(signal.SIGINT)
        while any(proc.poll() is None for _, proc, _ in self._service_processes):
            base_logger.info('Waiting for all services to stop...')
            sleep(1)
        
        # also remove the binaries
        for _, _, fl in self._service_processes:
            fl: NamedTemporaryFile = fl
            if os.path.exists(fl.name):
                base_logger.info('Removing temporary process binary %s', fl.name)
                fl.close()
                fl.unlink()
        
        base_logger.info('All services stopped')
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
