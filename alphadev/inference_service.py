"""
Inference server for AlphaDev. Intended to be run as a separate process.

- Load the config
- 
- Initialize the network
- Start listening for requests
- Do forever:
    - Accumulate requests for a given time period
    - Pull model weights from the variable storage
    - Run batched inference on the network
    - Send the results back to the clients
"""
import os
import time
import tree
import tensorflow as tf
from acme.tf import variable_utils as tf2_variable_utils
import threading

from .config import AlphaDevConfig
from .network import AlphaDevNetwork
from .service import RPCClient, _RedisRPCService, _RedisRPCClient
from .environment import environment_spec_from_config
from .variable_service import make_variable_client
from .service import make_client_backend, make_service_backend

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class InferenceService:
    def __init__(self, config: AlphaDevConfig):
        self.config = config
        self.network = AlphaDevNetwork(config)
        # create variables for the network
        tf2_variable_utils.create_variables(self.network, environment_spec_from_config(config))
        # create a client for the variable storage
        self.variable_client = make_variable_client(config)
        # update the network weights
        self.variable_client.update()
        # make a listener service to listen for requests
        self._listener_service = make_service_backend(
            conn_config=self.config.connection_config,
            label=self.config.inference_service_name,
            logger=logger.getChild('listener'),
        )
        # initialise state.
        self.requests = []
        self.request_callbacks = []
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.batch_size = config.batch_size
        self.accumulation_period = config.inference_accumulation_period
    
    def _batch_requests(self):
        """Batch the requests for inference."""
        if len(self.requests) == 0:
            return None
        batch_requests = []
        batch_callbacks = []
        with self._lock: # lock the requests queue
            for i in range(self.batch_size):
                if len(self.requests) == 0:
                    break
                callback, request = self.requests.pop(0)
                batch_requests.append(request)
                batch_callbacks.append(callback)
        # stack the requests
        if len(batch_requests) == 0:
            return None, None
        batch_requests = tree.map_structure(tf.stack, batch_requests)
        return batch_requests, batch_callbacks
    
    def _process_requests(self, batch: tf.Tensor, batch_callbacks: list):
        """Run inference on the network."""
        # Run the network on the batch
        logits, value = self.network(*batch['args'], **batch['kwargs'])
        # Send the results back to the clients
        for idx, callback in enumerate(batch_callbacks):
            callback(logits[idx], value[idx])
    
    def _update_weights(self):
        """Pull the model weights from the variable storage."""
        if self._variable_storage.has_variables():
            self.variable_client.update()
    
    def _listener(self):
        """Listen for requests from the clients."""
        while True:
            callback, payload = self._listener_service.next(timeout=self.accumulation_period)
            if callback is not None:
                with self._lock:
                    self.requests.append((callback, payload))
                    self.request_callbacks.append(callback)
            else:
                # If no callback is received, we can assume that the request is not valid
                # and we can ignore it.
                continue
    
    @property
    def _should_stop(self):
        return self._listener_service.should_close
    
    def run(self):
        """Run the inference server."""
        # Start the listener in a separate thread
        listener_thread = threading.Thread(target=self._listener)
        listener_thread.start()
        
        while not self._should_stop:
            # sleep while the listener is gathering data
            time.sleep(self.accumulation_period)
            # get the first batch
            batch_requests, batch_callbacks = self._batch_requests()
            while batch_requests is not None:
                # Pull the model weights from the variable storage
                self._update_weights()
                # Run inference on the batch
                self._process_requests(batch_requests, batch_callbacks)
                # Get the next batch of requests
                batch_requests, batch_callbacks = self._batch_requests()

class InferenceClient():
    def __init__(self, config: AlphaDevConfig):
        self._config = config
        self._service_name = config.inference_service_name
        self._client = make_client_backend(self._config.connection_config, f'{self._service_name}_queue', logger.getChild('client'))
    
    def __call__(self, *args, **kwargs):
        """Call the inference server with the given arguments.
        Block until the result is available.
        """
        # Call the inference server with the given arguments
        return self._client.rpc({'args': args, 'kwargs': kwargs}, timeout=20)
