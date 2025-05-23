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
import sys
import pickle
import time
import tree
import tensorflow as tf
import ml_collections
from acme.tf import variable_utils as tf2_variable_utils
from acme.tf import utils as tf2_utils
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
        self.network = AlphaDevNetwork(config.hparams, config.task_spec, name='inference_network')
        # create variables for the network
        tf2_utils.create_variables(self.network, [environment_spec_from_config(config).observations])
        # create a client for the variable storage
        self._variable_storage = make_variable_client(config)
        self.variable_client = tf2_variable_utils.VariableClient(
            client=self._variable_storage,
            variables={'network': self.network.trainable_variables},
            update_period=config.variable_update_period)
        # update the network weights
        self.variable_client.update()
        # make a listener service to listen for requests
        self._listener_service = make_service_backend(
            conn_config=self.config.distributed_backend_config,
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
            return None, None
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
        
        def stack_requests(*requests):
            return tf.stack(requests, axis=0)
        
        # Transpose the list of dicts to a dict of lists, then stack
        batched_dict = tree.map_structure(stack_requests, *batch_requests)

        # Check shapes
        return batched_dict, batch_callbacks
    
    def _process_requests(self, batch: tf.Tensor, batch_callbacks: list):
        """Run inference on the network."""
        # Run the network on the batch
        tensors = self.network(*batch['args'], **batch['kwargs'])
        logger.debug("InferenceService: network output %s", [t.shape for t in tensors])
        # Send the results back to the clients
        for idx, callback in enumerate(batch_callbacks):
            logger.debug('InferenceService: Sending results to client %s', callback)
            callback([t[idx] for t in tensors])
    
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
        logger.info("Starting inference server at %s (server queue %s)", self.config.inference_service_name, self._listener_service._server_queue)
        while not self._should_stop:
            # sleep while the listener is gathering data
            time.sleep(self.accumulation_period)
            # get the first batch
            batch_requests, batch_callbacks = self._batch_requests()
            while batch_requests is not None:
                logger.debug("Processing batch of %d requests", len(batch_requests))
                # Pull the model weights from the variable storage
                # self._update_weights()
                # Run inference on the batch
                self._process_requests(batch_requests, batch_callbacks)
                # Get the next batch of requests
                batch_requests, batch_callbacks = self._batch_requests()

class InferenceClient():
    def __init__(self, config: AlphaDevConfig):
        self._config = config
        self._service_name = config.inference_service_name
        self._client = make_client_backend(
            self._config.distributed_backend_config, f'{self._service_name}_queue', logger.getChild('client'))
    
    def __call__(self, *args, **kwargs):
        """Call the inference server with the given arguments.
        Block until the result is available.
        """
        # Call the inference server with the given arguments
        payload = {'args': args, 'kwargs': kwargs}
        # convert tensors to numpy arrays
        payload = tf2_utils.to_numpy(payload)
        logger.debug("InferenceClient: Sending request to %s (client queue %s)", self._service_name, self._client._server_queue)
        return self._client.rpc(payload, timeout=20)

def main():
    """Run the inference server as a standalone process."""
    config_path = sys.argv[1]
    with open(config_path, 'rb') as f:
        config: AlphaDevConfig = pickle.load(f)
    assert isinstance(config, AlphaDevConfig), "Config must be an instance of AlphaDevConfig"
    service = InferenceService(config)
    logger.info("Starting inference service...")
    service.run()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()