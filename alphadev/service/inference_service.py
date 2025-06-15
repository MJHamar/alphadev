from typing import Callable, Union, Optional, NamedTuple, Dict, Any
from time import sleep
import numpy as np
import sonnet as snn
import tensorflow as tf
import tree

from acme.tf import variable_utils as tf2_variable_utils
from acme.tf import utils as tf2_utils
from ..shared_memory.base import BlockLayout, ArrayElement, NestedArrayElement
from ..shared_memory.buffer import IOBuffer
from .variable_service import VariableService
from ..network import NetworkFactory, make_input_spec
from ..device_config import apply_device_config
from .service import Service

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InferenceTaskBase(BlockLayout):
    _required_attributes = ['node_offset', 'observation']
    _elements = {
        'node_offset': ArrayElement(np.int32, ()),  # offset of the node in the shared memory, needed for expansion/backpropagation
    }
    @classmethod
    def define(cls, input_model: Union[dict, NamedTuple]):
        class InferenceTask(cls):
            _elements = cls._elements.copy()
            _elements['observation'] = NestedArrayElement(
                dtype=np.float32, shape=(), model=input_model)
        return InferenceTask

    def __repr__(self):
        return f"InferenceTask(node_offset={self.node_offset}, observation={self.observation})"

class InferenceResultBase(BlockLayout):
    _required_attributes = ['node_offset', 'prior', 'value']
    _elements = {
        'node_offset': ArrayElement(np.int32, ()),  # offset of the node in the shared memory, needed for expansion/backpropagation
    }
    @classmethod
    def define(cls, num_actions):
        class InferenceResult(cls):
            _elements = cls._elements.copy()
            _elements.update({
                'prior': ArrayElement(np.float32, (num_actions,)),  # prior probabilities of actions
                'value': ArrayElement(np.float32, ()), # **SCALAR** value estimate of the node
            })
        return InferenceResult

    def __repr__(self):
        return f"InferenceResult(node_offset={self.node_offset}, prior={self.prior}, value={self.value})"

class AlphaDevInferenceClient(IOBuffer):
    """
    Client for the inference service.
    Contains information on how to connect to the inference_service's shared memory.
    Provides methods to submit inference tasks and read results.
    """
    def __init__(self,
        num_blocks: int,
        input_cls: InferenceTaskBase,
        output_cls: InferenceResultBase,
        name: int,
    ):
        super().__init__(
            num_blocks=num_blocks,
            input_element=input_cls,
            output_element=output_cls,
            name=name
        )

class AlphaDevInferenceService(Service, IOBuffer):
    def __init__(self,
            num_blocks:int,
            network_factory: Callable[[], snn.Module],
            input_spec:dict,
            output_spec:int, # TODO: num actions for now. in the future, we want this to be a description of the network output.
            batch_size:int = 1,
            variable_service: Optional[VariableService] = None,
            variable_update_period:int = 100,
            factory_args:tuple = (),
            factory_kwargs:dict = {},
            name:str = 'AlphaDevInferenceService'
            ):
        super().__init__(logger=None)
        # define the input and output elements
        self._input_spec = input_spec
        self.input_element = InferenceTaskBase.define(input_spec)
        self.output_element = InferenceResultBase.define(output_spec)
        IOBuffer.__init__(self,
            num_blocks=num_blocks,
            input_element=self.input_element,
            output_element=self.output_element,
            name=name
        )
        self.batch_size = batch_size
        self._network_factory = network_factory # to be initialized in the 
        self._factory_args = factory_args or ()
        self._factory_kwargs = factory_kwargs or {}
        self._variable_service = variable_service
        self._variable_update_period = variable_update_period
        # set stopping flag
        self._stop_requested = False
        # stuff to allocate when run is called
        self._network = None  # to be initialized in run()
        self._configured = False  # shared memory, to be initialized in run()
    
    # not overriding configure() or reset().
    # not overriding __del__() or attach() either
    
    def _create_network(self):
        network = self._network_factory(*self._factory_args, **self._factory_kwargs)
        self.logger.debug(f"AlphaDevInferenceService: created network {network}, initializing variables. with input_spec={self._input_spec}")
        tf2_utils.create_variables(network, input_spec=[self._input_spec])
        
        if self._variable_service is None:
            variable_client = None
        else:
            variable_client = tf2_variable_utils.VariableClient(
                client=self._variable_service,
                variables={'network': network.trainable_variables},
                update_period=self._variable_update_period,
            )
        return tf.function(network), variable_client
    
    def run(self, *args, **kwargs):
        """Run the inference service.
        Take batches of inference tasks, process them with the network and write the results back to the shared memory.
        Note that  IOBuffer onlly uses a lock for writing, reading is lock free. consequently,
        running the inference service on multiple processes can be quite redundant but not incorrect.
        """
        if not self._configured:
            self.configure()
            self._configured = True
        
        if self._network is None:
            self._network, self._variable_client = self._create_network()
            self.logger.info(f"AlphaDevInferenceService: created network {self._network}")
        
        def stack_requests(*requests):
            return tf.stack(requests, axis=0)
        
        self.logger.info(f"AlphaDevInferenceService: starting with batch size {self.batch_size}, network={self._network}, variable_client={self._variable_client}")
        while not self._stop_requested:
            poll_start = tf.timestamp()
            with self.read_submited(self.batch_size, localize=False) as tasks:
                if len(tasks) == 0: continue
                self.logger.debug(f"AlphaDevInferenceService: processing {len(tasks)} tasks")
                task_process_start = tf.timestamp()
                node_offsets = []
                inputs = []
                for task in tasks:
                    node_offsets.append(task.node_offset)
                    inputs.append(task.observation)
                inputs = tree.map_structure(stack_requests, *inputs)
                # release the context as soon as the tensors are created.
            # update the variables
            if self._variable_client is not None:
                self._variable_client.update(wait=False)
            inference_start = tf.timestamp()
            prior, *values = self._network(inputs)
            ready_start = tf.timestamp()
            self.logger.debug(f"AlphaDevInferenceService: prior type={type(prior)} values {values}")
            value = values[0] # ugly hack but there are versions of the network where >2 values are returned.
            self.logger.debug(f"obtained shapes from network: offsets={node_offsets}, prior={prior.shape}, value={value.shape}")
            self.ready([dict(
                node_offset=off,
                prior=p,
                value=v
            ) for off, p, v in zip(node_offsets, prior, value)])
            ready_end = tf.timestamp()
            self.logger.debug('inference total: %s; polling: %s, stacking: %s; inference: %s; ready: %s', ready_end - poll_start, task_process_start - poll_start, inference_start - task_process_start, ready_start - inference_start, ready_end - ready_start)
        self.logger.info("AlphaDevInferenceService: stopping run loop.")

    def stop(self):
        """Stop the inference service."""
        self.logger.info("AlphaDevInferenceService: stopping.")
        self._stop_requested = True

    def create_handle(self) -> 'AlphaDevInferenceClient':
        """
        Create a client for the inference service.
        The client can be used to submit inference tasks and read results.
        """
        return AlphaDevInferenceClient(
            num_blocks=self._num_blocks,
            input_cls=self.input_element,
            output_cls=self.output_element,
            name=self.name
        )

class InferenceNetworkFactory:
    """Callable which creates a network, its variables, and connects to a variable service.
    Calling this method will return a callable which updates the variables in the network and 
    runs inference on the network.
    """
    def __init__(self, network_factory: NetworkFactory, observation_spec, variable_service: VariableService, variable_update_period: int = 100):
        self._network_factory = network_factory
        self._observation_spec = observation_spec 
        self._variable_service = variable_service
        self._variable_update_period = variable_update_period
    
    def __call__(self, *args, **kwargs) -> Callable:
        """Create a network and return a callable which updates the variables and runs inference."""
        network = self._network_factory(make_input_spec(self._observation_spec))
        tf2_utils.create_variables(network, [self._observation_spec])
        if self._variable_service is None:
            variable_client = None
        else:
            variable_client = tf2_variable_utils.VariableClient(
                client=self._variable_service,
                variables={'network': network.trainable_variables},
                update_period=100,
            )
        compiled_network = tf.function(network)
        
        def inference(observation):
            if variable_client is not None:
                # update the variables in the network
                variable_client.update(wait=False)
            # ensure batch dimension
            observation = tree.map_structure(lambda o: tf.expand_dims(o, axis=0), observation)
            outputs = compiled_network(observation)
            # outputs is assumed to be a tuple of (prior, value, [other outputs])
            return ( # return prior and value
                tf2_utils.squeeze_batch_dim(outputs[0]),
                tf2_utils.squeeze_batch_dim(outputs[1]))
        
        return inference
