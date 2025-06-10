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
from ..network import NetworkFactory
from ..device_config import apply_device_config

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

class AlphaDevInferenceService(IOBuffer):
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
        # define the input and output elements
        self._input_spec = input_spec
        self.input_element = InferenceTaskBase.define(input_spec)
        self.output_element = InferenceResultBase.define(output_spec)
        super().__init__(
            num_blocks=num_blocks,
            input_element=self.input_element,
            output_element=self.output_element,
            name=name)
        self.batch_size = max(batch_size, 1)
        self._network_factory = network_factory # to be initialized in the 
        self._factory_args = factory_args or ()
        self._factory_kwargs = factory_kwargs or {}
        self._variable_service = variable_service
        self._variable_update_period = variable_update_period
    
    # not overriding configure() or reset().
    # not overriding __del__() or attach() either
    
    def _create_network(self):
        network = self._network_factory(*self._factory_args, **self._factory_kwargs)
        logger.debug(f"AlphaDevInferenceService: created network {network}, initializing variables. with input_spec={self._input_spec}")
        tf2_utils.create_variables(network, input_spec=[self._input_spec])
        
        if self._variable_service is None:
            variable_client = None
        else:
            variable_client = tf2_variable_utils.VariableClient(
                client=self._variable_service,
                variables={'network': network.trainable_variables},
                update_period=self._variable_update_period,
            )
        return network, variable_client
    
    def run(self):
        """Run the inference service.
        Take batches of inference tasks, process them with the network and write the results back to the shared memory.
        Note that  IOBuffer onlly uses a lock for writing, reading is lock free. consequently,
        running the inference service on multiple processes can be quite redundant but not incorrect.
        """
        if not hasattr(self, '_network'):
            self._network, self._variable_client = self._create_network()
            print(f"AlphaDevInferenceService: created network {self._network}")
        def stack_requests(*requests):
            return tf.stack(requests, axis=0)
        num_timeouts = 0
        while True:
            with self.poll_submitted(self.batch_size, localize=False) as tasks:
                if len(tasks) == 0:
                    num_timeouts += 1
                    if num_timeouts > 100000:
                        # print(f"AlphaDevInferenceService: No tasks submitted for the last {num_timeouts} iterations, going to sleep for a bit.")
                        num_timeouts = 0
                        sleep(0.001) # TODO: adjust this to a config parameter.
                    continue
                logger.debug(f"AlphaDevInferenceService: processing {len(tasks)} tasks")
                node_offset = [t.node_offset for t in tasks] # list of node offsets to update
                inputs = [t.observation for t in tasks] # list of observations to evaluate
                inputs = tree.map_structure(stack_requests, *inputs)
                # release the context as soon as the tensors are created.
            # update the variables
            if self._variable_client is not None:
                self._variable_client.update(wait=False)
            prior, *values = self._network(inputs)
            logger.debug(f"AlphaDevInferenceService: prior type={type(prior)} values{values}")
            value = values[0] # ugly hack but there are versions of the network where >2 values are returned.
            logger.debug(f"obtained shapes from network: offsets={node_offset}, prior={prior.shape}, value={value.shape}")
            self.ready([dict(
                node_offset=off,
                prior=p,
                value=v
            ) for off, p, v in zip(node_offset, prior, value)])

class InferenceFactory:
    """
    Factory pattern for creating AlphaDevInferenceService instances.
    In a single distributed run, there may be several inference services (one for each actor pool)
    
    Running the inference service can be done in a separate process,
    using the `run_inference` function as the insertion point.
    """
    def __init__(self,
        num_blocks:int,
        input_spec: Union[dict, NamedTuple],
        output_spec: Union[dict, NamedTuple],
        batch_size:int,
        network_factory: NetworkFactory,
        variable_update_period: int = 100,
        network_factory_args: tuple = (),
        network_factory_kwargs: Dict[str, Any] = {},
        name: str = 'AlphaDevInferenceService'
        ):
        self._num_blocks = num_blocks
        self._input_spec = input_spec
        self._output_spec = output_spec
        self._batch_size = batch_size
        self._network_factory = network_factory
        self._variable_service = None  # to be set later
        self._variable_update_period = variable_update_period
        self._network_factory_args = network_factory_args
        self._network_factory_kwargs = network_factory_kwargs
        self._name = name
    
    def set_variable_service(self, variable_service: VariableService):
        """
        Set the variable service for the inference factory.
        This is used to update the variables in the inference service.
        """
        self._variable_service = variable_service
        return self
    
    def __call__(
        self, variable_service: Optional[VariableService] = None, name:str = None) -> AlphaDevInferenceService:
        """
        Create an instance of the inference service.
        """
        return AlphaDevInferenceService(
            num_blocks=self._num_blocks,
            input_spec=self._input_spec,
            output_spec=self._output_spec,
            batch_size=self._batch_size,
            network_factory=self._network_factory,
            variable_service=variable_service or self._variable_service,
            variable_update_period=self._variable_update_period,
            factory_args=self._network_factory_args,
            factory_kwargs=self._network_factory_kwargs,
            name=name or self._name
        )

def run_inference(
    inference_factory: InferenceFactory,
    device_config: Optional[Dict] = None
    ):
    """
    Insertion point for the inference service.
    Instantiates the network and runs the inference service.
    To be called from a subprocess
    """
    if device_config is not None:
        logger.debug("APV_MCTS[inference process] Applying device configuration")
        apply_device_config(device_config)
    # initialize the inference service
    logger.debug("APV_MCTS[inference process] Initializing inference service.")
    inference_buffer = inference_factory()
    inference_buffer.attach()
    # run indefinitely
    inference_buffer.run()
