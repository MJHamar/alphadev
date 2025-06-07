from typing import Callable
from time import sleep
import numpy as np
import sonnet as snn
import tensorflow as tf

from .config import ADConfig
from .environment import AssemblyGame
from .shared_memory import *


class InferenceTask(BlockLayout):
    _elements = {
        'node_offset': ArrayElement(np.int32, ()),  # offset of the node in the shared memory, needed for expansion/backpropagation
        'observation': NestedArrayElement(dtype=np.float32, shape=(), model=AssemblyGame(ADConfig.task_spec).observation_spec()),
    }

class InferenceResult(BlockLayout):
    _elements = {
        'node_offset': ArrayElement(np.int32, ()),  # offset of the node in the shared memory, needed for expansion/backpropagation
        'prior': ArrayElement(np.float32, (ADConfig.task_spec.num_actions,)),  # prior probabilities of actions
        'value': ArrayElement(np.float32, ()),  # **SCALAR** value estimate of the node
    }

class AlphaDevInferenceService(IOBuffer):
    def __init__(self, 
            num_blocks:int,
            network_factory: Callable[[], snn.Module],
            batch_size:int = 1,
            factory_args:tuple = (),
            factory_kwargs:dict = {},
            name:str = 'AlphaDevInferenceService'
            ):
        super().__init__(
            num_blocks=num_blocks,
            input_element=InferenceTask,
            output_element=InferenceResult,
            name=name)
        self.batch_size = max(batch_size, 1)
        self._network_factory = network_factory # to be initialized in the 
        self._factory_args = factory_args or ()
        self._factory_kwargs = factory_kwargs or {}
    
    # not overriding configure() or reset().
    
    def _create_network(self):
        return self._network_factory(*self._factory_args, **self._factory_kwargs)

    def attach(self):
        super().attach() # the shared memory.
        # also instantiate the network.
        self._network = self._create_network()
    
    # not overriding __del__() either
    
    def run(self):
        """Run the inference service.
        Take batches of inference tasks, process them with the network and write the results back to the shared memory.
        Note that  IOBuffer onlly uses a lock for writing, reading is lock free. consequently,
        running the inference service on multiple processes can be quite redundant but not incorrect.
        """
        def stack_requests(*requests):
            return tf.stack(requests, axis=0)
        num_timeouts = 0
        while True:
            with self.poll_submitted(self.batch_size, localize=False) as tasks:
                if len(tasks) == 0:
                    num_timeouts += 1
                    if num_timeouts > 10:
                        print(f"AlphaDevInferenceService: No tasks submitted for the last {num_timeouts} iterations, going to sleep for a bit.")
                        num_timeouts = 0
                        sleep(5.0) # TODO: adjust this to a config parameter.
                    continue
                node_offset = [t.node_offset for t in tasks] # list of node offsets to update
                inputs = [t.observation for t in tasks] # list of observations to evaluate
                inputs = tree.map_structure(stack_requests, *inputs)
                # release the context as soon as the tensors are created.
            prior, *values = self._network(inputs)
            print(f"AlphaDevInferenceService: prior type={type(prior)} values{values}")
            value = values[0] # ugly hack but there are versions of the network where >2 values are returned.
            print(f"obtained shapes from network: offsets={node_offset}, prior={prior.shape}, value={value.shape}")
            self.ready([dict(
                node_offset=off,
                prior=p,
                value=v
            ) for off, p, v in zip(node_offset, prior, value)])
