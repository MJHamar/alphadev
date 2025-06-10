from .base import (
    ArrayElement, AtomicCounterElement, NestedArrayElement,
    BlockLayout, BaseMemoryManager)
import multiprocessing as mp
from typing import Union, Callable, Dict, List, Optional
import numpy as np
import contextlib
from time import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BufferHeaderBase(BlockLayout):
    _required_attributes = ['in_index', 'out_index', 'submitted', 'ready']
    _elements = {
        'in_index': AtomicCounterElement(),  # index of the next block to be used
        'out_index': AtomicCounterElement(),  # index of the next block to be used
    }
    @classmethod
    def define(cls, length: int):
        class BufferHeader(cls):
            _elements = cls._elements.copy()
            _elements.update({
                'submitted': ArrayElement(np.bool_, (length,)), # boolean mask of submitted blocks
                'ready': ArrayElement(np.bool_, (length,)),  # boolean mask of ready blocks
            })
        return BufferHeader

class IOBuffer(BaseMemoryManager):
    """The IOBuffer handles asynchronous communication between processes.
    It creates two blocks of shared memory one  for input and one for output.
    The input block is used to enqueue tasks and the output block is where the results are written.
    The buffer is lock-free and uses a circular buffer approach.
    
    1. A process can request a new block for input by calling `next()`.
        the process will receive a BlockLayout object to write its task data into.
    2. The process writes its task data into the input block and calls `submit()`.
    3. Consumer process reads the input block and processes the task. it also clears the submitted flag for that block.
    4. Once the task is processed, the consumer writes the result into the output block and calls the ready() function.
    5. Other processes can then poll() the ready output blocks and read the results.
    """
    def __init__(
        self, num_blocks, input_element: BlockLayout,
        output_element: BlockLayout, name: str = 'IOBuffer'):
        self._header_cls = BufferHeaderBase.define(length=num_blocks)
        self._header_size = self._header_cls.get_block_size(length=num_blocks)
        self._header_name = f'{name}_header'

        self._input_size = input_element.get_block_size()
        self._input_name = f'{name}_input'
        
        self._output_size = output_element.get_block_size()
        self._output_name = f'{name}_output'
        
        self._input_element_cls = input_element
        self._output_element_cls = output_element
        self._num_blocks = num_blocks
        
        self._is_main = False
        # maintain LOCAL read pointers for the two buffers.
        # ideally, only one process should read without localization, otherwise
        # ther will be race conditions. without localization they are also more severe.
        self.submitted_read_head = self.ready_read_head = 0
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True
        self._header_shm = mp.shared_memory.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._input_shm = mp.shared_memory.SharedMemory(name=self._input_name, create=True, size=self._input_size*self._num_blocks)
        self._output_shm = mp.shared_memory.SharedMemory(name=self._output_name, create=True, size=self._output_size*self._num_blocks)
        
        self.header = self._header_cls(self._header_shm, 0, length=self._num_blocks)
        
        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        # clear the header and input/output blocks
        self._header_cls.clear_block(self._header_shm, 0, length=self._num_blocks)
        for i in range(self._num_blocks):
            self._input_element_cls.clear_block(self._input_shm, i * self._input_size)
            self._output_element_cls.clear_block(self._output_shm, i * self._output_size)
    
    def attach(self):
        self._is_main = False
        self._header_shm = mp.shared_memory.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._input_shm = mp.shared_memory.SharedMemory(name=self._input_name, create=False, size=self._input_size)
        self._output_shm = mp.shared_memory.SharedMemory(name=self._output_name, create=False, size=self._output_size)
        self.header = self._header_cls(self._header_shm, 0, length=self._num_blocks)
    
    def __del__(self):
        """Clean up shared memory blocks in the main process."""
        del self.header
        self._header_shm.close()
        self._input_shm.close()
        self._output_shm.close()
        if self._is_main:
            self._header_shm.unlink()
            self._input_shm.unlink()
            self._output_shm.unlink()
    
    def _next_in(self, num_blocks: Optional[int] = 1):
        with self.header.in_index() as in_index:
            index = in_index.fetch_add(num_blocks)  # increment the in_index atomically
        indices = np.arange(index, index + num_blocks) % self._num_blocks
        assert not self.header.submitted[indices].any(), "Circular input buffer full."
        return indices
    def _next_out(self, num_blocks: Optional[int] = 1):
        with self.header.out_index() as out_index:
            index = out_index.fetch_add(num_blocks)  # increment the out_index atomically
        indices = np.arange(index, index + num_blocks) % self._num_blocks
        assert not self.header.ready[indices], "Circular output buffer full."
        return indices
    
    def submit(self, **payload):
        index = self._next_in()[0]
        iblock = self._input_element_cls(self._input_shm, index * self._input_size)
        iblock.write(**payload)
        self.header.submitted[index] = True
        logger.debug("IOBuffer: submit() called, set submitted at index=%d. offset %s", index, iblock.node_offset)
    
    def ready(self, payload_list: List[Dict[str, np.ndarray]]):
        indices = self._next_out(len(payload_list))
        for index, payload in zip(indices, payload_list):
            oblock = self._output_element_cls(self._output_shm, index * self._output_size)
            oblock.write(**payload)
            self.header.ready[index] = True
            logger.debug("IOBuffer: ready() called, set ready at index=%d. offset %s", index, oblock.node_offset)
    
    @contextlib.contextmanager
    def poll_submitted(self, max_samples:int, localize:bool=True):
        """
        Linear search once through the submitted blocks; picking up where the last read left off.
        Appends the first `max_samples` blocks that are ready to be processed.
        """
        # logger.debug("IOBuffer: poll_submitted() called with max_samples=%d, localize=%s", max_samples, localize)
        inp = []; indices = []
        
        start_mod = (self.submitted_read_head % self._num_blocks)
        start_mod = start_mod - 1 if start_mod != 0 else self._num_blocks-1  # avoid zero mod for circular buffer
        while (self.submitted_read_head % self._num_blocks) != start_mod and len(inp) < max_samples:
            if self.header.submitted[self.submitted_read_head]:
                # read the output block and localize its contents. 
                iblock = self._input_element_cls(self._input_shm, self.submitted_read_head * self._input_size)
                if localize:
                    inp.append(iblock.read())
                else:
                    inp.append(iblock)
                indices.append(self.submitted_read_head)  # keep the index for later use
            # increment the index
            self.submitted_read_head = (self.submitted_read_head + 1) % self._num_blocks
        if len(inp) > 0:
            logger.debug("IOBuffer: poll_submitted() inputs %s", [str(i)[:20] + '...' for i in inp])
        yield inp
        # clear the submitted flag after the context is exited
        self.header.submitted[indices] = False

    @contextlib.contextmanager
    def poll_ready(self, localize:bool=True, timeout=None):
        # logger.debug("IOBuffer: poll_ready() called with localize=%s, timeout=%s", localize, timeout)
        start = time()
        while timeout is None or time() - start < timeout:
            if self.header.ready[self.ready_read_head]:
                logger.debug("IOBuffer: poll_ready() found ready block at index %d.", self.ready_read_head)
                break
            self.ready_read_head = (self.ready_read_head + 1) % self._num_blocks
        else:
            yield None # timeout reached.
            return
        # read the output block and localize its contents.
        oblock = self._output_element_cls(self._output_shm, self.ready_read_head * self._output_size)
        if localize:
            oblock = oblock.read()
        yield oblock
        # clear the rady flag only after the context is exited
        self.header.ready[self.ready_read_head] = False

    def is_idle(self):
        header = self.header
        return not header.submitted.any() and not header.ready.any()
