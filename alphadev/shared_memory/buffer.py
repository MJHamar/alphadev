from .base import (
    ArrayElement, AtomicCounterElement, NestedArrayElement,
    BlockLayout, BaseMemoryManager)
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
from typing import Union, Callable, Dict, List, Optional, Generator
import numpy as np
import contextlib
from time import time, sleep
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BufferInputFull(Exception): pass
class BufferOutputFull(Exception): pass

class BufferHeader(BlockLayout):
    _required_attributes = ['in_index_w', 'in_index_f', 'in_index_r', 'out_index_w', 'out_index_r']
    _elements = {
        # MPSC input buffer with frontiers with the invariant
        # read_head <= write_frontier <= write head and
        # 0 <= write_head - read_head < num_blocks.
        'in_index_w': AtomicCounterElement(),  # write head of the MPSC.
        'in_index_f': AtomicCounterElement(),  # frontier pointer for the MPSC input buffer.
        'in_index_r': ArrayElement(dtype=np.int64, shape=()),  # read head of the MPSC input buffer.
        # SPSC output buffer with the invariant 
        # 0 <= write_head - read_head < num_blocks.
        'out_index_w': ArrayElement(dtype=np.int64, shape=()),  # write head for SPSC output buffer
        'out_index_r': ArrayElement(dtype=np.int64, shape=()),  # read head for the SPSC output buffer.
    }
    @classmethod
    def define(cls):
        pass

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
        self.name = name
        
        self._header_cls = BufferHeader
        self._header_size = self._header_cls.get_block_size()
        self._header_name = f'{name}_header'

        self._input_size = input_element.get_block_size()
        self._input_name = f'{name}_input'
        
        self._output_size = output_element.get_block_size()
        self._output_name = f'{name}_output'
        
        self._input_element_cls = input_element
        self._output_element_cls = output_element
        # round the number of blocks to the nearest power of two.
        self._num_blocks = 1 << (num_blocks - 1).bit_length()  # round up to the next power of two
        if num_blocks != self._num_blocks:
            logger.warning("IOBuffer: num_blocks rounded up to nearest power of two: %s --> %s", num_blocks, self._num_blocks)
        self._mask = self._num_blocks - 1  # for fast modulo operation
        
        self._is_main = False
        self._is_attached = False
    
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        self._is_main = True; self._is_attached = True
        self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=True, size=self._header_size)
        self._input_shm = mp_shm.SharedMemory(name=self._input_name, create=True, size=self._input_size*self._num_blocks)
        self._output_shm = mp_shm.SharedMemory(name=self._output_name, create=True, size=self._output_size*self._num_blocks)
        
        self.header = self._header_cls(self._header_shm, 0, length=self._num_blocks)
        
        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        # clear the header and input/output blocks
        self._header_cls.clear_block(self._header_shm, 0, length=self._num_blocks)
        self.input_index_r = 0
        for i in range(self._num_blocks):
            self._input_element_cls.clear_block(self._input_shm, i * self._input_size)
            self._output_element_cls.clear_block(self._output_shm, i * self._output_size)
    
    def attach(self):
        self._is_main = False; 
        while not self._is_attached:
            try:
                self._header_shm = mp_shm.SharedMemory(name=self._header_name, create=False, size=self._header_size)
                self._input_shm = mp_shm.SharedMemory(name=self._input_name, create=False, size=self._input_size)
                self._output_shm = mp_shm.SharedMemory(name=self._output_name, create=False, size=self._output_size)
                self.header = self._header_cls(self._header_shm, 0, length=self._num_blocks)
                self._is_attached = True
            except FileNotFoundError:
                logger.debug("IOBuffer: waiting for shared memory blocks to be created...")
                sleep(0.1)
    
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
    
    # The input buffer is implemented as a circular buffer.
    # following the Multi-producer, single-consumer (MPSC) pattern.
    # both read and write heads are incremented indefinitely, with the 
    # invariance that 0 <= write_head - read_head < num_blocks.
    
    def _next_in_w(self, increment: Optional[int] = 1):
        """Reserve `increment` blocks for writing to the input buffer."""
        logger.debug("IOBuffer: _next_in_w() called, increment=%d", increment)
        # (busy) wait until the write head has enough space to write the requested number of blocks.
        write_head_ptr = self.header.in_index_w.peek()
        # ensure the distance between the write head and the read frontier doesn't exceed the buffer size.
        while write_head_ptr + increment - self.header.in_index_r >= self._num_blocks:
            sleep(0.0001) # just a little, so reading can catch up.
        logger.debug("IOBuffer: _next_in_w() w %d, rf at %d r at %d", write_head_ptr, self.header.in_index_f.peek(), self.header.in_index_r)
        with self.header.in_index_w() as in_index_w:
            # increment the in_index atomically to let the other
            # producers know that we are reserving space.
            index = in_index_w.fetch_add(increment)  
        # now, we are guaranteed to have enough space to write stuff.
        indices = np.arange(index, index + increment) & self._mask  # use bitwise AND to wrap around the circular buffer.
        logger.debug("IOBuffer: _next_in_w() returning indices %s", indices)
        return indices
    def _update_in_w_frontier(self, increment: Optional[int] = 1):
        """Update the write frontier, so that the read head can catch up."""
        logger.debug("IOBuffer: _update_in_w_frontier() called, increment=%d", increment)
        with self.header.in_index_f() as in_index_f:
            in_index_f.add(increment)
    def _get_in_read_batch(self, increment: Optional[int] = 1):
        """
        Get the next <= increment blocks that are ready to be read from the input buffer,
        but don't update the read head.
        """
        # cap the increment to the number of blocks beetween the read head and the write frontier.
        # logger.debug("IOBuffer: _get_in_read_batch() called, increment=%d; in_r %d, in_f %d, in_w %s", increment, self.header.in_index_r, self.header.in_index_f.peek(), self.header.in_index_w.peek())
        increment = min(increment, self.header.in_index_f.peek() - self.header.in_index_r)
        indices = np.arange(self.header.in_index_r, self.header.in_index_r + increment, dtype=np.int32) & self._mask
        # logger.debug("IOBuffer: _get_in_read_batch() returning indices %s", indices)
        return indices
    def _set_in_read(self, increment: Optional[int] = 1):
        """Update the read head of the input buffer to let the producers know we are done reading."""
        # logger.debug("IOBuffer: _set_in_read() called, increment=%d", increment)
        self.header.in_index_r += increment
    def _get_out_write_batch(self, increment: Optional[int] = 1, strict: bool = True):
        # logger.debug("IOBuffer: _get_out_write_batch() called, increment=%d, strict=%s", increment, strict)
        # (busy) wait until the write head has enough space to write the requested number of blocks.
        while strict and self.header.out_index_w + increment - self.header.out_index_r >= self._num_blocks:
            sleep(0.0001)  # just a little, so reading can catch up.
        if not strict:
            increment = min(increment, self._num_blocks - (self.header.out_index_w - self.header.out_index_r))
        indices = np.arange(self.header.out_index_w, self.header.out_index_w + increment) & self._mask
        return indices
    def _set_out_written(self, increment: Optional[int] = 1):
        """Update the write head of the output buffer to let the reader know we are done writing."""
        # logger.debug("IOBuffer: _set_out_written() called, increment=%d", increment)
        self.header.out_index_w += increment
    def _get_out_read_batch(self, increment: Optional[int] = 1):
        # logger.debug("IOBuffer: _get_out_read_batch() called, increment=%d", increment)
        # cap the increment to the number of blocks between the read head and the write head.
        while self.header.out_index_w - self.header.out_index_r < increment:
            # logger.debug("IOBuffer: _get_out_read_batch() waiting for output blocks to be written...")
            sleep(0.0001)
        increment = min(increment, self.header.out_index_w - self.header.out_index_r)
        indices = np.arange(self.header.out_index_r, self.header.out_index_r + increment) & self._mask
        logger.debug("IOBuffer: _get_out_read_batch() returning indices %s", indices)
        return indices
    
    def submit(self, **payload):
        # logger.debug("IOBuffer: submit() called, payload=%s", payload)
        if not self._is_attached:
            self.attach()
        # get the next available index for writing to the input buffer.
        index = self._next_in_w()[0]
        # logger.debug("IOBuffer: submit() got index %d for writing", index)
        # write the payload to the input block at the given index.
        iblock = self._input_element_cls(self._input_shm, index * self._input_size)
        iblock.write(**payload)
        # update the write frontier to let the readers know that we have submitted a new block.
        self._update_in_w_frontier()
        # logger.debug("IOBuffer: submit(), set submitted at index=%d. offset %s", index, iblock.node_offset)
    
    def ready(self, payload_list: List[Dict[str, np.ndarray]]):
        # logger.debug("IOBuffer: ready() called, payload_list length=%d", len(payload_list))
        if not self._is_attached:
            self.attach()
        base = 0
        while len(payload_list) > 0:
            indices = self._get_out_write_batch(len(payload_list), strict=False) + base
            for index in indices:
                oblock = self._output_element_cls(self._output_shm, index * self._output_size)
                oblock.write(**payload_list.pop(0))
            self._set_out_written(len(indices))
            base += len(indices)
    
    @contextlib.contextmanager
    def read_submited(self, max_samples:int, localize:bool=True) -> Generator[List[Union[BlockLayout, Dict[str, np.ndarray]]], None, None]:
        """
        Read at most `max_samples` submitted blocks from the input buffer.
        This is a generator that yields the blocks as a batch.
        If `localize` is True, the blocks are localized to the current process before yielding.
        Designed for a single consumer process!
        If using with localize=False, make sure to read the blocks before exiting the context,
        otherwise the blocks will be marked as read and will not be available for reading again.
        """
        # logger.debug("IOBuffer: read_submited() max_samples=%d, localize=%s", max_samples, localize)
        if not self._is_attached:
            self.attach()
        indices = self._get_in_read_batch(max_samples)
        
        ret = []
        for index in indices:
            iblock = self._input_element_cls(self._input_shm, index * self._input_size)
            if localize:
                iblock = iblock.read()
                self._set_in_read(1)  # mark the block as read
            ret.append(iblock)
        if len(ret) > 0:
            logger.debug("IOBuffer: read_submited() returning %s", ret)
        yield ret
        if not localize: self._set_in_read(len(indices))
    
    @contextlib.contextmanager
    def read_ready(self, max_samples:int=1, localize:bool=True):
        """
        Read at most `max_samples` ready blocks from the output buffer.
        This is a generator that yields the blocks as a batch.
        If `localize` is True, the blocks are localized to the current process before yielding.
        Designed for a single consumer process!
        If `block` is True, the method will block until at least one block is ready.
        If using with localize=False, make sure to read the blocks before exiting the context,
        otherwise the blocks will be marked as read and will not be available for reading again.
        """
        # logger.debug("IOBuffer: read_ready() called, max_samples=%d, localize=%s", max_samples, localize)
        if not self._is_attached:
            self.attach()
        indices = self._get_out_read_batch(max_samples)
        
        ret = []
        for index in indices:
            oblock = self._output_element_cls(self._output_shm, index * self._output_size)
            if localize:
                oblock = oblock.read()
                self._set_out_written(1)
            ret.append(oblock)
        logger.debug("IOBuffer: read_ready() returning %s", ret)
        yield ret
        if not localize: self._set_out_written(len(indices))
    
    def is_idle(self):
        return (self.header.in_index_w.peek() == self.header.in_index_r and
                self.header.out_index_w == self.header.out_index_r)
