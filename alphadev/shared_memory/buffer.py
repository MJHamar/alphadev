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
logger.setLevel(logging.INFO)

class BufferInputFull(Exception): pass
class BufferOutputFull(Exception): pass

class BufferHeader(BlockLayout):
    _required_attributes = ['in_index_w', 'in_index_f', 'in_index_r', 'out_index_w', 'out_index_r']
    _elements = {
        # MPSC input buffer with frontiers with the invariant
        # read_head <= write_frontier <= write head and
        # 0 <= write_head - read_head < num_blocks.
        'in_index_w': AtomicCounterElement(),  # write head of the MPSC.
        'in_index_f': ArrayElement(dtype=np.uint64, shape=()),  # frontier pointer for the MPSC input buffer.
        'in_index_r': ArrayElement(dtype=np.uint64, shape=()),  # read head of the MPSC input buffer.
        # SPSC output buffer with the invariant 
        # 0 <= write_head - read_head < num_blocks.
        'out_index_w': ArrayElement(dtype=np.uint64, shape=()),  # write head for SPSC output buffer
        'out_index_f': ArrayElement(dtype=np.uint64, shape=()),  # frontier pointer for the SPSC output buffer.
        'out_index_r': ArrayElement(dtype=np.uint64, shape=()),  # read head for the SPSC output buffer.
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
        self._mask = np.array(self._num_blocks - 1, dtype=np.uint64)  # for fast modulo operation
        
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
        if hasattr(self, 'header'):
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
    # since reading and writing to the input buffer is not atomic,
    # we maintain a write frontier to ensure that the read head only reads well-defined blocks.
    # Producer processes can pre-allocate space in the input buffer and receive a list of indices 
    # that are guaranteed to be available for writing.
    # Once producers are done writing they increment the write frontier, while maintaining the order of indices.
    # i.e. reader can increment the frontier only if its smallest index is equals current fontier + 1.
    # The consumer process reads the buffer, maintaining the invariant that the read head is always less than or equal to the write frontier.
    
    def _next_in_w(self, increment: Optional[int] = 1):
        """Reserve `increment` blocks for writing to the input buffer."""
        # logger.debug("IOBuffer: _next_in_w() called, increment=%d", increment)
        # (busy) wait until the write head has enough space to write the requested number of blocks.
        write_head_ptr = self.header.in_index_w.peek()
        # ensure the distance between the write head and the read frontier doesn't exceed the buffer size.
        while write_head_ptr + increment - self.header.in_index_r >= self._num_blocks:
            sleep(0.0001) # just a little, so reading can catch up.
        logger.debug("IOBuffer: _next_in_w() w %d, rf at %d r at %d", write_head_ptr, self.header.in_index_f, self.header.in_index_r)
        with self.header.in_index_w() as in_index_w:
            # increment the in_index atomically to let the other
            # producers know that we are reserving space.
            index = in_index_w.fetch_add(increment)  
        # now, we are guaranteed to have enough space to write stuff.
        indices = np.arange(index, index + increment, dtype=np.uint64) & self._mask  # use bitwise AND to wrap around the circular buffer.
        logger.debug("IOBuffer: _next_in_w() returning indices %s dtype %s", indices, indices.dtype)
        return indices
    def _update_in_w_frontier(self, start_mod:int, increment:int):
        """Update the write frontier, so that the read head can catch up.
        start_mod is the modulo of of the first index in producer's write batch.
        This method will wait until the write frontier's modulus is equal to start_mod - 1.
        """
        logger.debug("IOBuffer: _update_in_w_frontier() called, start_mod %d, increment %d", start_mod, increment)
        frontier_ptr = self.header.in_index_f
        # busy wait until the write frontier is at the expected value.
        target = np.uint64(start_mod)
        logger.debug("IOBuffer: _update_in_w_frontier() waiting for frontier %d to be %d", frontier_ptr & self._mask, target & self._mask)
        while frontier_ptr & self._mask != target & self._mask:
            sleep(0.0001)  # just a little, so other producers can write their stuff.
        # now we can update the write frontier. no need to be atomic here.
        frontier_ptr += np.uint64(increment)
        logger.debug("IOBuffer: _update_in_w_frontier() f %d; w %d; r %d", frontier_ptr, self.header.in_index_w.peek(), self.header.in_index_r)
    def _get_in_read_batch(self, increment: Optional[int] = 1):
        """
        Get the next <= increment blocks that are ready to be read from the input buffer,
        but don't update the read head.
        """
        # cap the increment to the number of blocks beetween the read head and the write frontier.
        # logger.debug("IOBuffer: _get_in_read_batch() called, increment=%d; in_r %d, in_f %d, in_w %s", increment, self.header.in_index_r, self.header.in_index_f, self.header.in_index_w.peek())
        increment = min(increment, self.header.in_index_f - self.header.in_index_r)
        indices = np.arange(self.header.in_index_r, self.header.in_index_r + increment, dtype=np.uint64) & self._mask
        # logger.debug("IOBuffer: _get_in_read_batch() returning indices %s", indices)
        return indices
    def _set_in_read(self, increment: Optional[int] = 1):
        """Update the read head of the input buffer to let the producers know we are done reading."""
        logger.debug("IOBuffer: _set_in_read() called, increment=%d", increment)
        self.header.in_index_r += np.uint64(increment)
    
    # The output buffer is implemented as a single-producer, single-consumer (SPSC) buffer.
    # the producer (same process as the input consumer) 
    # pre-allocates space in the output buffer by incrementing the write head.
    # (so that the consumers know the producer is busy)
    # and increments the write frontier after writing the output.
    # the reader reads the output buffer and increments the read head after reading the output.
    # the invariant is the same, the read head can never exceed the write frontier.
    def _inc_out_w(self, increment: Optional[int] = 1):
        """Increment the write head regardless of the available space."""
        self.header.out_index_w += np.uint64(increment)
    def _next_out_w(self, increment: Optional[int] = 1, strict: bool = True):
        """
        Get the next <= `increment` blocks that are ready to be written to the output buffer.
        That is, take the next `increment` indices from between the write head and the write frontier.
        Since we assume that the output producer is the same process as the input consumer,
        we are guaranteed to have enough distance between the write head and the frontier.
        The write head, however, might be violating the circular buffer invariant
        so we can only take at most min(increment, num_blocks - (frontier - read_head)) indices.
        """
        # logger.debug("IOBuffer: _get_out_write_batch() called, increment=%d, strict=%s", increment, strict)
        # (busy) wait until the write head has enough space to write the requested number of blocks.
        while strict and self._num_blocks - (self.header.out_index_f - self.header.out_index_r) < increment:
            # logger.debug("IOBuffer: _get_out_write_batch() waiting for output blocks to be written...")
            sleep(0.0001)
        if not strict:
            increment = min(increment, self._num_blocks - (self.header.out_index_f - self.header.out_index_r))
        
        # return indices for writing to the output buffer.
        indices = np.arange(self.header.out_index_f, self.header.out_index_f + increment, dtype=np.uint64) & self._mask
        return indices
    def _set_out_written(self, increment: Optional[int] = 1):
        """Update the write frontier of the output buffer to signal that we are done."""
        # logger.debug("IOBuffer: _set_out_written() called, increment=%d", increment)
        self.header.out_index_f += np.uint64(increment)
        assert self.header.out_index_f <= self.header.out_index_w, \
            f"IOBuffer: out_index_f {self.header.out_index_f} > out_index_w {self.header.out_index_w}. someone read less than what they are writing"
    def _get_out_read_batch(self, increment: Optional[int] = 1):
        # logger.debug("IOBuffer: _get_out_read_batch() called, increment=%d", increment)
        # cap the increment to the number of blocks between the read head and the write head.
        while self.header.out_index_f - self.header.out_index_r < increment:
            # logger.debug("IOBuffer: _get_out_read_batch() waiting for output blocks to be written...")
            sleep(0.0001)
        logger.debug("IOBuffer: _get_out_read_batch() w %d, f %d r %d", self.header.out_index_w, self.header.out_index_f, self.header.out_index_r)
        # return the indices for reading from the output buffer.
        indices = np.arange(self.header.out_index_r, self.header.out_index_r + increment, dtype=np.uint64) & self._mask
        # logger.debug("IOBuffer: _get_out_read_batch() returning indices %s", indices)
        return indices
    def _set_out_read(self, increment: Optional[int] = 1):
        """Update the read head of the output buffer to let the producer know we are done reading."""
        # logger.debug("IOBuffer: _set_out_read() called, increment=%d", increment)
        self.header.out_index_r += np.uint64(increment)
    
    def submit(self, **payload):
        logger.debug("IOBuffer: submit() called, payload offset %s", payload['node_offset'])
        # get the next available index for writing to the input buffer.
        index = self._next_in_w()[0]
        # logger.debug("IOBuffer: submit() got index %d for writing", index)
        # write the payload to the input block at the given index.
        iblock = self._input_element_cls(self._input_shm, np.uint64(index * self._input_size))
        iblock.write(**payload)
        # update the write frontier to let the readers know that we have submitted a new block.
        self._update_in_w_frontier(start_mod=index, increment=1)
        # logger.debug("IOBuffer: submit(), set submitted at index=%d. offset %s", index, iblock.node_offset)
    
    def ready(self, payload_list: List[Dict[str, np.ndarray]]):
        # logger.debug("IOBuffer: ready() called, payload_list length=%d", len(payload_list))
        while len(payload_list) > 0:
            indices = self._next_out_w(len(payload_list), strict=False)
            for index in indices:
                oblock = self._output_element_cls(self._output_shm, np.uint64(index * self._output_size))
                oblock.write(**payload_list.pop(0))
            self._set_out_written(indices.shape[0])
    
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
        indices = self._get_in_read_batch(max_samples)
        
        # logger.debug("IOBuffer: read_submited() max_samples=%d, localize=%s, indices=%s", max_samples, localize, indices)
        if len(indices) == 0:
            yield []
            return
        # also reserve some output blocks for writing the results.
        # we don't care about buffer overflow; reading will take time and 
        # we won't get indices here either.
        self._inc_out_w(len(indices))
        ret = []
        for index in indices:
            iblock = self._input_element_cls(self._input_shm, np.uint64(index * self._input_size))
            if localize:
                iblock = iblock.read()
                self._set_in_read(1)  # mark the block as read
            ret.append(iblock)
        if len(ret) > 0:
            logger.debug("IOBuffer: read_submited() returning with %s inputs", len(ret))
        yield ret
        if not localize: self._set_in_read(indices.shape[0])
    
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
        indices = self._get_out_read_batch(max_samples)
        logger.debug("IOBuffer: read_ready() output indices to read: %s", indices)
        
        ret = []
        for index in indices:
            oblock = self._output_element_cls(self._output_shm, np.uint64(index * self._output_size))
            if localize:
                oblock = oblock.read()
                self._set_out_read(1)
            ret.append(oblock)
        logger.debug("IOBuffer: read_ready() returning with %d outputs", len(ret))
        yield ret
        if not localize: self._set_out_read(indices.shape[0])
    
    def is_idle(self):
        return (self.header.in_index_w.peek() == self.header.in_index_r and
                self.header.out_index_w == self.header.out_index_r)
