from typing import NamedTuple, Dict, Union, List
from collections import namedtuple
from time import time
import numpy as np
import multiprocessing as mp
import contextlib
import functools
import tree
import atomics

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ArrayElement(NamedTuple):
    dtype: np.dtype
    shape: tuple
    def size(self, *args, **kwargs):
        return np.int32(np.dtype(self.dtype).itemsize * np.prod(self.shape))
    def create(self, shm, offset, *args, **kwargs):
        return np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf, offset=offset)

AtomicContext = functools.partial
class AtomicCounterElement(NamedTuple):
    def size(self, *args, **kwargs):
        return np.dtype(np.int32).itemsize
    def create(self, shm, offset, *args, **kwargs):
        return functools.partial(
            atomics.atomicview,
            shm.buf[offset:offset + self.size(*args, **kwargs)], atype=atomics.INT
            )

class VarLenArrayElement(ArrayElement):
    def size(self, length, *args, **kwargs):
        return np.dtype(self.dtype).itemsize * np.prod((length, *self.shape))
    def create(self, shm, offset, length, *args, **kwargs):
        return np.ndarray((length, *self.shape), dtype=self.dtype, buffer=shm.buf, offset=offset)

class NestedArrayElement(NamedTuple):
    dtype: np.dtype
    shape: tuple
    model: Union[Dict[str, np.ndarray], NamedTuple]
    def size(self, *args, **kwargs):
        if isinstance(self.model, dict):
            return sum(element.dtype.itemsize*np.prod(element.shape) for element in self.model.values())
        elif isinstance(self.model, NamedTuple):
            return sum(getattr(self.model, name).dtype.itemsize * np.prod(getattr(self.model, name).shape) for name in self.model._fields)
        else:
            raise ValueError("self.model must be a dict or a NamedTuple.")
    def create(self, shm, offset, *args, **kwargs):
        elements = {}
        crnt_offset = offset
        if isinstance(self.model, dict):
            for name, element in self.model.items():
                elements[name] = np.ndarray(element.shape, dtype=element.dtype, buffer=shm.buf, offset=crnt_offset)
                crnt_offset += np.int32(element.dtype.itemsize * np.prod(element.shape))
        elif isinstance(self.model, NamedTuple):
            for name in self.model._fields:
                element = getattr(self.model, name)
                if not isinstance(element, np.ndarray):
                    continue
                elements[name] = np.ndarray(element.shape, dtype=element.dtype, buffer=shm.buf, offset=crnt_offset)
                crnt_offset += np.int32(element.dtype.itemsize * np.prod(element.shape))
        if isinstance(self.model, dict):
            return elements
        elif isinstance(self.model, NamedTuple):
            return self.model._make(**elements)

class BlockLayout:
    _elements: Dict[str, ArrayElement] = {}
    
    def __init__(self, shm, offset, *args, **kwargs):
        self.shm = shm
        self.offset = offset
        self._create_elements(*args, **kwargs)
    
    def _create_elements(self, *args, **kwargs):
        crnt_offset = self.offset
        for name, element_spec in self.__class__._elements.items():
            setattr(self, name, element_spec.create(self.shm, crnt_offset, *args, **kwargs))
            crnt_offset += element_spec.size(*args, **kwargs)
    
    def write(self, **kwargs):
        """
        Write the given values to the block.
        The values should be provided as keyword arguments, where the keys are the names of the elements.
        """
        def write_element(element, value):
            element[...] = value
        for name, value in kwargs.items():
            if hasattr(self, name):
                element = getattr(self, name)
                if isinstance(element, np.ndarray):
                    element[...] = value
                elif isinstance(element, (dict, NamedTuple)):
                    tree.map_structure(write_element, element, value)
                else:
                    raise ValueError(f"Element {name} is not a numpy array.")
            else:
                raise ValueError(f"Element {name} does not exist in the block.")
    
    def read(self):
        """Create a localized namedtuple with the contents of the block."""
        return namedtuple(self.__class__.__name__, self.__class__._elements.keys())(
            **{name: getattr(self, name).copy()
               for name in self.__class__._elements.keys()
               if type(getattr(self, name)) != AtomicContext}  # skip atomic counters for localization
        )
    
    @classmethod
    def clear_block(cls, shm, offset, *args, **kwargs):
        """
        Initialize a block of shared memory at the given offset.
        """
        off = offset
        for element_spec in cls._elements.values():
            element = element_spec.create(shm, off, *args, **kwargs)
            if type(element) == AtomicContext:
                # handle atomic counters
                with element() as atomic_counter:
                    atomic_counter.store(0)
            else:
                tree.map_structure(lambda x: (x.fill(0) if hasattr(x,'fill') else x), element)
            off += element_spec.size(*args, **kwargs)

    @classmethod
    def get_block_size(cls, *args, **kwargs):
        """
        Get the size of the block in bytes.
        """
        return int(sum(element.size(*args, **kwargs) for element in cls._elements.values()))

class BaseMemoryManager:
    def configure(self):
        """To be called by the parent process to allocate shared memory blocks."""
        raise NotImplementedError("configure() should be implemented in the subclass.")
    def reset(self):
        """Reset the shared memory blocks to their initial state. Should only be called in the parent process."""
        raise NotImplementedError("reset() should be implemented in the subclass.")
    def attach(self):
        """To be called by the child process to attach to the shared memory blocks."""
        raise NotImplementedError("attach() should be implemented in the subclass.")
    def __del__(self):
        """Clean up shared memory blocks (ideally) from the main process."""
        raise NotImplementedError("__del__() should be implemented in the subclass.")

class BufferHeader(BlockLayout):
    _elements = {
        'in_index': AtomicCounterElement(),  # index of the next block to be used
        'out_index': AtomicCounterElement(),  # index of the next block to be used
        'submitted': VarLenArrayElement(np.bool_, ()), # boolean mask of submitted blocks
        'ready': VarLenArrayElement(np.bool_, ()),  # boolean mask of ready blocks
    }

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
        self._header_size = BufferHeader.get_block_size(length=num_blocks)
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
        
        self.header = BufferHeader(self._header_shm, 0, length=self._num_blocks)
        
        self.reset()  # clear the header and input/output blocks
    
    def reset(self):
        assert self._is_main, "reset() should only be called in the main process."
        # clear the header and input/output blocks
        BufferHeader.clear_block(self._header_shm, 0, length=self._num_blocks)
        for i in range(self._num_blocks):
            self._input_element_cls.clear_block(self._input_shm, i * self._input_size)
            self._output_element_cls.clear_block(self._output_shm, i * self._output_size)
    
    def attach(self):
        self._is_main = False
        self._header_shm = mp.shared_memory.SharedMemory(name=self._header_name, create=False, size=self._header_size)
        self._input_shm = mp.shared_memory.SharedMemory(name=self._input_name, create=False, size=self._input_size)
        self._output_shm = mp.shared_memory.SharedMemory(name=self._output_name, create=False, size=self._output_size)
        self.header = BufferHeader(self._header_shm, 0, length=self._num_blocks)
    
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
    
    def _next_in(self):
        with self.header.in_index() as in_index:
            index = in_index.fetch_inc()  # increment the in_index atomically
        index = index % self._num_blocks  # wrap around the index
        assert not self.header.submitted[index], "Circular input buffer full."
        return index
    def _next_out(self):
        with self.header.out_index() as out_index:
            index = out_index.fetch_inc()  # increment the out_index atomically
        index = index % self._num_blocks  # wrap around the index
        assert not self.header.ready[index], "Circular output buffer full."
        return index
    
    def submit(self, **payload):
        index = self._next_in()
        iblock = self._input_element_cls(self._input_shm, index * self._input_size)
        iblock.write(**payload)
        self.header.submitted[index] = True
        logger.debug("IOBuffer: submit() called, set submitted at index=%d. offset %s", index, iblock.node_offset)
    
    def ready(self, payload_list: List[Dict[str, np.ndarray]]):
        for payload in payload_list:
            index = self._next_out()
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
