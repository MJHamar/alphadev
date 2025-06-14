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
logger.setLevel(logging.DEBUG)

class ArrayElement(object):
    def __init__(self, dtype: np.dtype, shape: tuple):
        self.dtype = np.dtype(dtype)
        self.shape = shape
        self._size = None
    
    def _get_size(self, *args, **kwargs):
        return np.int32(np.dtype(self.dtype).itemsize * np.prod(self.shape))
    
    def size(self, *args, **kwargs):
        if self._size is None:
            self._size = self._get_size(*args, **kwargs)
        return self._size
    def create(self, shm=None, offset=None, *args, **kwargs):
        # if shm is None, numpy ignores the offset 
        self._shm = shm
        self._offset = offset
        
        buffer = shm.buf if shm is not None else None
        return np.ndarray(self.shape, dtype=self.dtype, buffer=buffer, offset=offset)

class NestedArrayElement(ArrayElement):
    dtype: np.dtype
    shape: tuple
    model: Union[Dict[str, np.ndarray], NamedTuple]
    def __init__(self, dtype, shape, model: Union[Dict[str, np.ndarray], NamedTuple]):
        super().__init__(dtype, shape)
        self.model = model
    
    def _get_size(self, *args, **kwargs):
        if isinstance(self.model, dict):
            return sum(element.dtype.itemsize*np.prod(element.shape) for element in self.model.values())
        elif isinstance(self.model, NamedTuple):
            return sum(getattr(self.model, name).dtype.itemsize * np.prod(getattr(self.model, name).shape) for name in self.model._fields)
        else:
            raise ValueError("self.model must be a dict or a NamedTuple.")
    
    def create(self, shm=None, offset=None, *args, **kwargs):
        elements = {}
        buffer = shm.buf if shm is not None else None
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

# Actomics should only be used in a multiprocessing context.
class AtomicContext:
    """
    Wrapper for atomics.atomicview with a peek() method that avoids having to initialize 
    the atomic view context for read-only operations.
    """
    def __init__(self, buffer, offset, dtype=np.uint64):
        self._buffer = buffer
        self._offset = offset
        self._dtype = dtype
        self._size = np.dtype(self._dtype).itemsize
        assert np.dtype(self._dtype) == np.dtype(np.uint64), "AtomicContext only supports uint64 dtype."
        self._atype = atomics.UINT
    def __call__(self):
        """return an atomicview context manager"""
        return atomics.atomicview(
            self._buffer[self._offset:self._offset+self._size],
            atype=atomics.INT)
    
    def peek(self):
        """
        Peek the value without initializing the atomic view context.
        Saves a bunch of time wasted on system calls.
        """
        return np.frombuffer(
            self._buffer, offset=self._offset, count=1, dtype=self._dtype)[0]
    
    def copy(self):
        """
        Copy the atomic context to a new buffer and offset.
        This is useful for creating a new atomic context with the same value.
        """
        raise NotImplementedError("AtomicContext cannot be copied.")

class AtomicCounterElement(NamedTuple):
    def size(self, *args, **kwargs):
        return np.dtype(np.uint64).itemsize
    def create(self, shm, offset, *args, **kwargs):
        return AtomicContext(
            shm.buf, offset, dtype=np.uint64)  # create an atomic context for the shared memory buffer

class BlockLayout:
    _required_attributes = []  
    _elements: Dict[str, ArrayElement] = {}
    
    @classmethod
    def define(cls):
        """Define the shared memory blocks. Should be called in the parent process."""
        raise NotImplementedError("define() should be implemented in the subclass.")
    
    def __init__(self, shm=None, offset=None, *args, **kwargs):
        self.shm = shm
        self.offset = offset
        self._lazy_elements = {}
        self._create_elements(*args, **kwargs)
        is_fine = True
        for attr in self._required_attributes:
            if not hasattr(self, attr) and not attr in self._lazy_elements:
                logger.error(f"BlockLayout {self.__class__.__name__} is missing required attribute: {attr}")
                is_fine = False
        if not is_fine:
            raise ValueError(f"BlockLayout {self.__class__.__name__} is not properly defined. Missing required attributes: {self._required_attributes}")
    
    def _create_elements(self, *args, **kwargs):
        crnt_offset = self.offset or 0
        for name, element_spec in self.__class__._elements.items():
            # logger.debug(f"Creating element {name} at offset {crnt_offset} with spec {element_spec}")
            # Remove existing attribute/property from the instance or its class (including inherited properties)
            if hasattr(type(self), name):
                for cls in type(self).mro():
                    if name in cls.__dict__:
                        delattr(cls, name)
                        # logger.debug(f"Removed property {name} from class {cls.__name__}")
                        break
            self._lazy_elements[name] = functools.partial(
                element_spec.create, self.shm, crnt_offset, *args, **kwargs)
            # logger.debug(f"Set lazy element {name} with spec {element_spec} at offset {crnt_offset}")
            crnt_offset += element_spec.size(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        Lazy loading of elements. If the element is not found, it will be created.
        This allows for dynamic creation of elements based on the class definition.
        """
        if name in self._lazy_elements:
            element_spec = self._lazy_elements[name]
            # logger.debug(f"Accessing element {name} in {self.__class__.__name__}. Element spec: {element_spec}")
            if isinstance(element_spec, functools.partial):
                # If the element is an ArrayElement, create it with the current shm and offset
                created = element_spec()
                self._lazy_elements[name] = created
                # logger.debug(f"Created element {name}: {created}")
                return created
            return element_spec
        # If we get here, the attribute doesn't exist anywhere
        raise AttributeError(f"{self.__class__.__name__} has no element named '{name}'. Available elements: {list(self._lazy_elements.keys())}")
    
    def write(self, **kwargs):
        """
        Write the given values to the block.
        The values should be provided as keyword arguments, where the keys are the names of the elements.
        """
        def write_element(element, value):
            element[...] = value
        for name, value in kwargs.items():
            if hasattr(self, name):
                element = getattr(self, name)()
                if isinstance(element, np.ndarray):
                    element[...] = value
                elif isinstance(element, (dict, namedtuple)):
                    tree.map_structure(write_element, element, value)
                else:
                    raise ValueError(f"Element {name} is not a numpy array.")
            else:
                raise ValueError(f"Element {name} does not exist in the block.")
    
    def read(self):
        """Create a localized namedtuple with the contents of the block."""
        return namedtuple(self.__class__.__name__, self.__class__._elements.keys())(
            **{name: getattr(self, name)().copy()
               for name in self.__class__._elements.keys()}  # skip atomic counters for localization
        )
    
    @classmethod
    def clear_block(cls, shm=None, offset=None, *args, **kwargs):
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
