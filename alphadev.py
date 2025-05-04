# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pseudocode description of the AlphaDev algorithm."""

###########################
########## Content ########
# 1. Environment
# 2. Networks
#   2.1 Network helpers
#   2.2 Representation network
#   2.3 Prediction network (correctness and latency values and policy)
# 3. Helpers
# 4. Part 1: Self-Play
# 5. Part 2: Training
###########################

import os
import collections
import functools
import itertools
import json
import math
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Union, Tuple, List, Set, Mapping, Generator

import time
import chex
import haiku as hk
import jax
import jax.lax
import jax.numpy as jnp
import ml_collections
import numpy
import optax

from tinyfive.multi_machine import multi_machine

import logging
logger = logging.getLogger(__name__)

############################
###### 1. Environment ######


class CPUState(NamedTuple):
    registers: jnp.ndarray # num_inputs x num_regs current state of the registers in each cpu
    memory: jnp.ndarray # num_inputs x mem_size current state of the memory in each cpu
    register_mask: jnp.ndarray # num_inputs x num_regs active registers in each cpu
    memory_mask: jnp.ndarray # num_inputs x mem_size active memory in each cpu
    program: jnp.ndarray # current program applied to all inputs
    program_length: jnp.ndarray # current position of the program counter for different inputs
    
    def __hash__(self):
        return hash(self.register_mask.tobytes() + self.memory_mask.tobytes())

X0 = 0
X1 = 1 # reserved for comparison
REG_T = 10
MEM_T = 11

riscv_valid_registers = [X0] + list(range(2, 32)) # x0 is hard-coded zero register

# define the x86 instruction set used in AlphaDev fixed sort programs
# NOTE: we use risc-like move instructions to more easily distinguish between different modes
x86_signatures = {
    "mv" : (REG_T, REG_T), # move <reg1>, <reg2> 
    "lw" : (MEM_T, REG_T), # move <mem>, <reg1>
    "sw" : (REG_T, MEM_T), # move <reg1>, <mem>
    # no load immediates are used in AlphaDev fixed-sort programs
    "cmp" : (REG_T, REG_T),
    "cmovg" : (REG_T, REG_T),
    "cmovle" : (REG_T, REG_T),
    # skip jump instructions, they are not used in the published sort algorithms
    }
x86_opcode2int = {
    k: i for i, k in enumerate(x86_signatures.keys())
}


class Action(object):
    """Action representation."""

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

    def __repr__(self):
        return f"A({self.index})"

class RiscvAction(Action):
    def __init__(self, index: int,
        opcode: str,
        operands: Tuple[int, ...],
                 ):
        super().__init__(index)
        self.opcode = opcode
        self.operands = operands

    def __repr__(self):
        return f"RiscvAction(i={self.index}, opcode={self.opcode}, operands={self.operands}"

    def __str__(self):
        return f"{self.opcode} {', '.join(map(str, self.operands))}"

class x86Action(Action):
    def __init__(self, index, 
        opcode: str,
        operands: Tuple[int, int]):
        super().__init__(index)
        self.opcode = opcode
        self.operands = operands

    def asm(self, state) -> List[RiscvAction]:
        """
        Convert x86 opcode and operands to RISC-V opcode and operands.
        Args:
            state: The current state of the CPU.
        Returns:
            list: A list of RISC-V instructions. Using the conversion described above
        """
        if self.opcode == "mv": # move between registers
            return [RiscvAction(self.index, "ADD", (self.operands[0], X0, self.operands[1]))]
        elif self.opcode == "lw": # load word from memory to register
            return [RiscvAction(self.index, "LW", (self.operands[1], self.operands[0], X0))] # rd,imm,rs -- rd, rs(imm)
        elif self.opcode == "sw": # store word from register to memory
            return [RiscvAction(self.index, "SW", (self.operands[0], self.operands[1], X0))] # rs1,imm,rs2 -- rs1, rs2(imm)
        elif self.opcode == "cmp": # compare two registers
            return [RiscvAction(self.index, "SUB", (X1, self.operands[0], self.operands[1]))]
            # if A > B, then X1 > 0
            # if A < B, then X1 < 0
            # riscv has bge (>=) and blt (<) instructions
        elif self.opcode == "cmovg": # conditional move if greater than
            return [ # A > B <=> B < A -- 0 < X1
                RiscvAction(self.index, "BLT", (0, X1, 4*len(state.program)+8)),  # skip next instruction if A < B
                RiscvAction(self.index, "ADD", (self.operands[1], X0, self.operands[0]))  # copy C to D
            ]
        elif self.opcode == "cmovle": # conditional move if less than or equal
            return [ # A <= B <=> B >= A -- 0 >= X1
                RiscvAction(self.index, "BGE", (0, X1, 4*len(state.program)+8)),  # skip next instruction if A > B
                RiscvAction(self.index, "ADD", (self.operands[1], X0, self.operands[0]))  # copy E to F
            ]
        else:
            raise ValueError(f"Unknown opcode: {self.opcode}")

    def to_numpy(self):
        return jnp.array([x86_opcode2int[self.opcode], *self.operands], dtype=jnp.int32)

def x86_enumerate_actions(max_reg: int, max_mem: int) -> List[Tuple[str, Tuple[int, int]]]:
    def apply_opcode(opcode: str, operands: Tuple[int, int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        # operands is a triple (reg1, reg2, mem)
        signature = x86_signatures[opcode]
        if signature == (REG_T, REG_T):
            return [(opcode, (operands[0], operands[1]))]
        elif signature == (MEM_T, REG_T):
            return [(opcode, (operands[2], operands[0]))]
        elif signature == (REG_T, MEM_T):
            return [(opcode, (operands[0], operands[2]))]
        else:
            assert False, f"No signature of type {signature} should be used. fix this."
    def enum_actions(r1: int, r2: int, m: int) -> Generator[Tuple[str, Tuple[int, int]], None, None]:
        for i in range(r1):
            for j in range(r2):
                for k in range(m):
                    # generate all combinations of registers and memory
                    for opcode in x86_signatures.keys():
                        yield from apply_opcode(opcode, (i, j, k))
    #   logger.debug("Enumerating actions for max_reg=%d, max_mem=%d", max_reg, max_mem)
    actions = set(enum_actions(max_reg, max_reg, max_mem))
    actions = {i: x86Action(i, *action) for i, action in enumerate(actions)}
    #   logger.debug("Enumerated %d actions", len(actions))
    return actions

riscv_opcode_to_int = {
    "ADD": 0,
    "SUB": 1,
    "LW": 2,
    "SW": 3,
    "BLT": 4,
    "BGE": 5,
}
RISCV_NUM_OPERANDS = 3 # TODO: actually 4 but we don't use the last one

# the emulator should call ActionSpace.get(action) to get the action
# the action space can use the emulator to get the action
# the action space also needs to define a masking function that masks invalid actions based on the emulator state
class ActionSpace(object):
    # placeholder for now.
    pass

class x86ActionSpace(ActionSpace):
    def __init__(self,
            actions: Dict[int, Action],
            state: 'CPUState'):
        self._actions = actions
        self.state = state
    
    @property
    def actions(self):
        return jnp.array(list(self._actions.keys()))
    
    def __len__(self):
        return len(self._actions)
    
    def get(self, action_id: int) -> List[RiscvAction]:
        return self._actions[action_id].asm(self.state) # convert to RISC-V

class ActionSpaceStorage(object):
    # placeholder for now.
    pass

class x86ActionSpaceStorage(ActionSpaceStorage):
    def __init__(self, max_reg: int, max_mem: int):
        self.max_reg = max_reg
        self.max_mem = max_mem
        self.actions: Dict[int, Action] = x86_enumerate_actions(max_reg, max_mem)
        # there is a single action space for the given task
        self.action_space_cls = x86ActionSpace # these are still x86 instructions
        # TODO: make sure we don't flood the memory with this
        self.masks = {}
        # for pruning the action space (one read and one write per memory location)
        self._history_cache = None
        self._mems_read = set()
        self._mems_written = set()
        # build mask lookup tables
        self._build_masks()
    
    def _build_masks(self):
        """
        Build masks over the action space for each register and memory location.
        At runtime, we can dynamically take the union of a subset of these masks
        to efficiently mask the action space. 
        
        Each row in a mask is a boolean array over the action space, indicating whether
        the action uses the register or memory location. 
        """
        # we create a max_reg x action_space_size and 
        # max_mem x action_space_size masks
        action_space_size = len(self.actions)
        # table mapping all registers and memory locations to the action space 
        # a cell (i,j) is True if location i is accessed by action j.
        act_loc_table = jnp.zeros(((self.max_reg + self.max_mem), action_space_size), dtype=jnp.bool)
        # mask for register locations
        reg_locs = jnp.zeros((self.max_reg + self.max_mem), dtype=jnp.bool).at[:self.max_reg].set(True)
        # mask for memory locations
        mem_locs = jnp.zeros((self.max_reg + self.max_mem), dtype=jnp.bool).at[self.max_reg:].set(True)
        # boolean mask for actions that only use register locations
        reg_only_actions = jnp.zeros((action_space_size,), dtype=jnp.bool)
        # boolean mask for actions that read from memory locations
        mem_read_actions = jnp.zeros((action_space_size,), dtype=jnp.bool)
        # boolean mask for actions that write to memory locations
        mem_write_actions = jnp.zeros((action_space_size,), dtype=jnp.bool)
        for i, action in enumerate(self.actions.values()):
            # iterate over the x86 instructions currently under consideration
            x86_opcode, x86_operands = action.opcode, action.operands
            signature = x86_signatures[x86_opcode]
            if signature == (REG_T, REG_T):
                act_loc_table    = act_loc_table.at[x86_operands, i].set(True)
                reg_only_actions = reg_only_actions.at[i].set(True)
            else: # action that accesses memory.
                mem_loc, reg_loc = x86_operands if signature == (MEM_T, REG_T) else reversed(x86_operands)
                mem_loc += self.max_reg # offset the memory locations
                act_loc_table    = act_loc_table.at[reg_loc, i].set(True)
                act_loc_table    = act_loc_table.at[mem_loc, i].set(True)
                if x86_opcode.startswith("l"): # load action
                    mem_read_actions = mem_read_actions.at[i].set(True)
                else:
                    mem_write_actions = mem_write_actions.at[i].set(True)
        
        assert (reg_only_actions & mem_read_actions & mem_write_actions == 0).any(), \
            "Action space was not partitioned correctly"
        assert (reg_only_actions | mem_read_actions | mem_write_actions).all(), \
            "Action space was not partitioned correctly"
        
        self.act_loc_table = act_loc_table
        self.reg_locs = reg_locs
        self.mem_locs = mem_locs
        self.reg_only_actions = reg_only_actions
        self.mem_read_actions = mem_read_actions
        self.mem_write_actions = mem_write_actions

    def get_mask(self, state, history:list=None) -> jnp.ndarray:
        """Get the mask over the action space for the given state and history.
        
        Returns a boolean array over the action space, with True values indicating
        valid actions.
        """
        def update_history(action):
            # update the history with the action
            for act in action:
                if act.opcode.startswith("L"): # (MEM_T, _)
                    #   logger.debug("load %s", act.operands[0])
                    self._mems_read.add(act.operands[0])
                elif act.opcode.startswith("S"): # (_, MEM_T)
                    #   logger.debug("store %s", act.operands[1])
                    self._mems_written.add(act.operands[1])
                
        # check if the history seems to be a continuation of the current state
        if history is not None: # update history
            #   logger.debug("pruning: history")
            crnt_space = self.get_space(state) # the action space. almost constant lookup time
            if self._history_cache is not None and\
                len(self._history_cache) + 1 == len(history) and \
                self._history_cache[-1] == history[-2]: # check only the last action to save time
                #   logger.debug("pruning: using cached history")
                # we can use the cached history
                update_history(crnt_space.get(history[-1].index))
            else:
                #   logger.debug("pruning: recomputing history")
                # we need to recompute the history
                self._mems_read = set()
                self._mems_written = set()
                # iterate over the history
                for action in history:
                    update_history(crnt_space.get(action.index))
            # update the history cache
            self._history_cache = history
            # update the masks
            #   logger.debug("mems_read %s", self._mems_read)
            #   logger.debug("mems_written %s", self._mems_written)

        act_loc_table = self.act_loc_table
        reg_locs = self.reg_locs
        mem_locs = self.mem_locs
        reg_only_actions = self.reg_only_actions
        mem_read_actions = self.mem_read_actions
        mem_write_actions = self.mem_write_actions
        
        # get the active registers and memory locations from the CPU state
        # note they are matrices of shape E x R and E x M 
        # (E: number of examples, R: number of registers, M: number of memory locations)
        # we consider the largest window of active registers and memory locations
        # so we take their union
        active_registers = state.register_mask # shape E x R+(unused) # TODO: we might want to let the emulator know
        active_registers = jnp.any(active_registers, axis=0)[:self.max_reg] # shape R
        active_memory = state.memory_mask      # shape E x M
        active_memory = jnp.any(active_memory, axis=0) # shape M

        assert active_registers.shape[0] == self.max_reg, \
            "active registers and max_reg do not match."
        assert active_memory.shape[0] == self.max_mem, \
            "active memory and max_mem do not match."

        #   logger.debug("active registers (shape %s) %s", active_registers.shape, active_registers)
        #   logger.debug("active memory (shape %s) %s", active_memory.shape, active_memory)

        # find windows of locations that are valid
        reg_window = jnp.zeros_like(reg_locs, dtype=jnp.bool)
        last_reg = - jnp.argmax(active_registers[::-1])
        #   logger.debug("last_reg %d", last_reg)
        active_registers = active_registers.at[last_reg].set(True)
        reg_window = reg_window.at[:self.max_reg].set(active_registers)
        # same for the memory locations
        mem_window = jnp.zeros_like(mem_locs, dtype=jnp.bool)
        last_mem = - jnp.argmax(active_memory[::-1])
        #   logger.debug("last_mem %d", last_mem)
        active_memory = active_memory.at[last_mem].set(True)
        mem_window = mem_window.at[self.max_reg:].set(active_memory)
        
        #   logger.debug("register window (shape %s) %s", reg_window.shape, reg_window)
        #   logger.debug("memory window (shape %s) %s", mem_window.shape, mem_window)
        
        # reg_window and mem_window now address the rows
        # of the act_loc_table, which should be considered.
        
        # we select all register-only actions, which operate within the register window
        #   logger.debug("register-only actions shape %s", reg_only_actions.shape)
        #   logger.debug("act_loc_table shape %s", act_loc_table.shape)
        
        assert reg_window.shape[0] == act_loc_table.shape[0], \
            "register window and action location table do not match."
        assert reg_only_actions.shape[0] == act_loc_table.shape[1], \
            "register-only actions and action location table do not match."

        # Identify register-only actions that access *any* location *outside* the active register window.
        # 1. Get access pattern for locations outside the window:
        inactive_loc_access = act_loc_table[~reg_window, :] # Shape (N_inactive_locs, N_actions)
        # 2. Check for each action if it accesses *any* inactive location:
        accesses_inactive_loc = jnp.any(inactive_loc_access, axis=0) # Shape (N_actions,)
        # A register-only action is valid if it is a register-only action
        # AND it does NOT access any inactive location.
        reg_only_mask = reg_only_actions & (~accesses_inactive_loc) # Shape (N_actions,)
        #   logger.debug("register-only mask shape %s", reg_only_mask.shape) # Should be (num_actions,)
        
        # to enfoce that only one read and one write is allowed at each memory location,
        # we also need to look at the history of the program
        # and mask out any actions that are illegal
        read_locs = jnp.array(list(self._mems_read), dtype=jnp.int32) + self.max_reg
        #      logger.debug("read_locs %s", read_locs)
        # create a mask of memory locations that are read
        mem_read_locs = jnp.zeros_like(mem_locs, dtype=jnp.bool).at[read_locs].set(True)
        #      logger.debug("mem_read_locs %s", mem_read_locs)
        # subtract the read locations mask from the memory window
        mem_read_window = mem_window & ~mem_read_locs
        #      logger.debug("mem_read_window %s", mem_read_window)
        # select all memory read actions, which operate within the memory window
        invalid_mem_read_loc = act_loc_table[~(mem_read_window | reg_locs), :]
        #      logger.debug("act_loc invalids %s", invalid_mem_read_loc.any(axis=0))
        accesses_invalid_mem = jnp.any(invalid_mem_read_loc, axis=0)
        #      logger.debug("num invalid mem read actions: %s", jnp.sum(accesses_invalid_mem))
        mem_read_mask = mem_read_actions & (~accesses_invalid_mem)
        #      logger.debug("mem_read_mask num selected %s", mem_read_mask.sum())

        # do the same for write actions
        write_locs = jnp.array(list(self._mems_written), dtype=jnp.int32) + self.max_reg
        mem_write_locs = jnp.zeros_like(mem_locs, dtype=jnp.bool).at[write_locs].set(True)
        mem_write_window = mem_window & ~mem_write_locs
        invalid_mem_write_loc = act_loc_table[~(mem_write_window | reg_locs), :]
        accesses_invalid_mem = jnp.any(invalid_mem_write_loc, axis=0)
        mem_write_mask = mem_write_actions & (~accesses_invalid_mem)
        #      logger.debug("mem_write_mask num selected %s", mem_read_mask.sum())

        assert reg_only_mask.shape[0] == len(self.actions), \
            "mask and action space size do not match."
        assert not (reg_only_mask & mem_read_mask & mem_write_mask).any(), \
            "masks do not partition the action space."
        assert (reg_only_mask | mem_read_mask | mem_write_mask).any(), \
            "no actions left in the action space."

        # combine the masks by taking their union
        return reg_only_mask | mem_read_mask | mem_write_mask

    def get_space(self, state) -> x86ActionSpace:
        return self.action_space_cls(self.actions, state)

class IOExample(NamedTuple):
    inputs: jnp.ndarray # num_inputs x <sequence_length>
    outputs: jnp.ndarray # num_inputs x <sequence_length>

class TaskSpec(NamedTuple):
    max_program_size: int
    num_inputs: int # number of input examples
    num_funcs: int # number of x86 instructions to consider
    num_regs: int # number of registers to consider
    num_mem: int # number of memory locations to consider. num_mem+num_regs = num_locations
    num_locations: int # memory + register locations to consider
    num_actions: int # number of actions in the action space
    correct_reward: float
    correctness_reward_weight: float
    latency_reward_weight: float
    latency_quantile: float

class AssemblyGame(object):
    """The environment AlphaDev is interacting with."""

    class AssemblyInstruction(object):
        def __init__(self, action: RiscvAction):
            """
            Represent an action as an assembly instruction.
            """
            self.opcode = action.opcode # str
            self.operands = action.operands # list

    class AssemblySimulator(object):
        
        def __init__(self, task_spec, inputs:List[IOExample]):
            """Initialize the simulator with the task specification."""
            self.task_spec = task_spec
            self.inputs = inputs
            self.emulator = multi_machine(task_spec.num_mem*4, task_spec.num_inputs)
            self.reset()
        
        def reset(self):
            """Reset the simulator to an initial state."""
            self.emulator.reset_state()
            for i, example in enumerate(self.inputs):
                self.emulator.set_memory(i, example.inputs)

        # pylint: disable-next=unused-argument
        def apply(self, instruction):
            """Apply an assembly instruction to the simulator."""
            self.emulator.append_instruction(instruction.opcode, instruction.operands)
            # reset the state of the simulator
            self.reset()
            # execute the program
            self.emulator.exe()

        @property
        def registers(self): return self.emulator.registers
        @property
        def memory(self): return self.emulator.memory
        @property
        def register_mask(self): return self.emulator.register_mask
        @property
        def memory_mask(self): return self.emulator.memory_mask
        @property
        def program_counter(self): return self.emulator.program_counter

        def measure_latency(self) -> float:
            """Measure the latency of a program."""
            self.emulator.measure_latency()

        def invalid(self) -> bool:
            # FIXME: need to check if the program can be invalid at any point in time.
            return False

    def __init__(self, task_spec, inputs: List[IOExample], action_space_storage: ActionSpaceStorage):
        self.task_spec = task_spec
        self.inputs = inputs
        self.storage = action_space_storage
        self.program = [] # program here is an array, which suggests that there are only append actions
        self.simulator = self.AssemblySimulator(task_spec, inputs)
        self.previous_correct_items = 0
        self.expected_outputs = self.make_expected_outputs()

    def step(self, action:Action):
        action_space = self.storage.get_space(self.state())
        self.program.append(action) # append the action (no swap moves)
        #   logger.debug("step: action index %s", action.index)
        instructions = action_space.get(action.index) # lookup x86 instructions and convert to riscv
        # logger.debug("step: act %s, instruction %s", action, instructions)
        # there might be multiple instructions in a single action
        if not isinstance(instructions, list):
            instructions = [instructions]
        for riscv_action in instructions:
            insn = self.AssemblyInstruction(riscv_action)
            self.simulator.apply(insn)
        return self.observation(), self.correctness_reward()

    def state(self) -> CPUState:
        # convert from numpy to jax.numpy arrays
        state = CPUState(
            registers=jnp.array(self.simulator.registers),
            memory=jnp.array(self.simulator.memory),
            register_mask=jnp.array(self.simulator.register_mask),
            memory_mask=jnp.array(self.simulator.memory_mask),
            program=jnp.array([p.to_numpy() for p in self.program]),
            program_length=jnp.array(self.simulator.program_counter),
        )
        logger.debug("AssemblyGame: state program %s", state.program.shape)
        return state

    def observation(self):
        return self.state()._asdict()

    def make_expected_outputs(self):
        return jnp.stack(
            [example.outputs for example in self.inputs]
        )

    def correctness_reward(self) -> float:
        """Computes a reward based on the correctness of the output."""
        state = self.state()

        # Weighted sum of correctly placed items
        correct_items = 0
        # NOTE: this assumes that the expected outputs are always written from index 0
        for output, expected in zip(state.memory, self.expected_outputs):
            correct_items += sum(
                output[i] == expected[i] for i in range(len(output))
            )
            reward = self.task_spec.correctness_reward_weight * (
                correct_items - self.previous_correct_items
            )
        self.previous_correct_items = correct_items

        # Bonus for fully correct programs
        all_correct = jnp.all(state.memory[:,:self.expected_outputs.shape[1]] == self.expected_outputs)
        reward += self.task_spec.correct_reward * all_correct

        return reward

    def latency_reward(self) -> float:
        latency_samples = [
            # NOTE: measure latency n times
            self.simulator.measure_latency()
            for _ in range(self.task_spec.num_latency_simulation)
        ]
        return (
            numpy.quantile(latency_samples, self.task_spec.latency_quantile)
            * self.task_spec.latency_reward_weight
        )

    def clone(self):
        # reinitialises the program, simulator (which makes a new emulator)
        ret = AssemblyGame(
            task_spec=self.task_spec,
            inputs=self.inputs,
            action_space_storage=self.storage,
        ) 
        # only mutable thing left is the program inside the simulator and in this class
        # also note that since "program" is independent of the machine instance,
        # we can simply shallow copy the array
        del ret.simulator.emulator # make sure we don't allocate too much space
        ret.simulator.emulator = self.simulator.emulator.clone()
        ret.program = self.program.copy()
        return ret

    def terminal(self) -> bool:
        return self.simulator.invalid() or self.correct()

    def correct(self) -> bool:
        state = self.state()
        return jnp.all(state.memory[:, :self.expected_outputs.shape[1]] == self.expected_outputs)

######## End Environment ########
#################################

#####################################
############ 2. Networks ############

######## 2.1 Network helpers ########

class NetworkOutput(NamedTuple):
    value: float
    correctness_value_logits: jnp.ndarray
    latency_value_logits: jnp.ndarray
    policy_logits: Dict[Action, float]


class Network(object):
    """Wrapper around Representation and Prediction networks."""

    def __init__(self, hparams: ml_collections.ConfigDict, task_spec: TaskSpec):
        self.hparams = hparams
        self.task_spec = task_spec
        self.training_steps = 0
        self.representation = None
        self.prediction = None
        self.params = None
    
    def init_network(self, action_space: ActionSpace) -> None:
        hparams = self.hparams
        task_spec = self.task_spec
        def representation_fn(*a, **k):
            return RepresentationNet(hparams, task_spec, hparams.embedding_dim)(*a, **k)
        
        def prediction_fn(*a, **k):
            return PredictionNet(
                task_spec=task_spec,
                value_max=hparams.value.max,
                value_num_bins=hparams.value.num_bins,
                embedding_dim=hparams.embedding_dim,
            )(*a, **k)
        
        # Transform the functions that build and apply modules
        self.representation = hk.transform(representation_fn)
        self.prediction = hk.transform(prediction_fn)
        # Initialize parameters using PRNGKeys
        rep_key, pred_key = jax.random.split(jax.random.PRNGKey(42), 2)
        # We need dummy inputs to initialize the networks
        # Observation has four fields:
        #    - program: shape (1, max_program_size, 3) -- opcode, arg1, arg2 in range [0, num_regs + num_mem)] # TODO: ensure memory offset
        #    - program_length: shape (num_inputs,) -- can be used to maks the program
        #    - registers: shape (1, num_inputs, num_regs)
        #    - memory: shape (1, num_inputs, num_mem)
        dummy_obs = {
            'program': jnp.zeros((1, task_spec.max_program_size, 3), dtype=jnp.int32),
            'program_length': jnp.zeros((1,task_spec.num_inputs,), dtype=jnp.int32) + 3,
            'registers': jnp.zeros((1, task_spec.num_inputs, task_spec.num_regs), dtype=jnp.int32),
            'memory': jnp.zeros((1, task_spec.num_inputs, task_spec.num_mem), dtype=jnp.int32),
        }
        logger.debug("init_params: dummy_obs %s", str({k:v.shape for k,v in dummy_obs.items()}))
        self.params = {
            'representation': self.representation.init(rep_key, dummy_obs),
            'prediction': self.prediction.init(pred_key, jnp.zeros((1, hparams.embedding_dim)), action_space),
        }

    def inference(self, params: Any, observation: Dict, action_space: ActionSpace) -> NetworkOutput:
        if self.representation is None or self.prediction is None:
            raise ValueError("Network not initialized. Call init_network() first.")
        embedding = self.representation.apply(params['representation'], None, observation)
        return self.prediction.apply(params['prediction'], None, embedding, action_space)

    def get_params(self):
        # Returns the weights of this network.
        return self.params

    def update_params(self, updates: Any) -> None:
        # Update network weights internally.
        self.params = jax.tree.map(lambda p, u: p + u, self.params, updates)

    def copy(self):
        # Returns a copy of the network.
        net = Network(self.hparams, self.task_spec)
        net.representation = self.representation
        net.prediction = self.prediction
        net.params = jax.tree.map(lambda p: p.copy(), self.params)
        return net


class UniformNetwork(object):
    """Network representation that returns uniform output."""
    # NOTE: this module is returned instead of the Network in case no parameters are in the buffer.
    # pylint: disable-next=unused-argument
    def inference(self, observation, action_space: x86ActionSpace) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(
            value=jax.random.uniform(jax.random.PRNGKey(0), minval=-1.0, maxval=1.0),
            correctness_value_logits=jax.random.uniform(jax.random.PRNGKey(1), shape=(1,), minval=-1.0, maxval=1.0),
            latency_value_logits=jax.random.uniform(jax.random.PRNGKey(2), shape=(1,), minval=-1.0, maxval=1.0),
            policy_logits={a: jax.random.uniform(jax.random.PRNGKey(3), minval=-1.0, maxval=1.0)
                for a in action_space._actions.values()},
        )

    def get_params(self):
        # Returns the weights of this network.
        return {}

    def update_params(self, updates: Any) -> None:
        # Update network weights internally.
        pass

    @property
    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0


######## 2.2 Representation Network ########


class MultiQueryAttentionBlock(hk.Module):
    """Attention with multiple query heads and a single shared key and value head.

    Implementation of "Fast Transformer Decoding: One Write-Head is All You Need",
    see https://arxiv.org/abs/1911.02150.
    """
    def __init__(self,
            attention_params: ml_collections.ConfigDict,
            name: str | None = None,
        ):
        super().__init__(name=name)
        self.head_depth = attention_params.head_depth
        self.num_heads = attention_params.num_heads
        self.attention_dropout = attention_params.attention_dropout
        self.position_encoding = attention_params.position_encoding
    
    def __call__(self, inputs, encoded_state=None):
        """
        Tensoflow implementatiion from the paper:
        def MultiqueryAttentionBatched(X, M, mask , P_q, P_k, P_v, P_o) :
            \""" Multi-Query Attention.
            Args :
                X: Inputs (queries    shape [ b, n, d] 
                M: other inputs (k/v) shape [ b, m, d]
                mask : a tensor with  shape [ b, h, n , m]
                P_q: Query proj mat   shape [ h, d, k]
                P_k: Key proj mat     shape [    d, k]
                P_v: Value proj mat   shape [    d, v]
                P_o: Output proj mat  shape [ h, d, v]
            where 
                'h' is the number of heads, 
                'm' is the number of input vectors,
                'n' is the number of inputs, for which we want to compute the attention
                'd' is the dimension of the input vectors,
            Returns :
                Y: a tensor with shape [ b , n , d ]
            \"""
            Q = tf.einsum ( "bnd, hdk->bhnk " , X, P_q)
            K = tf.einsum ( "bmd, dk->bmk" , M, P_k)
            V = tf.einsum ( "bmd, dv->bmv" , M, P_v)
            logits = tf.einsum ( " bhnk , bmk->bhnm " , Q, K)
            weights = tf.softmax ( logits + mask )
            O = tf.einsum ( "bhnm, bmv->bhnv " , weights , V)
            Y = tf.einsum ( "bhnv , hdv->bnd " , O, P_o)
            return Y
        """
        # P_q, P_k, P_v, P_o are parameters, which we declare here
        Q = self._linear_projection(inputs, self.num_heads, self.head_depth, name='P_q') # B x N x H x K
        K = self._linear_projection(inputs, 1, self.head_depth, name='P_k') # B x M x K
        V = self._linear_projection(inputs, 1, self.head_depth, name='P_v') # B x M x V
        
        logger.debug("MQAB: logits einsum Q (bnhk) %s, K (bmk) %s -> bhnm", Q.shape, K.shape)
        logits = jnp.einsum("bnhk,bmk->bhnm", Q, K) # B x N x H x M
        logger.debug("MQAB: logits shape (bhnm) %s", logits.shape)
        weights = jax.nn.softmax(logits) # NOTE: no causal masking, this is an encoder block
        logger.debug("MQAB: weights shape (bhnm) %s", weights.shape)
        if self.attention_dropout: # boolean
            logger.debug("MQAB: applying attention dropout %s", self.attention_dropout)
            weights = hk.dropout(hk.next_rng_key(), self.attention_dropout, weights)
        logger.debug("MQAB: output projection einsum weights (bhnm) %s, V (bmv) %s -> bhnv", weights.shape, V.shape)
        O = jnp.einsum("bhnm,bmv->bhnv", weights, V) # B x N x H x V
        logger.debug("MQAB: O shape (bhnv) %s", O.shape)
        B, _, N, _ = O.shape # B x H x N x V
        # reshape O to B x N x (H*V)
        O = O.reshape((B, N, -1)) # B x N x (H*V)
        logger.debug("MQAB: O reshaped to %s", O.shape)
        # apply the output projection
        logger.debug("MQAB: aggregate head projection bn[h*v] %s to bnd %s ", O.shape, inputs.shape)
        Y = self._linear_projection(O, 1, inputs.shape[-1], name='P_o') # B x N x V
        logger.debug("MQAB: output shape (bnd) %s", Y.shape)
        assert Y.shape == inputs.shape, f"Output shape {Y.shape} does not match input shape {inputs.shape}."
        
        return Y # B x N x D
    
    @hk.transparent
    def _linear_projection(
        self,
        x: jax.Array, # [B, N, D] (batch, seqence length, embedding dim)
        num_heads: int,
        head_size: int,
        name: str | None = None,
    ) -> jax.Array:
        """Copy-paste from hhk.MultiHeadAttention."""
        y = hk.Linear(num_heads * head_size, name=name)(x) # proj mat. D x (H*K) 
        # y should be [B, N, H*K]
        assert y.ndim == 3, f"y should be 3D, but got {y.ndim}D"
        assert y.shape[-1] == num_heads * head_size, f"y should be of shape [B, N, H*K], but got {y.shape}"
        *leading_dims, _ = x.shape # [B, N, D]
        # split last dimension (H*K) into H, K
        new_shape = (*leading_dims, num_heads, head_size) if num_heads > 1 else (*leading_dims, head_size)
        logger.debug("linear_projection, reshaping y from %s to %s", y.shape, new_shape)
        return y.reshape(new_shape) # [B, N, H, K] or [B, N, K]

    def sinusoid_position_encoding(seq_size, feat_size):
        """Compute sinusoid absolute position encodings, 
        given a sequence size and feature dimensionality"""
        # SOURCE: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
        pe = jnp.zeros((seq_size, feat_size))
        position = jnp.arange(0, seq_size, dtype=jnp.float32)[:,None]
        div_term = jnp.exp(jnp.arange(0, feat_size, 2) * (-math.log(10000.0) / feat_size))
        pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[None]
        return pe


class ResBlockV2(hk.Module):
    """Layer-normed variant of the block from https://arxiv.org/abs/1603.05027.
    Implementation based on dm-haiku's ResNetBlockV2.
    """
    def __init__(
        self,
        channels: int,
        stride: int | Sequence[int] = 1,
        use_projection: bool = False,
        ln_config: Mapping[str, Any] = {},
        bottleneck: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.use_projection = use_projection

        ln_config = dict(ln_config)
        ln_config.setdefault("axis", -1)
        ln_config.setdefault("create_scale", True)
        ln_config.setdefault("create_offset", True)
        
        if self.use_projection:
            self.proj_conv = hk.Conv1D(
                output_channels=channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="SAME",
                name="shortcut_conv")

        channel_div = 4 if bottleneck else 1
        conv_0 = hk.Conv1D(
            output_channels=channels // channel_div,
            kernel_shape=1 if bottleneck else 3,
            stride=1 if bottleneck else stride,
            with_bias=False,
            padding="SAME",
            name="conv_0")

        ln_0 = hk.LayerNorm(name="LayerNorm_0", **ln_config)

        conv_1 = hk.Conv1D(
            output_channels=channels // channel_div,
            kernel_shape=3,
            stride=stride if bottleneck else 1,
            with_bias=False,
            padding="SAME",
            name="conv_1")

        ln_1 = hk.LayerNorm(name="LayerNorm_1", **ln_config)
        layers = ((conv_0, ln_0), (conv_1, ln_1))

        if bottleneck:
            conv_2 = hk.Conv1D(
                output_channels=channels,
                kernel_shape=1,
                stride=1,
                with_bias=False,
                padding="SAME",
                name="conv_2")

            # NOTE: Some implementations of ResNet50 v2 suggest initializing
            # gamma/scale here to zeros.
            ln_2 = hk.LayerNorm(name="LayerNorm_2", **ln_config)
            layers = layers + ((conv_2, ln_2),)

        self.layers = layers

    def __call__(self, inputs, is_training=True, test_local_stats=False):
        # FIXME: figure out what to do with the is_training and test_local_stats
        logger.debug("ResBlockV2: inputs shape %s", inputs.shape)
        x = shortcut = inputs

        for i, (conv_i, ln_i) in enumerate(self.layers):
            x = ln_i(x)
            x = jax.nn.relu(x)
        if i == 0 and self.use_projection:
            shortcut = self.proj_conv(x)
        x = conv_i(x)

        return x + shortcut


def int2bin(integers_array: jnp.array) -> jnp.array:
    """Converts an array of integers to an array of its 32bit representation bits.

    Conversion goes from array of shape (S1, S2, ..., SN) to (S1, S2, ..., SN*32),
    i.e. all binary arrays are concatenated. Also note that the single 32-long
    binary sequences are reversed, i.e. the number 1 will be converted to the
    binary 1000000... . This is irrelevant for ML problems.

    Args:
        integers_array: array of integers to convert.

    Returns:
        array of bits (on or off) in boolean type.
    """
    flat_arr = integers_array.astype(jnp.int32).reshape(-1, 1)
    bin_mask = jnp.tile(2 ** jnp.arange(32), (flat_arr.shape[0], 1))
    return ((flat_arr & bin_mask) != 0).reshape(
        *integers_array.shape[:-1], integers_array.shape[-1] * 32
    )


def bin2int(binary_array: jnp.array) -> jnp.array:
    """Reverses operation of int2bin."""
    u_binary_array = binary_array.reshape(
        *binary_array.shape[:-1], binary_array.shape[-1] // 32, 32
    )
    exp = jnp.tile(2 ** jnp.arange(32), u_binary_array.shape[:-1] + (1,))
    return jnp.sum(exp * u_binary_array, axis=-1)


class RepresentationNet(hk.Module):
    """Representation network."""

    def __init__(
        self,
        hparams: ml_collections.ConfigDict,
        task_spec: TaskSpec,
        embedding_dim: int,
        name: str = 'representation',
    ):
        super().__init__(name=name)
        self._hparams = hparams
        self._task_spec = task_spec
        self._embedding_dim = embedding_dim

    def __call__(self, inputs):
        logger.debug("representation_net program shape %s", inputs['program'].shape)
        # inputs is the observation dict
        batch_size = inputs['program'].shape[0]

        program_encoding = None
        if self._hparams.representation.use_program:
            program_encoding = self._encode_program(inputs, batch_size)

        if (
            self._hparams.representation.use_locations # i.e. CPU state
            and self._hparams.representation.use_locations_binary
        ):
            raise ValueError(
                'only one of `use_locations` and `use_locations_binary` may be used.'
            )
        # encode the locations (registers and memory) in the CPU state
        locations_encoding = None
        if self._hparams.representation.use_locations:
            locations_encoding = self._make_locations_encoding_onehot(
                inputs, batch_size
            )
        elif self._hparams.representation.use_locations_binary:
            locations_encoding = self._make_locations_encoding_binary(
                inputs, batch_size
            )

        # NOTE: this is not used.
        permutation_embedding = None
        if self._hparams.representation.use_permutation_embedding:
            permutation_embedding = self.make_permutation_embedding(batch_size)

        # aggregate the locations and the program to produce a single output vector
        return self.aggregate_locations_program(
            locations_encoding, permutation_embedding, program_encoding, batch_size
        )

    def _encode_program(self, inputs, batch_size):
        logger.debug("encode_program shape %s", inputs['program'].shape)
        program = inputs['program']
        max_program_size = inputs['program'].shape[1] # TODO: this might not be a constant
        program_length = inputs['program_length'].astype(jnp.int32)
        program_onehot = self.make_program_onehot(
            program, batch_size, max_program_size
        )
        program_encoding = self.apply_program_mlp_embedder(program_onehot)
        program_encoding = self.apply_program_attention_embedder(program_encoding)
        # select the embedding corresponding to the current instruction in the corr. CPU state
        return self.pad_program_encoding( # size B x num_inputs x embedding_dim
            program_encoding, batch_size, program_length, max_program_size
        )

    def aggregate_locations_program(
        self,
        locations_encoding,
        unused_permutation_embedding,
        program_encoding,
        batch_size,
    ):
        # note that Haiku passes the parameters to the entire class
        # when apply() is called on it. So don't look for parameters here.
        locations_embedder = hk.Sequential(
            [
                # input is embedding_dim size, because we already encoded in either one-hot or binary
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name='per_locations_embedder',
        )

        # locations_encoding.shape == [B, P, D] so map embedder across locations to
        # share weights
        logger.debug("locations_encoding shape %s", locations_encoding.shape)
        
        locations_embedding = hk.vmap(
            locations_embedder, in_axes=1, out_axes=1, split_rng=False
        )(locations_encoding)
        logger.debug("locations_embedding shape %s", locations_embedding.shape)

        # broadcast the program encoding for each example.
        # this way, it matches the size of the observations.
        logger.debug("program_encoding shape %s", program_encoding.shape)
        program_encoded_repeat = self.repeat_program_encoding(
            program_encoding, batch_size
        )
        logger.debug("program_encoded_repeat shape %s", program_encoded_repeat.shape)

        grouped_representation = jnp.concatenate( # concat the CPU state and the program.
            [locations_embedding, program_encoded_repeat], axis=-1
        )
        logger.debug("grouped_representation shape %s", grouped_representation.shape)

        return self.apply_joint_embedder(grouped_representation, batch_size)

    def repeat_program_encoding(self, program_encoding, batch_size):
        logger.debug("repeat_program_encoding pre shape %s", program_encoding.shape)
        program_encoding = jnp.broadcast_to(
            program_encoding,
            [batch_size, self._task_spec.num_inputs, program_encoding.shape[-1]],
        )
        logger.debug("repeat_program_encoding post shape %s", program_encoding.shape)
        return program_encoding

    def apply_joint_embedder(self, grouped_representation, batch_size):
        all_locations_net = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name='per_element_embedder',
        )
        joint_locations_net = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name='joint_embedder',
        )
        joint_resnet = [
            ResBlockV2(self._embedding_dim, name=f'joint_resblock_{i}')
            for i in range(self._hparams.representation.repr_net_res_blocks)
        ]

        chex.assert_shape(
            grouped_representation, (batch_size, self._task_spec.num_inputs, None)
        )
        logger.debug("apply_joint_embedder grouped_rep shape %s", grouped_representation.shape)
        # apply MLP to the combined program and locations embedding
        permutations_encoded = all_locations_net(grouped_representation)
        logger.debug("apply_joint_embedder permutations_encoded shape %s", permutations_encoded.shape)
        # Combine all permutations into a single vector using a ResNetV2
        joint_encoding = joint_locations_net(jnp.mean(permutations_encoded, axis=1))
        logger.debug("apply_joint_embedder joint_encoding shape %s", joint_encoding.shape)
        for net in joint_resnet:
            joint_encoding = net(joint_encoding)
        return joint_encoding

    def make_program_onehot(self, program, batch_size, max_program_size):
        logger.debug("make_program_onehot shape %s", program.shape)
        func = program[:, :, 0] # the opcode -- int
        arg1 = program[:, :, 1] # the first operand -- int 
        arg2 = program[:, :, 2] # the second operand -- int
        func_onehot = jax.nn.one_hot(func, self._task_spec.num_funcs)
        arg1_onehot = jax.nn.one_hot(arg1, self._task_spec.num_locations)
        arg2_onehot = jax.nn.one_hot(arg2, self._task_spec.num_locations)
        logger.debug("func %s, arg1 %s, arg2 %s", func_onehot.shape, arg1_onehot.shape, arg2_onehot.shape)
        program_onehot = jnp.concatenate(
            [func_onehot, arg1_onehot, arg2_onehot], axis=-1
        )
        chex.assert_shape(program_onehot, (batch_size, max_program_size, None))
        logger.debug("program_onehot shape %s", program_onehot.shape)
        return program_onehot

    def pad_program_encoding(
        self, program_encoding, batch_size, program_length, max_program_size
    ):
        """Pads the program encoding to account for state-action stagger."""
        logger.debug("pad_program_encoding shape %s", program_encoding.shape)
        chex.assert_shape(program_encoding, (batch_size, max_program_size, None))
        chex.assert_shape(program_length, (batch_size, self._task_spec.num_inputs))

        empty_program_output = jnp.zeros(
            [batch_size, program_encoding.shape[-1]],
        )
        program_encoding = jnp.concatenate(
            [empty_program_output[:, None, :], program_encoding], axis=1
        )

        program_length_onehot = jax.nn.one_hot(program_length, max_program_size + 1)
        logger.debug("pad_program_encoding pre program_length_onehot shape %s", program_length_onehot.shape)
        logger.debug("pad_program_encoding pre program_encoding shape %s", program_encoding.shape)
        # two cases here:
        # - program length is a batch of scalars corr. to the program length
        # - program length is a batch of vectors (of len num_inputs) corr. to the state of the program counters
        if len(program_length_onehot.shape) == 3:
            program_encoding = jnp.einsum(
                'bnd,bNn->bNd', program_encoding, program_length_onehot
            )
        else:
            program_encoding = jnp.einsum(
                'bnd,bn->bd', program_encoding, program_length_onehot
            )
        logger.debug("pad_program_encoding post program_encoding shape %s", program_encoding.shape)

        return program_encoding

    def apply_program_mlp_embedder(self, program_encoding):
        # input: B x P x (num_funcs + 2 * num_locations)
        # output: B x P x embedding_dim
        # P is the program length
        logger.debug("apply_program_mlp_embedder program shape %s, embedding_dim %s", program_encoding.shape, self._embedding_dim)
        program_embedder = hk.Sequential(
            [
                hk.Linear(self._embedding_dim), # (nF + 2*nL) x D -- input size is decided automatically
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self._embedding_dim), # D x D
            ],
            name='per_instruction_program_embedder',
        )
        logger.debug("program.shape %s -->", program_encoding.shape)
        logger.debug("  program_embedder %s", program_embedder)
        program_encoding = program_embedder(program_encoding)
        logger.debug("program_encoding shape %s <--", program_encoding.shape)
        return program_encoding

    def apply_program_attention_embedder(self, program_encoding):
        logger.debug("apply_program_attention_embedder program shape %s", program_encoding.shape)
        # input is B x P x D (batch, program length, embedding dim)
        # output is B x P x D
        _, program_length, d = program_encoding.shape
        assert program_length == self._task_spec.max_program_size, (
            f"program length {program_length} does not match max program size "
            f"{self._task_spec.max_program_size}"
        )
        assert d == self._embedding_dim, (
            f"program encoding dim {d} does not match embedding dim {self._embedding_dim}"
        ) 
        attention_params = self._hparams.representation.attention
        make_attention_block = functools.partial(
            MultiQueryAttentionBlock, attention_params
        )
        attention_encoders = [
            make_attention_block(name=f'attention_program_sequencer_{i}')
            for i in range(self._hparams.representation.attention_num_layers)
        ]

        *_, seq_size, feat_size = program_encoding.shape

        position_encodings = jnp.broadcast_to(
            MultiQueryAttentionBlock.sinusoid_position_encoding(
                seq_size, feat_size
            ),
            program_encoding.shape,
        )
        program_encoding += position_encodings

        for e in attention_encoders:
            logger.debug("apply_program_attention_embedder layer %s", e.name)
            program_encoding = e(program_encoding, encoded_state=None)
        logger.debug("apply_program_attention_embedder post MQAB %s", program_encoding.shape)

        return program_encoding

    def _make_locations_encoding_onehot(self, inputs, batch_size):
        """Creates location encoding using onehot representation."""
        logger.debug("make_locations_encoding_onehot shapes %s", str({k:v.shape for k,v in inputs.items()}))
        memory = inputs['memory'] # B x E x M (batch, num_inputs, memory size)
        registers = inputs['registers'] # B x E x R (batch, num_inputs, register size)
        # NOTE: originall implementation suggests the shape [B, H, P, D]
        # where we can only assume that 
        #   B - batch,
        #   H - num_inputs,
        #   P - program length,
        #   D - num_locations
        # this goes against what the paper suggests (although very vaguely)
        # that only the current state is passed to the network as input,
        # instead of the whole sequence of states,
        # that the CPU has seen while executing the program.
        locations = jnp.concatenate([registers, memory], axis=-1) # B x E x (R + M)
        logger.debug("locations shape %s", locations.shape)
        # to support inputs with sequences of states, we conditinally transpose
        # the locations tensor to have the shape [B, P, H, D]
        if len(locations.shape) == 4:
            # in this case, locations is [B, H, P, D]
            # and we need to transpose it to [B, P, H, D]
            locations = jnp.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D]

        # One-hot encode the values in the memory and average everything across
        # permutations.
        locations_onehot = jax.nn.one_hot( # shape is now B x E x num_locations x num_locations
            locations, self._task_spec.num_locations, dtype=jnp.float32
        )
        logger.debug("locations_onehot shape %s", locations_onehot.shape)
        locations_onehot = locations_onehot.reshape(
            [batch_size, self._task_spec.num_inputs, -1]
        )
        logger.debug("locations_onehot reshaped to %s", locations_onehot.shape)
        return locations_onehot

    def _make_locations_encoding_binary(self, inputs, batch_size):
        """Creates location encoding using binary representation."""

        memory_binary = int2bin(inputs['memory']).astype(jnp.float32)
        registers_binary = int2bin(inputs['registers']).astype(jnp.float32)
        # Note the extra I dimension for the length of the binary integer (32)
        locations = jnp.concatenate(
            [memory_binary, registers_binary], axis=-1
        )  # [B, H, P, D*I]
        locations = jnp.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D*I]

        locations = locations.reshape([batch_size, self._task_spec.num_inputs, -1])

        return locations

######## 2.3 Prediction Network ########


def make_head_network(
    embedding_dim: int,
    output_size: int,
    num_hidden_layers: int = 2,
    name: Optional[str] = None,
) -> Callable[[jnp.ndarray,], jnp.ndarray]:
    return hk.Sequential(
        [ResBlockV2(embedding_dim) for _ in range(num_hidden_layers)]
        + [hk.Linear(output_size)],
        name=name,
    )


class DistributionSupport(object):

    def __init__(self, value_max: float, num_bins: int):
        self.value_max = value_max
        self.num_bins = num_bins

    def mean(self, logits: jnp.ndarray) -> float:
        logger.debug("DistSup.mean compute twohot logits for %s", logits.shape)
        twohot_logits = jnp.stack([
            self.scalar_to_two_hot(i) for i in logits
        ])
        logger.debug("DistSup.mean twohot logits shape %s", twohot_logits.shape)
        sum_twohots = jnp.sum(twohot_logits, axis=1)
        logger.debug("DistSup.mean sum_twohots shape %s", sum_twohots.shape)
        assert logits.shape == sum_twohots.shape, f"Logits shape {logits.shape} and sum_twohots shape {sum_twohots.shape} do not match."
        mean = sum_twohots / jnp.sum(sum_twohots)
        logger.debug("DistSup.mean mean shape %s", mean.shape)
        return mean

    def scalar_to_two_hot(self, scalar: float) -> jnp.ndarray:
        """
        Converts a scalar to a two-hot encoding.
        Finds the two closest bins to the scalar (lower and upper) and
        sets these indices to 1. All other indices are set to 0.
        """
        # Bins are -probably- a linear interpolation between 0 and value_max
        # and we need to assign non-zero values to the two closest bins
        # based on proximity to the scalar.
        toohot = jnp.zeros(self.num_bins, dtype=jnp.float32)
        # bin indices correspond to scalar values uniformly
        # distributed on the range [0,value_max]
        # val(bin_i) = i * (value_max / num_bins)
        # i = val(bin_i) * (value_max / num_bins)^-1
        # logger.debug("scalar_to_two_hot: value_max %s, num_bins %s", self.value_max, self.num_bins)
        step = self.value_max / self.num_bins
        # logger.debug("scalar_to_two_hot:\n scalar %s,\n step %s", scalar, step)
        low_bin = jnp.floor(scalar / step).astype(jnp.int32)
        high_bin = jnp.ceil(scalar / step).astype(jnp.int32)
        # low_bin and high_bin are the indices of the two closest bins
        # to the scalar value.
        low_prox = jnp.abs(scalar - low_bin * step)
        high_prox = jnp.abs(scalar - high_bin * step)
        # low_prox and high_prox are the distances from the scalar
        # to the two closest bins.
        low_weight = 1 - low_prox / (low_prox + high_prox)
        high_weight = 1 - low_weight
        
        # # logger.debug("scalar_to_two_hot:\n scalar %s,\n low_bin %s,\n high_bin %s,\n low_prox %s,\n high_prox %s\n low_weight %s\n high_weight %s ", scalar, low_bin, high_bin, low_prox, high_prox, low_weight, high_weight)
        
        toohot = toohot.at[low_bin].set(low_weight)
        toohot = toohot.at[high_bin].set(high_weight)
        
        # logger.debug("scalar_to_two_hot: sum = %s", jnp.sum(toohot))
        
        return toohot

class CategoricalHead(hk.Module):
    """A head that represents continuous values by a categorical distribution."""

    def __init__(
        self,
        embedding_dim: int,
        support: DistributionSupport,
        name: str = 'CategoricalHead',
    ):
        super().__init__(name=name)
        self._value_support = support
        self._embedding_dim = embedding_dim
        self._head = make_head_network(
            embedding_dim, output_size=self._value_support.num_bins
        )

    def __call__(self, x: jnp.ndarray):
        # For training returns the logits, for inference the mean.
        logits = self._head(x) # project the embedding to the value support's numbeer of bins 
        probs = jax.nn.softmax(logits) # take softmax
        mean = jax.vmap(self._value_support.mean)(probs) # compute the mean
        return dict(logits=logits, mean=mean)

class PredictionNet(hk.Module):
    """MuZero prediction network."""

    def __init__(
        self,
        task_spec: TaskSpec,
        value_max: float,
        value_num_bins: int,
        embedding_dim: int,
        name: str = 'prediction',
    ):
        super().__init__(name=name)
        self.task_spec = task_spec
        self.value_max = value_max
        self.value_num_bins = value_num_bins
        self.support = DistributionSupport(self.value_max, self.value_num_bins)
        self.embedding_dim = embedding_dim

    def __call__(self, embedding: jnp.ndarray, action_space: x86ActionSpace):
        logger.debug("PredictionNet: embedding shape %s", embedding.shape)
        policy_head = make_head_network(
            self.embedding_dim, self.task_spec.num_actions
        )
        logger.debug("PredictionNet: policy_head %s", policy_head)
        value_head = CategoricalHead(self.embedding_dim, self.support)
        logger.debug("PredictionNet: value_head %s", value_head)
        latency_value_head = CategoricalHead(self.embedding_dim, self.support)
        logger.debug("PredictionNet: latency_value_head %s", latency_value_head)
        correctness_value = value_head(embedding)
        logger.debug("PredictionNet: correctness_value shape %s", str({k:v.shape for k, v in correctness_value.items()}))
        latency_value = latency_value_head(embedding)
        logger.debug("PredictionNet: latency_value shape %s", str({k:v.shape for k, v in latency_value.items()}))

        policy = policy_head(embedding)
        logger.debug("Policy shape: %s", policy.shape)
        policy_dict = { # map Action -> logit
            a: logit for a, logit in zip(action_space._actions.values(), policy)
        }
        assert isinstance(list(policy_dict.keys())[0], Action), \
            f"Expected action to be of type Action, got {type(list(policy_dict.keys())[0])}"

        output = NetworkOutput(
            value=correctness_value['mean'] + latency_value['mean'],
            correctness_value_logits=correctness_value['logits'],
            latency_value_logits=latency_value['logits'],
            policy_logits=policy_dict,
        )
        logger.debug("PredictionNet: output %s", str({k: v.shape for k, v in output._asdict().items() if isinstance(v, jnp.ndarray)}))
        return output


####### End Networks ########
#############################

#############################
####### 3. Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class AlphaDevConfig(object):
    """AlphaDev configuration."""

    def __init__(
        self,
    ):
        ### Self-Play
        self.num_actors = 128  # TPU actors
        # pylint: disable-next=g-long-lambda
        self.visit_softmax_temperature_fn = lambda steps: (
            1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25
        )
        self.max_moves = jnp.inf
        self.num_simulations = 5
        self.discount = 1.0

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.03
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.known_bounds = KnownBounds(-6.0, 6.0)

        # Environment: spec of the Variable Sort 3 task
        self.task_spec = TaskSpec(
            max_program_size=10,
            num_inputs=17,
            num_funcs=len(x86_opcode2int),
            num_locations=19,
            num_regs=5,
            num_mem=14,
            num_actions=240, # original value was 271
            correct_reward=1.0,
            correctness_reward_weight=2.0,
            latency_reward_weight=0.5,
            latency_quantile=0,
        )
        # TODO: this assert is stupid.
        assert self.task_spec.num_locations == self.task_spec.num_regs + self.task_spec.num_mem, \
            f"number of registers {self.task_spec.num_regs} and memory {self.task_spec.num_mem} do not add up to the number of locations {self.task_spec.num_locations}"

        ### Network architecture
        self.hparams = ml_collections.ConfigDict()
        self.hparams.embedding_dim = 512
        self.hparams.representation = ml_collections.ConfigDict()
        self.hparams.representation.use_program = True
        self.hparams.representation.use_locations = True
        self.hparams.representation.use_locations_binary = False
        self.hparams.representation.use_permutation_embedding = False
        self.hparams.representation.repr_net_res_blocks = 8
        self.hparams.representation.attention = ml_collections.ConfigDict()
        self.hparams.representation.attention.head_depth = 128
        self.hparams.representation.attention.num_heads = 4
        self.hparams.representation.attention.attention_dropout = False
        self.hparams.representation.attention.position_encoding = 'absolute'
        self.hparams.representation.attention_num_layers = 6
        self.hparams.value = ml_collections.ConfigDict()
        self.hparams.value.max = 3.0  # These two parameters are task / reward-
        self.hparams.value.num_bins = 301  # dependent and need to be adjusted.

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = 500
        self.target_network_interval = 100
        self.window_size = int(1e6)
        self.batch_size = 1
        self.td_steps = 5
        self.lr_init = 2e-4
        self.momentum = 0.9
        
        self.inputs = generate_sort_inputs(3, self.task_spec.num_inputs)
        #   logger.debug("Inputs: %s", self.inputs)
        assert self.task_spec.num_inputs == len(self.inputs),\
            f"Expected {self.task_spec.num_inputs} inputs, got {len(self.inputs)}"

    def new_game(self):
        return Game(self.task_spec.num_actions, self.discount, self.task_spec, self.inputs)


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Player(object):
    # NOTE: single player game, so we don't care about this.
    # probably included to fit the AlphaZero framework.
    pass


class Node(object):
    """MCTS node."""

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        # self.hidden_state = None # TODO: what is this?
        self.reward = 0

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def __repr__(self):
        return f"Node(vc={self.visit_count}, prior={self.prior}, value={self.value()}, reward={self.reward})"

    def show_tree(self, level: int = 0, only_expanded: bool = False):
        """Prints the tree starting from this node."""
        print(" " * level, self)
        num_since_expanded = 0
        for action, child in self.children.items():
            if only_expanded and not child.expanded():
                num_since_expanded += 1
                continue
            if num_since_expanded > 0:
                print(" " * (level + 2), f"({num_since_expanded} not expanded)")
                num_since_expanded = 0
            print(" " * (level + 2), action)
            child.show_tree(level=level + 4, only_expanded=only_expanded)


class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: Sequence[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def to_play(self) -> Player:
        return Player()


class Target(NamedTuple):
    correctness_value: float
    latency_value: float
    policy: Sequence[int]
    bootstrap_discount: float


class Sample(NamedTuple):
    observation: Dict[str, jnp.ndarray]
    bootstrap_observation: Dict[str, jnp.ndarray]
    target: Target


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(
        self, action_space_size: int, discount: float, task_spec: TaskSpec, inputs: List[IOExample]
    ):
        self.task_spec = task_spec
        self.inputs = inputs
        # TODO: figure out how to split num_locations to registers and memory.
        # FIXME: this is hard-coded for the 3-sort task for now.
        self.action_space_storage = x86ActionSpaceStorage(max_reg=task_spec.num_regs, max_mem=task_spec.num_mem)
        self.environment = AssemblyGame(task_spec, inputs, self.action_space_storage)
        self.history = []
        self.rewards = []
        self.latency_reward = 0
        self.child_visits = []
        self.root_values = []
        # TODO: action space size is redundant and prone to be wrong.
        # action space size should not really be stored, especially when we use dynamic action spaces.
        # leaving this for now to the same of consistency with the original code.
        self.action_space_size = action_space_size
        assert action_space_size == len(self.action_space_storage.actions), \
            f"Expected {action_space_size} actions, got {len(self.action_space_storage.actions)}"
        self.discount = discount

    def terminal(self) -> bool:
        # Game specific termination rules.
        # For sorting, a game is terminal if we sort all sequences correctly or
        # we reached the end of the buffer.
        toolong = len(self.history) >= self.task_spec.max_program_size
        iscorrect = self.is_correct()
        return toolong or iscorrect

    def is_correct(self) -> bool:
        # Whether the current algorithm solves the game.
        # for all inputs, right?
        self.environment.correct()

    def legal_actions(self) -> Sequence[Action]:
        # Game specific calculation of legal actions.
        # TODO: implement as an enumeration of the allowed actions,
        # instead of filtering the pre-defined action space. (for the next iteration of this codebase)
        state = self.environment.state()
        # get a mask over the action space, which indicates which actions are valid in the current state.
        actions_mask = self.action_space_storage.get_mask(state, self.history)
        actions = self.action_space_storage.get_space(state).actions
        pruned_actions = actions[actions_mask]
        # pruned_actions = actions # FIXME: this is a hotfix, since pruning is broken rn.
        #   logger.debug("Pruned actions: (len %d/%d)", pruned_actions.shape[0], actions.shape[0])
        # if logger.level == logging.DEBUG:
        #     space = self.action_space_storage.get_space(state)
        #     logger.debug("Legal actions:")
        #     for i, action in enumerate(actions):
        #         if actions_mask[i]:
        #                 logger.debug("  %s", space.get(action))
        
        return [self.action_space_storage.actions[a] for a in pruned_actions.tolist()]

    def apply(self, action: Action):
        _, reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)
        if self.terminal() and self.is_correct():
            self.latency_reward = self.environment.latency_reward()

    def store_search_statistics(self, root: Node):
        # NOTE: this function is used to store statistics about the observed trajectory (outside of MCTS)
        sum_visits = sum(child.visit_count for child in root.children.values())
        # FIXME: in the next iteration with dynamic action spaces, this no longer works.
        action_space = self.action_space_storage.actions
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits
                if a in root.children
                else 0
                for a in action_space.values()
            ]
        )
        self.root_values.append(root.value())

    def make_observation(self, state_index: int):
        logger.debug("Game.make_observation: state_index %d", state_index)
        # NOTE: returns the observation corresponding to the last action
        if state_index == -1:
            return self.environment.observation()
        # re-play the game from the initial state.
        if state_index > len(self.history):
            state_index = len(self.history) # for safety
        env = AssemblyGame(self.task_spec, self.inputs, self.action_space_storage)
        for action in self.history[:state_index]:
            observation, _ = env.step(action)
        return observation

    def make_target(
        # pylint: disable-next=unused-argument
        self, state_index: int, td_steps: int, to_play: Player
    ) -> Target:
        """Creates the value target for training."""
        # The value target is the discounted sum of all rewards until N steps
        # into the future, to which we will add the discounted boostrapped future
        # value.
        # make sure we don't go out of bounds
        bootstrap_index = min(state_index + td_steps, len(self.history))
        # use an offset to account for cases when the td_target is further than the last action.
        # TODO: this might be a mistake and hard to check.
        offset = max(0, (state_index + td_steps) - len(self.history))
        value = 0
        for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
            value += reward * self.discount**(i+offset)  # pytype: disable=unsupported-operands

        if bootstrap_index < len(self.root_values):
            bootstrap_discount = self.discount**td_steps
        else:
            bootstrap_discount = 0

        # NOTE: this is a TD(n) target
        return Target(
            value,
            self.latency_reward, # =0 unless the game is finished
            self.child_visits[state_index],
            bootstrap_discount,
        )

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)


class ReplayBuffer(object):
    """Replay buffer object storing games for training."""

    def __init__(self, config: AlphaDevConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)
        assert len(self.buffer) <= self.window_size+10, \
            f"Replay buffer size growing uncontrollably: {len(self.buffer)} > {self.window_size}"

    def sample_batch(self, td_steps: int) -> Sequence[Sample]:
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        # pylint: disable=g-complex-comprehension
        # FIXME: there is no batching here...
        return [
            Sample(
                observation=g.make_observation(i), # NOTE: this re-computes the game from scratch each time.
                bootstrap_observation=g.make_observation(i + td_steps),
                target=g.make_target(i, td_steps, g.to_play()),
            )
            for (g, i) in game_pos
        ]
        # pylint: enable=g-complex-comprehension

    def sample_game(self) -> Game:
        # sample an index uniformly
        idx = numpy.random.randint(len(self.buffer))
        return self.buffer[idx]

    # pylint: disable-next=unused-argument
    def sample_position(self, game:Game) -> int:
        # Sample position from game either uniformly or according to some priority.
        idx = numpy.random.randint(len(game.history))
        return idx

    def save(self, path: str):
        jnp.save(path, self.buffer)
    def load(self, path: str):
        self.buffer = jnp.load(path, allow_pickle=True)

class SharedStorage(object):
    """Controls which network is used at inference."""

    def __init__(self):
        self._networks = {}
        self._latest_step = 0

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[self._latest_step]
        else:
        # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
        self._latest_step = max(step, self._latest_step)
        logger.debug("SharedStorage: saved network at step %d", step)


##### End Helpers ########
##########################


# AlphaDev training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphadev(config: AlphaDevConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    debug_buffer_path = 'buffer.npy'
    if os.path.exists(debug_buffer_path):
        replay_buffer.load(debug_buffer_path)

    # TODO: this should handled by a multiprocessing pool
    # or a job queue.
    # for _ in range(config.num_actors):
    #     launch_job(run_selfplay, config, storage, replay_buffer)
    if len(replay_buffer.buffer) == 0:
        logger.debug("Starting self-play job")
        launch_job(run_selfplay, config, storage, replay_buffer)
        logger.debug("Self-play job done, saving the buffer to %s", debug_buffer_path)
        replay_buffer.save(debug_buffer_path)
    
    logger.debug("Starting training job")
    # it's fine to keep this in the main thread
    train_network(config, storage, replay_buffer)

    return storage.latest_network()


#####################################
####### 4. Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
# NOTE: runs in parallel with the training job.
# TODO: we should make sure that training and self-play somewhat synchronized
def run_selfplay(
    config: AlphaDevConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)
        logger.debug("Self-play game finished. Game length: %d", len(game.history))
        break


def play_game(config: AlphaDevConfig, network: Network) -> Game:
    """Plays an AlphaDev game. 
    NOTE: i.e. an episode

    Each game is produced by starting at the initial empty program, then
    repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    of the game is reached.

    Args:
        config: An instance of the AlphaDev configuration.
        network: Networks used for inference.

    Returns:
        The played game.
    """

    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        min_max_stats = MinMaxStats(config.known_bounds)

        # Initialisation of the root node and addition of exploration noise
        root = Node(0) # NOTE: a dummy root
        current_observation = game.make_observation(-1)
        state = CPUState(**current_observation) # easier to create a new one
        action_space = game.action_space_storage.get_space(state)
        network_output = network.inference(current_observation, action_space)
        _expand_node( # expand current root
            root, game.to_play(), game.legal_actions(), network_output, reward=0
        )
        _backpropagate(
            [root], # only the root
            network_output.value, # predicted value at the current root
            game.to_play(), # dummy player
            config.discount,
            min_max_stats,
        )
        _add_exploration_noise(config, root)
        
        # NOTE: above, we expanded the current root (after some n moves), effectively initializing the search tree.
        # we used game.legal_actions() to get the action space at the root.
        # 

        # We then run a Monte Carlo Tree Search using the environment.
        run_mcts(
            config,
            root,
            game.action_history(), # newly initialized action history at the current root
            game.action_space_storage, # action space storage
            network,
            min_max_stats,
            game.environment,
        )
        # NOTE: make a move after the MCTS policy improvement step
        action = _select_action(config, len(game.history), root, network, game.action_space_storage)
        logger.debug("play_game: selected action %s", action)
        game.apply(action) # step the environment
        game.store_search_statistics(root)
    return game


def run_mcts(
    config: AlphaDevConfig,
    root: Node,
    action_history: ActionHistory,
    action_space_storage: x86ActionSpaceStorage, # NOTE: added this. also, type should be the superclass
    network: Network,
    min_max_stats: MinMaxStats,
    env: AssemblyGame,
):
    """Runs the Monte Carlo Tree Search algorithm.

    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.

    Args:
        config: AlphaDev configuration
        root: The root node of the MCTS tree from which we start the algorithm
        action_history: history of the actions taken so far.
        network: instances of the networks that will be used.
        min_max_stats: min-max statistics for the tree.
        env: an instance of the AssemblyGame.
    """
    # logger.debug("Running MCTS with %d simulations", config.num_simulations)
    for r_ in range(config.num_simulations): # rollouts
        #   logger.debug("Rollout %d", r_)
        history = action_history.clone()
        node = root # start from the current root
        search_path = [node] # initialise new trajectory from the current root
        sim_env = env.clone() # start from the current state of the environment
        # Traverse the tree until we reach a leaf node.

        #   logger.debug("root node expanded: %s", node.expanded())

        while node.expanded():
            action, node = _select_child(config, node, min_max_stats) # based on UCB
            sim_env.step(action) # step the environment
            history.add_action(action) # update history 
            search_path.append(node) # append to the current trajectory
        
        #   logger.debug("mcts: at leaf. Action: %s, Node: %s", action, node)
        # Inside the search tree we use the environment to obtain the next
        # observation and reward given an action.
        observation, reward = sim_env.observation(), sim_env.correctness_reward()
        action_space = action_space_storage.get_space(CPUState(**observation))
        # get the priors
        network_output = network.inference(observation, action_space)
        _expand_node( # expand the leaf node
            # here, game is not available. I suppose we do not perform action pruning.
            # instead, the history should be initialized either 
            #   with the game's current legal actions. -- UPDATE: this is stupid. The game will never find a solution
            #   or with the action space of the environment.
            # using current legal actions is bad, since as used memory grows we need to allocate new locations
            # using the action space of the environment is also not great because it would allow many illegal actions.
            # pruning the action space during MCTS might entail a big overhead.
            # we need to experiment with this.
            # NOTE: for now, we go with AlphaDev's implementation of returning all actions.
            node, history.to_play(), list(action_space._actions.values()), network_output, reward
        )
        _backpropagate(
            search_path,
            network_output.value,
            history.to_play(),
            config.discount,
            min_max_stats,
        )
    #   logger.debug("MCTS finished. Root node: %s", root)
    #   logger.debug("Tree:\n")
    # root.show_tree(only_expanded=True)

def _select_action(
    # pylint: disable-next=unused-argument
    config: AlphaDevConfig, num_moves: int, node: Node, network: Network,
    action_space_storage: x86ActionSpaceStorage,
) -> Action:
    # logger.debug("select_action at node %s", node)
    visit_counts = jnp.array([
        (child.visit_count, action.index) for action, child in node.children.items()
    ])
    t = config.visit_softmax_temperature_fn(
        steps=network.training_steps
    )
    _, action = softmax_sample(visit_counts, t)
    # i.e. posterior probability based on the visit counts
    # logger.debug("select_action: selected action %s", action)
    return action_space_storage.actions[action]


def _select_child(
    config: AlphaDevConfig, node: Node, min_max_stats: MinMaxStats
):
    """Selects the child with the highest UCB score."""
    scores = [(_ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()]
    # unique_scores, unique_indices = numpy.unique(
    #     [s[0] for s in scores], return_index=True
    # )
    #   logger.debug("select_child: unique scores: %s", unique_scores)
    #   logger.debug("select_child: unique indices: %s", unique_indices)
    _, action, child = max(scores)
    return action, child


def _ucb_score(
    config: AlphaDevConfig,
    parent: Node,
    child: Node,
    min_max_stats: MinMaxStats,
) -> float:
    """Computes the UCB score based on its value + exploration based on prior."""
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(
            child.reward + config.discount * child.value()
        )
    else:
        value_score = 0
    return prior_score + value_score


def _expand_node(
    node: Node,
    to_play: Player,
    actions: Sequence[Action],
    network_output: NetworkOutput,
    reward: float,
):
    """Expands the node using value, reward and policy predictions from the NN."""
    node.to_play = to_play
    # node.hidden_state = network_output.hidden_state
    node.reward = reward
    # Masked softmax. actions() are the legal actions and network output is the prior
    # TODO: use a more efficient softmax instead of the lines below.
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions} # softmax
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    #   logger.debug("Expanded node: %s", node)
    # unique_scores, unique_indices = numpy.unique(
    #     [c.prior for c in node.children.values()], return_index=True
    # )
    #   logger.debug("expand: unique scores: %s", unique_scores)
    #   logger.debug("expand: unique indices: %s", unique_indices)


def _backpropagate(
    search_path: Sequence[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats,
):
    """Propagates the evaluation all the way up the tree to the root."""
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def _add_exploration_noise(config: AlphaDevConfig, node: Node):
    """Adds dirichlet noise to the prior of the root to encourage exploration."""
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


########### End Self-Play ###########
#####################################

#####################################
####### 5. Part 2: Training #########


def train_network(
    config: AlphaDevConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
    """Trains the network on data stored in the replay buffer."""
    # TODO: we might want to raise action space storage 
    # above the context of a game but then we will also have race conditions.
    game = replay_buffer.sample_game()
    action_space_storage = game.action_space_storage # any game.
    action_space = action_space_storage.get_space(game.environment.state())

    network = Network(config.hparams, config.task_spec) # the one we train
    network.init_network(action_space) # initialize the network
    target_network = network.copy() # copy the network for bootstrapping
    
    # init optimizer
    optimizer = optax.sgd(config.lr_init, config.momentum)
    optimizer_state = optimizer.init(network.get_params())

    for i in range(config.training_steps): # insertion point for training pipeline
        logger.info("Training step %d", i)
        network.training_steps = i # increment the training steps
        if i % config.checkpoint_interval == 0:
            logger.info("Saving network at step %d", i)
            storage.save_network(i, network) # save the current network
        if i % config.target_network_interval == 0:
            logger.info("Updating target network at step %d", i)
            target_network = network.copy() # update the bootstrap network
        batch = replay_buffer.sample_batch(config.td_steps)
        optimizer_state = _update_weights(
            optimizer, optimizer_state, network, target_network, batch,
            action_space_storage)
    storage.save_network(config.training_steps, network)

def scale_gradient(tensor: Any, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + jax.lax.stop_gradient(tensor) * (1 - scale)

def _loss_fn(
    network_params: jnp.array,
    target_network_params: jnp.array,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample],
    action_space_storage: ActionSpaceStorage
) -> float:
    """Computes loss."""
    loss = 0
    for observation, bootstrap_obs, target in batch:
        # FIXME: we process the elements in the batch one by one. there is also no batch dimension this way.

        state = CPUState(**observation)
        action_space = action_space_storage.get_space(state)
        # NOTE: re-compute the priors instead of using the cached ones
        # which is fine, we want the updated network to be used for each batch.
        predictions = network.inference(network_params, observation, action_space)
        logger.debug("<train_label>")
        logger.debug("loss_fn: prediction dict shapes %s", str({k:v.shape for k, v in predictions._asdict().items()}))
        logger.debug("loss_fn: policy all zeros %s", (predictions.policy_logits == 0).all())
        
        # TODO: understand the impact of having potentially
        # different action spaces for network and target network.
        bootstrap_space = action_space_storage.get_space(CPUState(**bootstrap_obs))
        bootstrap_predictions = target_network.inference(
            target_network_params, bootstrap_obs, bootstrap_space)
        target_correctness, target_latency, target_policy, bootstrap_discount = (
            target
        )
        target_correctness += (
            bootstrap_discount * bootstrap_predictions.correctness_value_logits
        )

        l = optax.softmax_cross_entropy(predictions.policy_logits, target_policy)
        l += scalar_loss(
            predictions.correctness_value_logits, target_correctness, network
        )
        l += scalar_loss(predictions.latency_value_logits, target_latency, network)
        loss += l
    loss /= len(batch)
    return loss


_loss_grad = jax.grad(_loss_fn, argnums=0)


def _update_weights(
    optimizer: optax.GradientTransformation,
    optimizer_state: Any,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample],
    action_space_storage: ActionSpaceStorage,
) -> Any:
    """Updates the weight of the network."""
    updates = _loss_grad(
        network.get_params(),
        target_network.get_params(),
        network,
        target_network,
        batch,
        action_space_storage)

    optim_updates, new_optim_state = optimizer.update(updates, optimizer_state)
    network.update_params(optim_updates)
    return new_optim_state


def scalar_loss(prediction, target, network) -> float:
    support = network.prediction.support
    return optax.softmax_cross_entropy(
        prediction, support.scalar_to_two_hot(target)
    )


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy.
# pylint: disable-next=unused-argument
def softmax_sample(distribution, temperature: float):
    scaled = distribution[:,0] / temperature
    action_idx = jax.random.categorical(jax.random.PRNGKey(0), scaled)
    return distribution[action_idx].tolist()


def launch_job(f, *args):
    # NOTE: a simple wrapper to launch a job in a separate thread.
    f(*args)

def make_uniform_network():
    return UniformNetwork()

def generate_sort_inputs(items_to_sort: int, num_samples: int=None, rnd_key:int=42) -> List[IOExample]:
    """
    This is equivalent to the C++ code sort_functioons_test.cc:
    TestCases GenerateSortTestCases(int items_to_sort) {
        TestCases test_cases;
        auto add_all_permutations = [&test_cases](const std::vector<int>& initial) {
            std::vector<int> perm(initial);
            do {
            std::vector<int> expected = perm;
            std::sort(expected.begin(), expected.end());
            test_cases.push_back({perm, expected});
            } while (std::next_permutation(perm.begin(), perm.end()));
        };
        // Loop over all possible configurations of binary relations on sorted items.
        // Between each two consecutive items we can insert either '==' or '<'. Then,
        // for each configuration we generate all possible permutations.
        for (int i = 0; i < 1 << (items_to_sort - 1); ++i) {
            std::vector<int> relation = {1};
            for (int mask = i, j = 0; j < items_to_sort - 1; mask /= 2, ++j) {
            relation.push_back(mask % 2 == 0 ? relation.back() : relation.back() + 1);
            }
            add_all_permutations(relation);
        }
        return test_cases;
        }
    """
    # generate all weak orderings
    io_list = []
    def generate_testcases(items_to_sort: int) -> List[Tuple[List[int], List[int]]]:
        #   logger.debug("Generating test cases for %d items to sort", items_to_sort)
        def add_all_permutations(initial: List[int]) -> List[Tuple[List[int], List[int]]]:
            for perm in itertools.permutations(initial, len(initial)):
                expected = numpy.array(sorted(perm))
                io_list.append((numpy.array(perm), expected))
        for i in range(0, items_to_sort+1):
            relation = [1]
            mask = i; j=0
            while j < items_to_sort - 1: # no idea how to express this more pythonic
                j += 1
                relation.append(relation[-1] if mask % 2 == 0 else relation[-1] + 1)
                mask //= 2
            add_all_permutations(relation)

    def remap_input(inp: numpy.ndarray, out: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        mapping = {}
        prev = 0
        for o in out.tolist():
            if o not in mapping:
                mapping[o] = numpy.random.randint(prev+1, prev+3) # don't blow it up. 
                prev = mapping[o]
        out = numpy.array([mapping[o] for o in out.tolist()])
        inp = numpy.array([mapping[i] for i in inp.tolist()])
        return inp, out

    generate_testcases(items_to_sort)
    i_list = numpy.stack([i for i, _ in io_list])
    o_list = numpy.stack([o for _, o in io_list])
    io_list = numpy.stack([i_list, o_list], axis=1)
    _, uidx = numpy.unique(io_list, axis=0, return_index=True) # remove duplicates
    io_list = io_list[uidx, :]
    
    #   logger.debug("Generated %d test cases", len(io_list))
    # shuffle the permutations. if num_samples > len(permutations), we set
    # inputs = permutations + num_samples - len(permutations) random samples from permutations
    # otherwise, we set inputs = random.sample(permutations, num_samples)
    new_indices = numpy.random.permutation(len(io_list))
    perm = io_list[new_indices]
    if num_samples is None:
        io_list = perm
    elif num_samples > len(io_list):
        io_list = jnp.concatenate([io_list, io_list[:num_samples - len(io_list)]])
    else:
        io_list = io_list[:num_samples]
    # convert to list of IOExample
    return [
        IOExample(*remap_input(*io)) for io in io_list
    ]

#definitions for RISC-V dynamic action spaces, not used rn
# op_list = [
#     'LUI',
#     'AUIPC',
#     'JAL',
#     'JALR',
#     'BEQ',
#     'BNE',
#     'BLT',
#     'BGE',
#     'BLTU',
#     'BGEU',
#     'LB',
#     'LH',
#     'LW',
#     'LBU',
#     'LHU',
#     'SB',
#     'SH',
#     'SW',
#     'ADDI',
#     'SLTI',
#     'SLTIU',
#     'XORI',
#     'ORI',
#     'ANDI',
#     'SLLI',
#     'SRLI',
#     'SRAI',
#     'ADD',
#     'SUB',
#     'SLL',
#     'SLT',
#     'SLTU',
#     'XOR',
#     'SRL',
#     'SRA',
#     'OR',
#     'AND',
#     # M-extension
#     'MUL',
#     'MULH',
#     'MULHSU',
#     'MULHU',
#     'DIV',
#     'DIVU',
#     'REM',
#     'REMU',
#     # F-extension
#     'FLW_S',
#     'FSW_S',
#     'FMADD_S',
#     'FMSUB_S',
#     'FNMSUB_S',
#     'FNMADD_S',
#     'FADD_S',
#     'FSUB_S',
#     'FMUL_S',
#     'FDIV_S',
#     'FSGNJ_S',
#     'FSGNJN_S',
#     'FSGNJX_S',
#     'FMIN_S',
#     'FMAX_S',
#     'FEQ_S',
#     'FLT_S',
#     'FLE_S',
#     'FSQRT_S',
#     'FCVT_W_S',
#     'FCVT_WU_S',
#     'FMV_X_W',
#     'FCLASS_S',
#     'FCVT_S_W',
#     'FCVT_S_WU',
#     'FMV_W_X',
# ]
# idx_to_op = {idx: op for idx, op in enumerate(op_list)}
# op_to_idx = {op: idx for idx, op in enumerate(op_list)}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    config = AlphaDevConfig()
    alphadev(config)

    # # test enum actions
    # actions = x86_enumerate_actions(3,3)
    # for action in actions:
    #     print(action)