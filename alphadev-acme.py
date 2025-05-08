"""
Main script for running AlphaDev with an ACME and reverb backend.
"""
from typing import NamedTuple, Dict, Tuple, Any, Callable, List, Generator

import sonnet as snn
import tensorflow as tf
import numpy as np

from acme.specs import EnvironmentSpec, Array, BoundedArray, DiscreteArray
from dm_env import Environment, TimeStep, StepType
# from acme.agents.tf.mcts
# from acme.agents.tf.mcts.agent_distributed import DistributedMCTS

from .alphadev import PredictionNet, RepresentationNet
from .tinyfive.multi_machine import multi_machine
from .mcts_distributed import DistributedMCTS # copied from github (not in the dm-acme package)

# #################
# Type definitions
# #################



# #################
# Config
# #################


# #################
# Action Spaces 
# TODO: make distributed
# #################

class ActionSpace:
    def __init__(self, actions: Dict[int, Any], asm: Dict[int, Any]):
        """Immutable action space."""
        self.actions = actions
        self.asm = asm
    
    def __getitem__(self, index):
        """
        Get the action at the given index.
        """
        return self.actions[index]
    
    def get_asm(self, index):
        """
        Get the action at the given index.
        """
        return self.asm[index]
    
    def __len__(self):
        """
        Get the number of actions in the action space.
        """
        return len(self.actions)


X0 = 0 # hard-wired zero register
X1 = 1 # reserved for comparison ops
REG_T = 10
MEM_T = 11
IMM_T = 12

def x86_to_riscv(opcode: str, operands: Tuple[int, int], mem_offset) -> Tuple[str, Callable[[int], Tuple[int, int]]]:
    """
    Convert an x86 (pseudo-) instruction to a RISC-V (pseudo-) instruction.
    """
    
    if opcode == "mv": # move between registers
        return [lambda _: ("ADD", (operands[0], X0, operands[1]),)]
    elif opcode == "lw": # load word from memory to register
        return [lambda _: ("LW", (operands[1], operands[0]-mem_offset, X0),)]
    # rd,imm,rs -- rd, rs(imm)
    elif opcode == "sw": # store word from register to memory
        return [lambda _: ("SW", (operands[0], operands[1]-mem_offset, X0),)] 
    # rs1,imm,rs2 -- rs1, rs2(imm)
    elif opcode == "cmp": # compare two registers
        return [lambda _: ("SUB", (X1, operands[0], operands[1]),)]
        # if A > B, then X1 > 0
        # if A < B, then X1 < 0
        # riscv has bge (>=) and blt (<) instructions
    elif opcode == "cmovg": # conditional move if greater than
        return [ # A > B <=> B < A -- 0 < X1
            lambda pc: ("BLT", (0, X1, pc+8),), 
            # skip next instruction if A < B
            lambda _ : ("ADD", (operands[1], X0, operands[0]),)
            # copy C to D
        ]
    elif opcode == "cmovle": # conditional move if less than or equal
        return [ # A <= B <=> B >= A -- 0 >= X1
            lambda pc: ("BGE", (0, X1, pc+8),),
            # skip next instruction if A > B
            lambda _ : ("ADD", (operands[1], X0, operands[0]),) 
            # copy E to F
        ]
    else:
        raise ValueError(f"Unknown opcode: {opcode}")

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

def x86_enumerate_actions(max_reg: int, max_mem: int) -> Dict[Tuple[str, Tuple[int, int]]]:
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
                for k in range(max_reg, max_reg + m): # offset the memory locations
                    # generate all combinations of registers and memory
                    for opcode in x86_signatures.keys():
                        yield from apply_opcode(opcode, (i, j, k))
    #   logger.debug("Enumerating actions for max_reg=%d, max_mem=%d", max_reg, max_mem)
    actions = set(enum_actions(max_reg, max_reg, max_mem))
    actions = {i: action for i, action in enumerate(actions)}
    #   logger.debug("Enumerated %d actions", len(actions))
    return actions


class ActionSpaceStorage:
    """Action Space Storage."""
    
    def __init__(self, max_reg: int, max_mem: int, name:str):
        self._max_reg = max_reg
        self._max_mem = max_mem
        self._name = name
    
    def get_space(self, state) -> ActionSpace:
        raise NotImplementedError()

    def get_mask(self, state, history:list=None) -> tf.Tensor:
        """
        Get the mask over the action space for the given state and history.
        
        Returns a boolean array over the action space, with True values indicating
        valid actions.
        """
        raise NotImplementedError()

class x86ActionSpaceStorage(ActionSpaceStorage):
    def __init__(self, max_reg: int, max_mem: int):
        self.max_reg = max_reg
        self.max_mem = max_mem
        self.actions: Dict[int, Dict[int, Tuple[str, Tuple[int,int]]]] =\
            x86_enumerate_actions(max_reg, max_mem)
        # pre-compute the assembly representation of the actions
        self.asm_actions = {
            i: x86_to_riscv(action[0], action[1], self.max_reg) for i, action in self.actions.items()
        }
        # there is a single action space for the given task
        self.action_space_cls = ActionSpace # these are still x86 instructions
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
        act_loc_table = tf.Variable(tf.zeros((self.max_reg + self.max_mem, action_space_size), dtype=tf.bool))
        # mask for register locations
        reg_locs = tf.concat([
            tf.ones(self.max_reg, dtype=tf.bool),
            tf.zeros(self.max_mem, dtype=tf.bool)
        ], axis=0)
        # mask for memory locations
        mem_locs = tf.concat([
            tf.zeros(self.max_reg, dtype=tf.bool),
            tf.ones(self.max_mem, dtype=tf.bool)
        ], axis=0)
        # boolean mask for actions that only use register locations
        reg_only_actions = tf.Variable(tf.zeros(action_space_size, dtype=tf.bool))
        # boolean mask for actions that read from memory locations
        mem_read_actions = tf.Variable(tf.zeros(action_space_size, dtype=tf.bool))
        # boolean mask for actions that write to memory locations
        mem_write_actions = tf.Variable(tf.zeros(action_space_size, dtype=tf.bool))
        
        for i, action in enumerate(self.actions.values()):
            # iterate over the x86 instructions currently under consideration
            x86_opcode, x86_operands = action # a tuple of (opcode, operands)
            signature = x86_signatures[x86_opcode]
            if signature == (REG_T, REG_T):
                indices = tf.constant([[op, i] for op in x86_operands], dtype=tf.int32)
                updates = tf.constant([True] * len(x86_operands), dtype=tf.bool)
                act_loc_table.scatter_nd_update(indices, updates)
                reg_only_actions.scatter_nd_update([[i]], [True])
            else: # action that accesses memory.
                mem_loc, reg_loc = x86_operands if signature == (MEM_T, REG_T) else reversed(x86_operands)
                mem_loc += self.max_reg # offset the memory locations
                act_loc_table.scatter_nd_update([[reg_loc, i], [mem_loc, i]], [True, True])
                if x86_opcode.startswith("l"): # load action
                    mem_read_actions.scatter_nd_update([[i]], [True])
                else:
                    mem_write_actions.scatter_nd_update([[i]], [True])
        
        assert tf.reduce_any(tf.logical_and(tf.logical_and(reg_only_actions, mem_read_actions), mem_write_actions)) == False, \
            "Action space was not partitioned correctly"
        assert tf.reduce_all(tf.logical_or(tf.logical_or(reg_only_actions, mem_read_actions), mem_write_actions)), \
            "Action space was not partitioned correctly"
        
        self.act_loc_table = act_loc_table.read_value()
        self.reg_locs = reg_locs
        self.mem_locs = mem_locs
        self.reg_only_actions = reg_only_actions.read_value()
        self.mem_read_actions = mem_read_actions.read_value()
        self.mem_write_actions = mem_write_actions.read_value()

    def get_mask(self, state, history: List[Tuple[str, Tuple[int, int]]]) -> tf.Tensor:
        """
        Get the mask over the action space for the given state and history.

        Returns a boolean tensor over the action space, with True values indicating
        valid actions.
        """
        _mems_read = set()
        _mems_written = set()

        def update_history(opcode, operands):
            # update the history with the action
            if opcode.startswith("L"):  # (MEM_T, _)
                _mems_read.add(operands[0])
            elif opcode.startswith("S"):  # (_, MEM_T)
                _mems_written.add(operands[1])

        if history is not None:
            # iterate over the history
            for action in history:
                update_history(*action)

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
        active_registers = state.register_mask  # shape E x R+(unused) # TODO: we might want to let the emulator know
        active_registers = tf.reduce_any(active_registers, axis=0)  # shape R
        active_memory = state.memory_mask  # shape E x M
        active_memory = tf.reduce_any(active_memory, axis=0)  # shape M

        assert active_registers.shape[0] == self.max_reg, \
            "active registers and max_reg do not match."
        assert active_memory.shape[0] == self.max_mem, \
            "active memory and max_mem do not match."

        # find windows of locations that are valid
        reg_window = tf.Variable(tf.zeros_like(reg_locs, dtype=tf.bool))
        last_reg = tf.cast(tf.argmax(tf.reverse(tf.cast(active_registers, tf.int64), axis=[0])), tf.int32)
        last_reg = -last_reg
        active_registers = tf.tensor_scatter_nd_update(active_registers, [[last_reg]], [True])
        reg_window.assign(tf.concat([active_registers, tf.zeros([self.max_reg + self.max_mem - self.max_reg], dtype=tf.bool)], axis=0))
        reg_window = reg_window.read_value()[:self.max_reg]

        # same for the memory locations
        mem_window = tf.Variable(tf.zeros_like(mem_locs, dtype=tf.bool))
        last_mem = tf.cast(tf.argmax(tf.reverse(tf.cast(active_memory, tf.int64), axis=[0])), tf.int32)
        last_mem = -last_mem
        active_memory = tf.tensor_scatter_nd_update(active_memory, [[last_mem]], [True])
        mem_window.assign(tf.concat([tf.zeros([self.max_reg], dtype=tf.bool), active_memory], axis=0))
        mem_window = mem_window.read_value()[self.max_reg:]

        # Identify register-only actions that access *any* location *outside* the active register window.
        # 1. Get access pattern for locations outside the window:
        inactive_loc_access = tf.boolean_mask(act_loc_table, ~reg_window)  # Shape (N_inactive_locs, N_actions)
        # 2. Check for each action if it accesses *any* inactive location:
        accesses_inactive_loc = tf.reduce_any(inactive_loc_access, axis=0)  # Shape (N_actions,)
        # A register-only action is valid if it is a register-only action
        # AND it does NOT access any inactive location.
        reg_only_mask = tf.logical_and(reg_only_actions, ~accesses_inactive_loc)  # Shape (N_actions,)

        # to enforce that only one read and one write is allowed at each memory location,
        # we also need to look at the history of the program
        # and mask out any actions that are illegal
        read_locs = tf.constant(list(_mems_read), dtype=tf.int32) + self.max_reg
        # create a mask of memory locations that are read
        mem_read_locs = tf.tensor_scatter_nd_update(
            tf.zeros_like(mem_locs, dtype=tf.bool), tf.expand_dims(read_locs, axis=1),
            tf.ones(tf.shape(read_locs), dtype=tf.bool))
        # subtract the read locations mask from the memory window
        mem_read_window = tf.logical_and(mem_window, ~mem_read_locs)
        # select all memory read actions, which operate within the memory window
        invalid_mem_read_loc = tf.boolean_mask(act_loc_table, ~(mem_read_window | reg_locs))
        accesses_invalid_mem = tf.reduce_any(invalid_mem_read_loc, axis=0)
        mem_read_mask = tf.logical_and(mem_read_actions, ~accesses_invalid_mem)

        # do the same for write actions
        write_locs = tf.constant(list(_mems_written), dtype=tf.int32) + self.max_reg
        mem_write_locs = tf.tensor_scatter_nd_update(
            tf.zeros_like(mem_locs, dtype=tf.bool), tf.expand_dims(write_locs, axis=1), tf.ones(tf.shape(write_locs), dtype=tf.bool))
        mem_write_window = tf.logical_and(mem_window, ~mem_write_locs)
        invalid_mem_write_loc = tf.boolean_mask(act_loc_table, ~(mem_write_window | reg_locs))
        accesses_invalid_mem = tf.reduce_any(invalid_mem_write_loc, axis=0)
        mem_write_mask = tf.logical_and(mem_write_actions, ~accesses_invalid_mem)

        assert reg_only_mask.shape[0] == len(self.actions), \
            "mask and action space size do not match."
        assert not tf.reduce_any(tf.logical_and(tf.logical_and(reg_only_mask, mem_read_mask), mem_write_mask)), \
            "masks do not partition the action space."
        assert tf.reduce_any(tf.logical_or(tf.logical_or(reg_only_mask, mem_read_mask), mem_write_mask)), \
            "no actions left in the action space."

        # combine the masks by taking their union
        return tf.logical_or(tf.logical_or(reg_only_mask, mem_read_mask), mem_write_mask)

    def get_space(self) -> ActionSpace:
        return self.action_space_cls(self.actions, self.asm_actions)

# #################
# Environment definition
# #################
class TaskSpec(NamedTuple):
    max_program_size: int
    num_inputs: int # number of input examples
    num_funcs: int # number of x86 instructions to consider
    num_regs: int # number of registers to consider
    num_mem: int # number of memory locations to consider. num_mem+num_regs = num_locations
    num_locations: int # memory + register locations to consider
    num_actions: int # number of actions in the action space
    correct_reward: float # reward for correct program
    correctness_reward_weight: float # weight for correctness reward
    latency_reward_weight: float # weight for latency reward
    latency_quantile: float # quantile for latency reward

class CPUState(NamedTuple):
    registers: tf.Tensor # num_inputs x num_regs array of register values
    active_registers: tf.Tensor # num_inputs x num_regs boolean array of active registers
    memory: tf.Tensor # num_inputs x num_mem array of memory locations
    active_memory: tf.Tensor # num_inputs x num_mem boolean array of active memory locations
    program: tf.Tensor # max_program_size x 1 array of progrram instructions.
    program_length: tf.Tensor  # scalar length of the program 
    program_counter: tf.Tensor # num_inputs x 1 array of program counters (in int32)

class IOExample(NamedTuple):
    inputs: tf.Tensor # num_inputs x <sequence_length>
    outputs: tf.Tensor # num_inputs x <num_mem>
    output_mask: tf.Tensor # num_inputs x <num_mem> boolean array masking irrelevant parts of the output

class AssemblyGame(Environment):
    def __init__(self, task_spec: TaskSpec, inputs: IOExample, action_space_storage: ActionSpaceStorage):
        """
        Create an AssemblyGame environment.
        Args:
            task_spec: Task specification for the environment.
            inputs: Inputs to the environment.
        """
        self._task_spec = task_spec
        self._inputs = inputs.inputs
        self._output_mask = inputs.output_mask
        self._outputs = tf.boolean_mask(inputs.outputs, self._output_mask)
        self._max_num_hits = tf.reduce_sum(self._output_mask)
        self._emulator = multi_machine(
            mem_size=task_spec.num_mem*4, # 4 bytes per memory location
            num_machines=task_spec.num_inputs,
            initial_state=self._inputs
        )
    
    def _eval_output(self, output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        masked_output = tf.boolean_mask(output, self._output_mask)
        hits = tf.equal(masked_output, self._outputs)
        num_hits = tf.reduce_sum(hits)
        all_hits = tf.equal(num_hits, self._max_num_hits)
        return all_hits, num_hits
    
    def _convert_instruction(self, instruction: tf.Tensor) -> tf.Tensor:
        """
        Instruction is a scalar int32 value. We translate it to an assembly pseudo-op.
        in RISC-V.
        """
        return self._asm_lookup[instruction][1]
    
    def reset(self) -> TimeStep:
        pass
    def step(self, action) -> TimeStep:
        pass
    def reward_spec(self):
        pass
    def discount_spec(self):
        pass
    def observation_spec(self):
        pass
    def action_spec(self):
        pass
    def close(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    


# #################
# network definition
# #################

class AlphaDevNetwork(snn.Module):
    # NOTE: this won't work :( need to convert from jax/haiku to tf/sonnet
    prediction_net = PredictionNet
    representation_net = RepresentationNet
    

# #################
# model definition 
# #################
# wrapper for AlphaGame

# predictor net from JEPA

# #################
# factory functions
# #################

def environment_factory(): # no args
    pass

def network_factory(env_spec: DiscreteArray): # env_spec
    pass

def model_factory(env_spec: EnvironmentSpec): # env_spec
    pass

# #################
# Agent definition
# #################

# TODO: replace
from .alphadev import AlphaDevConfig
config = AlphaDevConfig()

agent = DistributedMCTS(
    environment_factory=environment_factory, # AlphaGame environment
    network_factory=network_factory,         # AlphaDev network
    model_factory=model_factory,             # Either a learned model or a wrapper around the environment
    num_actors=config.num_actors, # number of parallel actors
    num_simulations=config.num_simulations, # number of rollouts per action
    batch_size=config.batch_size,  # batch size used by the learner.
    prefetch_size=config.prefetch_size,  # parameter passed to make_reverb_dataset
    target_update_period=config.target_update_period, # NOTE: not used
    samples_per_insert=config.samples_per_insert, # to balance the replay buffer
    min_replay_size=config.min_replay_size, # used as a limiter for the replay buffer
    max_replay_size=config.max_replay_size, # maximum size of the replay buffer
    importance_sampling_exponent=config.importance_sampling_exponent, # not used
    priority_exponent=config.priority_exponent, # not used
    n_step=config.n_step, # how many steps to buffer before adding to the replay buffer
    learning_rate=config.learning_rate, # learning rate for the learner
    discount=config.discount, # discount factor for the environment
    environment_spec=config.env_spec, # defines the environment
    save_logs=config.save_logs, # whether to save logs or not
)
