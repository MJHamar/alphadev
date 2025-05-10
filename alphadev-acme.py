"""
Main script for running AlphaDev with an ACME and reverb backend.
"""
from typing import NamedTuple, Dict, Tuple, Any, Callable, List, Generator, Sequence, Mapping, Optional, Union
import functools
import itertools

import sonnet as snn
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import ml_collections

from acme.specs import EnvironmentSpec, make_environment_spec, Array, BoundedArray, DiscreteArray
from acme.agents.tf.mcts import models
from acme.agents.tf.mcts import types
from dm_env import Environment, TimeStep, StepType
# from acme.agents.tf.mcts
# from acme.agents.tf.mcts.agent_distributed import DistributedMCTS

from .alphadev import PredictionNet, RepresentationNet
from .tinyfive.multi_machine import multi_machine
from .mcts_distributed import DistributedMCTS # copied from github (not in the dm-acme package)
from .distribution import DistributionSupport

# #################
# Type definitions
# #################

class IOExample(NamedTuple):
    inputs: tf.Tensor # num_inputs x <sequence_length>
    outputs: tf.Tensor # num_inputs x <num_mem>
    output_mask: tf.Tensor # num_inputs x <num_mem> boolean array masking irrelevant parts of the output

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
    inputs: IOExample # input examples for the task
    return_loss_components: bool # whether to return the loss components for the value function (for categorical value loss)

class CPUState(NamedTuple):
    registers: tf.Tensor # num_inputs x num_regs array of register values
    active_registers: tf.Tensor # num_inputs x num_regs boolean array of active registers
    memory: tf.Tensor # num_inputs x num_mem array of memory locations
    active_memory: tf.Tensor # num_inputs x num_mem boolean array of active memory locations
    program: tf.Tensor # max_program_size x 1 array of progrram instructions.
    program_length: tf.Tensor  # scalar length of the program 
    program_counter: tf.Tensor # num_inputs x 1 array of program counters (in int32)

class Program(NamedTuple):
    npy_program: tf.Tensor # <max_program_size> x 3 (opcode, op1, op2)
    asm_program: List[Callable[[int], Any]] # list of pseudo-asm instructions
    
    def __len__(self):
        return len(self.asm_program)

# #################
# Input geneerators
# #################

def generate_sort_inputs(
    items_to_sort: int, max_len:int, num_samples: int=None, rnd_key:int=42) -> IOExample:
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
                expected = np.array(sorted(perm))
                io_list.append((np.array(perm), expected))
        for i in range(0, items_to_sort+1):
            relation = [1]
            mask = i; j=0
            while j < items_to_sort - 1: # no idea how to express this more pythonic
                j += 1
                relation.append(relation[-1] if mask % 2 == 0 else relation[-1] + 1)
                mask //= 2
            add_all_permutations(relation)

    def remap_input(inp: np.ndarray, out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mapping = {}
        prev = 0
        for o in out.tolist():
            if o not in mapping:
                mapping[o] = np.random.randint(prev+1, prev+3) # don't blow it up. 
                prev = mapping[o]
        out = np.array([mapping[o] for o in out.tolist()])
        inp = np.array([mapping[i] for i in inp.tolist()])
        return inp, out

    generate_testcases(items_to_sort)
    i_list = np.stack([i for i, _ in io_list])
    # padded outputs
    o_list = np.stack([
        np.pad(o, (0, max_len-len(o)), 'constant', constant_values=0)
        for _, o in io_list])
    o_mask = np.stack([
        [1 for _ in range(len(o))] +\
        [0 for _ in range(max_len-len(o))]
        for _, o in io_list])
    assert o_list.shape[1] == o_mask.shape[1] == max_len, \
        f"Expected output shape {max_len}, got {o_list.shape[1]}"
    # remove duplicates
    _, uidx = np.unique(i_list, axis=0, return_index=True) # remove duplicates
    i_list = i_list[uidx, :] # shape F(items_to_sort) x items_to_sort
    o_list = o_list[uidx, :] # shape F(items_to_sort) x max_len
    o_mask = o_mask[uidx, :] # shape F(items_to_sort) x max_len
    
    
    #   logger.debug("Generated %d test cases", len(io_list))
    # shuffle the permutations. if num_samples > len(permutations), we set
    # inputs = permutations + num_samples - len(permutations) random samples from permutations
    # otherwise, we set inputs = random.sample(permutations, num_samples)
    new_indices = np.random.permutation(len(i_list))
    if num_samples is None:
        pass # select all elements
    elif num_samples > i_list.shape[0]:
        new_indices = np.concatenate([new_indices, new_indices[:num_samples - i_list.shape[0]]])
    else:
        new_indices = new_indices[:num_samples]
    
    i_list = i_list[new_indices]
    o_list = o_list[new_indices]
    o_mask = o_mask[new_indices]
    # add some noise to the inputs
    for i in range(i_list.shape[0]):
        i_list[i], o_list[i] = remap_input(i_list[i], o_list[i])
    
    # logger.debug("generate_sort_inputs: i_list shape %s", i_list.shape)
    # logger.debug("generate_sort_inputs: o_list shape %s", o_list.shape)
    # logger.debug("generate_sort_inputs: o_mask shape %s", o_mask.shape)
    
    return IOExample(
        inputs=i_list,
        outputs=o_list,
        output_mask=o_mask,
    )

# #################
# Config
# #################


class AlphaDevConfig(object):
    """AlphaDev configuration."""

    @staticmethod
    def visit_softmax_temperature_fn(steps): 
        return 1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25
    
    def __init__(
        self,
    ):
        self.experiment_name = 'AlphaDev-test'
        ### Self-Play
        self.num_actors = 1 # TPU actors
        # pylint: disable-next=g-long-lambda
        self.max_moves = np.inf
        self.num_simulations = 5
        self.discount = 1.0

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.03
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

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
        self.hparams.categorical_value_loss = False # wheether to treat the value functions as a distribution

        ### Training
        self.training_steps = 1000 #int(1000e3)
        self.checkpoint_interval = 50 # used to be 500
        self.target_network_interval = 10 # used to be 100
        self.window_size = int(1e3) # 1e6 originally
        self.batch_size = 8
        self.td_steps = 5
        self.lr_init = 2e-4
        self.momentum = 0.9


        # Environment: spec of the Variable Sort 3 task
        self.task_spec = TaskSpec(
            max_program_size=100,
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
            inputs=generate_sort_inputs(
                items_to_sort=3,
                max_len=self.task_spec.num_mem,
                num_samples=self.task_spec.num_inputs
            ),
            observe_reward_components=self.hparams.categorical_value_loss,
        )
        
        #   logger.debug("Inputs: %s", self.inputs)
        assert self.task_spec.num_inputs == self.inputs.inputs.shape[0],\
            f"Expected {self.task_spec.num_inputs} inputs, got {self.inputs.inputs.shape[0]}"

    @classmethod
    def from_yaml(cls, path) -> 'AlphaDevConfig':
        """Create a config from a ml_collections.ConfigDict."""
        pass

# #################
# Action Spaces 
# TODO: make distributed
# #################

class ActionSpace:
    def __init__(self, actions: Dict[int, Any], asm: Dict[int, Any], nump:Dict[int, Any]):
        """Immutable action space."""
        self.actions = actions
        self.asm = asm
        self.np = nump
    
    def get(self, index):
        """
        Get the action at the given index.
        
        Returns a tuple of the form (opcode, operands).
        We can use this for printing the action.
        """
        return self.actions[index]
    
    def get_asm(self, index):
        """
        Get the action at the given index.
        """
        return self.asm[index]
    
    def get_np(self, index):
        """
        Get the action at the given index.
        
        Returns a numpy array of the form [opcode, reg1, reg2].
        """
        return self.np[index]
    
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

x86_opcode2int = {
    k: i for i, k in enumerate(x86_signatures.keys())
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

    def npy_to_asm(self, npy_program: tf.Tensor) -> List[Callable[[int], Any]]:
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
        self.np_actions = {
            i: np.array([x86_opcode2int[action[0]], action[1][0], action[1][1]])
            for i, action in self.actions.items()
        }
        self._npy_reversed = {
            tuple(v): k for k, v in self.np_actions.items()
        }
        # there is a single action space for the given task
        self.action_space_cls = ActionSpace # these are still x86 instructions
        # TODO: make sure we don't flood the memory with this
        self.masks = {}
        # for pruning the action space (one read and one write per memory location)
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
        
        assert not tf.reduce_any(tf.logical_and(reg_only_actions, mem_read_actions)), \
            "Action in both reg_only and mem_read"
        assert not tf.reduce_any(tf.logical_and(reg_only_actions, mem_write_actions)), \
            "Action in both reg_only and mem_write"
        assert not tf.reduce_any(tf.logical_and(mem_read_actions, mem_write_actions)), \
            "Action in both mem_read and mem_write"
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
        return self.action_space_cls(self.actions, self.asm_actions, self.np_actions)

    def npy_to_asm(self, npy_program):
        """
        Convert a numpy program to a list of assembly instructions.
        
        Args:
            npy_program: numpy array of shape (max_program_size, 3) containing
            the program instructions.
        
        Returns:
            A list of assembly instructions.
        """
        # convert the numpy program to a list of assembly instructions
        asm_program = []
        for insn in npy_program:
            asm_insn = self._npy_reversed.get(tuple(insn))
            asm_program.append(asm_insn)
        return asm_program

# #################
# Environment definition
# #################

class AssemblyGame(Environment):
    def __init__(self, task_spec: TaskSpec):
        """
        Create an AssemblyGame environment.
        Args:
            task_spec: Task specification for the environment.
            inputs: Inputs to the environment.
        """
        self._task_spec = task_spec
        self._inputs = task_spec.inputs.inputs
        self._output_mask = task_spec.inputs.output_mask
        self._outputs = tf.boolean_mask(task_spec.inputs.outputs, self._output_mask)
        self._max_num_hits = tf.reduce_sum(self._output_mask)
        
        # whether to return the correctness and latency components of the reward
        # in the TimeSteps
        self._observe_reward_components = task_spec.observe_reward_components
        
        self._emulator = multi_machine(
            mem_size=task_spec.num_mem*4, # 4 bytes per memory location
            num_machines=task_spec.num_inputs,
            initial_state=self._inputs
        )
        # TODO: make this distributed
        self._action_space_storage = x86ActionSpaceStorage(
            max_reg=task_spec.num_regs,
            max_mem=task_spec.num_mem
        )
        self.reset()

    def _reset_program(self):
        self._program = Program(
            npy_program=tf.Variable(
                initial_value=tf.zeros((self._task_spec.max_program_size, 3), dtype=tf.int32),
                trainable=False,
                name="program"),
            asm_program=[],
        )
    
    def _eval_output(self, output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        masked_output = tf.boolean_mask(output, self._output_mask)
        hits = tf.equal(masked_output, self._outputs)
        num_hits = tf.reduce_sum(hits)
        all_hits = tf.equal(num_hits, self._max_num_hits)
        return all_hits, num_hits
    
    def _eval_latency(self) -> tf.Tensor:
        """Returns a scalar latency for the program."""
        latencies = tf.constant([
            self._emulator.measure_latency(self._program.asm_program)
            for _ in range(self._task_spec.num_inputs)], dtype=tf.float32
        )
        return latencies
    
    def _compute_reward(self, include_latency: float) -> float:
        # compute the reward based on the latency and correctness
        correctness_reward = self._task_spec.correctness_reward_weight * (
            self._num_hits - self._num_hits # NOTE: num_hits here **is** from the previous iteration.
        )
        correctness_reward += self._task_spec.correct_reward * self._is_correct
        # update the previous correct items
        latency_reward = 0.0
        if include_latency: # cannot be <0 btw
            latencies = self._eval_latency()
            latency_reward = tfp.stats(
                latencies, self._task_spec.latency_quantile * 100
            ) * self._task_spec.latency_reward_weight
        reward = correctness_reward + latency_reward
        
        return reward, latency_reward, correctness_reward
    
    def _make_observation(self) -> CPUState:
        # get the current state of the CPU
        return CPUState(
            registers=tf.constant(self._emulator.registers),
            active_registers=tf.constant(self._emulator.register_mask),
            memory=tf.constant(self._emulator.memory),
            active_memory=tf.constant(self._emulator.memory_mask),
            program=tf.constant(self.program.npy_program),
            program_length=tf.constant(len(self._program), dtype=tf.int32),
            program_counter=tf.constant(self._emulator.program_counter, dtype=tf.int32)
        )
    
    def _check_invalid(self) -> bool:
        # either too long or the emulator is in an invalid state
        return len(self._program) > self._task_spec.max_program_size or \
            False # TODO: self._emulator.invalid()
    
    def _update_state(self):
        # first we make an observation
        observation = self._make_observation()
        # then we check if the program is correct
        self._is_invalid = self._check_invalid()
        if self._is_invalid:
            self._is_correct = False; self._num_hits = 0
        else:
            self._is_correct, self._num_hits = self._eval_output(self._emulator.memory)
        # terminality check
        is_terminal = self._is_correct or self._is_invalid
        
        # we can now compute the reward
        reward, latency, correctness = self._compute_reward(include_latency=is_terminal)
        
        ts = TimeStep(
            step_type=StepType.MID if not is_terminal else StepType.LAST,
            # too many components in acme hard-code the structure of TimeStep, and not
            # everything supports reward to be a dictionary, so we concatenate
            # the reward components into a single tensor
            reward=reward if not self._observe_reward_components else tf.constant([reward, correctness, latency], dtype=tf.float32),
            discount=1.0, # NOTE: not sure what discount here means.
            observation=observation,
            # skip latency and correctness
        )
        self._last_ts = ts
        return ts
    
    def reset(self, state: Union[TimeStep, CPUState, None]=None) -> TimeStep:
        # deletes the program and resets the
        # CPU state to the original inputs
        if state is None:
            self._emulator.reset_state()
            self._reset_program()
        else:
            # decode the program and execute it.
            # basically the same overhead as copying everything
            # but copying is also not fully possible
            # and program numpy -> asm is unavoidable anyway
            if isinstance(state, TimeStep):
                ts_program = state.observation['program']
            else:
                ts_program = state.program
            
            # either B x num_inputs x 3 or no batch dimension
            if len(ts_program.shape) > 2:
                # we need to remove the batch dimension
                assert ts_program.shape[0] == 1, "Batch dimension is not 1, resetting is ambigouous."
                ts_program = tf.squeeze(ts_program, axis=0)

            # convert the numpy program to a list of assembly instructions
            asm_program = self._action_space_storage.npy_to_asm(ts_program)
            self._program = Program(
                npy_program=tf.Variable(ts_program, trainable=False, name="program"),
                asm_program=asm_program,
            )
            self._emulator.reset_state()
            # execute the program
            self._emulator.exe(program=self._program.asm_program)
            # update the state
        return self._update_state()
    
    def step(self, action:int) -> TimeStep:
        action_space = self._action_space_storage.get_space()
        # append the action to the program
        action_np = action_space.get_np(action)
        action_asm = action_space.get_asm(action)
        
        self._program = Program(
            npy_program=tf.tensor_scatter_nd_update(
                self._program.npy_program, [[len(self._program)]], [action_np]
            ),
            asm_program=self._program.asm_program + [action_asm] # preserves immutability
        )
        # reset the emulator
        self._emulator.reset_state()
        # execute the program
        self._emulator.exe(program=self._program.asm_program)
        # update observation and cached values
        # and return the updated timestep
        return self._update_state()
    
    def reward_spec(self):
        return Array(shape=(), dtype=tf.float32)
    def discount_spec(self):
        return Array(shape=(), dtype=tf.float32)
    def observation_spec(self):
        return CPUState(
            registers=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_regs), dtype=tf.float32),
            active_registers=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_regs), dtype=tf.bool),
            memory=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_mem), dtype=tf.float32),
            active_memory=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_mem), dtype=tf.bool),
            program=Array(shape=(self._task_spec.max_program_size, 3), dtype=tf.int32),
            program_length=Array(shape=(), dtype=tf.int32),
            program_counter=Array(shape=(self._task_spec.num_inputs,), dtype=tf.int32)
        )#._asdict()
    def action_spec(self):
        # TODO: this won't work for dynamic action spaces
        return DiscreteArray(num_values=len(self._action_space_storage.actions))
    def close(self):
        del self._emulator
        del self._program
        del self._last_ts

    def __enter__(self):
        return super().__enter__()
    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

    def copy(self):
        new_game = object.__new__(self.__class__)
        # copy the immutable parts of the state
        new_game._task_spec = self._task_spec
        new_game._inputs = self._inputs
        new_game._output_mask = self._output_mask
        new_game._outputs = self._outputs
        new_game._max_num_hits = self._max_num_hits
        new_game._action_space_storage = self._action_space_storage
        # copy the two mutable parts of the state
        new_game._emulator = self._emulator.copy()
        new_game._program = self._program.copy()
        # these are also 'immmutable'
        new_game._last_ts = self._last_ts
        new_game._is_correct = self._is_correct
        new_game._num_hits = self._num_hits
        new_game._is_invalid = self._is_invalid
        return new_game

# #################
# network definition
# #################


class MultiQueryAttentionBlock(snn.Module):
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
        Tensoflow implementation from the paper:
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
        
        # logger.debug("MQAB: logits einsum Q (bnhk) %s, K (bmk) %s -> bhnm", Q.shape, K.shape)
        logits = tf.einsum("bnhk,bmk->bhnm", Q, K) # B x N x H x M
        # logger.debug("MQAB: logits shape (bhnm) %s", logits.shape)
        weights = tf.nn.softmax(logits) # NOTE: no causal masking, this is an encoder block
        # logger.debug("MQAB: weights shape (bhnm) %s", weights.shape)
        if self.attention_dropout: # boolean
            # logger.debug("MQAB: applying attention dropout %s", self.attention_dropout)
            weights = snn.Dropout(self.attention_dropout, name='attn_dropout')(weights)
        # logger.debug("MQAB: output projection einsum weights (bhnm) %s, V (bmv) %s -> bhnv", weights.shape, V.shape)
        O = tf.einsum("bhnm,bmv->bhnv", weights, V) # B x N x H x V
        # logger.debug("MQAB: O shape (bhnv) %s", O.shape)
        B, _, N, _ = O.shape # B x H x N x V
        # reshape O to B x N x (H*V)
        O = O.reshape((B, N, -1)) # B x N x (H*V)
        # logger.debug("MQAB: O reshaped to %s", O.shape)
        # apply the output projection
        # logger.debug("MQAB: aggregate head projection bn[h*v] %s to bnd %s ", O.shape, inputs.shape)
        Y = self._linear_projection(O, 1, inputs.shape[-1], name='P_o') # B x N x V
        # logger.debug("MQAB: output shape (bnd) %s", Y.shape)
        assert Y.shape == inputs.shape,\
            f"Output shape {Y.shape} does not match input shape {inputs.shape}."
        
        return Y # B x N x D
    
    def _linear_projection(
        self,
        x: tf.Tensor, # [B, N, D] (batch, seqence length, embedding dim)
        num_heads: int,
        head_size: int,
        name: str | None = None,
    ) -> tf.Tensor:
        """Copy-paste from hk.MultiHeadAttention."""
        y = tf.Linear(num_heads * head_size, name=name)(x) # proj mat. D x (H*K) 
        # y should be [B, N, H*K]
        assert y.ndim == 3, f"y should be 3D, but got {y.ndim}D"
        assert y.shape[-1] == num_heads * head_size, f"y should be of shape [B, N, H*K], but got {y.shape}"
        *leading_dims, _ = x.shape # [B, N, D]
        # split last dimension (H*K) into H, K
        new_shape = (*leading_dims, num_heads, head_size) if num_heads > 1 else (*leading_dims, head_size)
        # logger.debug("linear_projection, reshaping y from %s to %s", y.shape, new_shape)
        return y.reshape(new_shape) # [B, N, H, K] or [B, N, K]

    def sinusoid_position_encoding(seq_size, feat_size):
        """Compute sinusoid absolute position encodings, 
        given a sequence size and feature dimensionality"""
        # SOURCE: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
        pe = tf.zeros((seq_size, feat_size))
        position = tf.cast(tf.range(0, seq_size), dtype=tf.float32)[:,None]
        div_term = tf.exp(tf.range(0, feat_size, 2) * (-tf.math.log(10000.0) / feat_size))
        pe = tf.tensor_scatter_nd_update(pe, indices=tf.stack([tf.range(seq_size), tf.zeros(seq_size, dtype=tf.int32)], axis=1), updates=tf.sin(position * div_term))
        pe = tf.tensor_scatter_nd_update(pe, indices=tf.stack([tf.range(seq_size), tf.ones(seq_size, dtype=tf.int32)], axis=1), updates=tf.cos(position * div_term))
        pe = tf.reshape(pe, (1, seq_size, feat_size))
        return pe

class ResBlockV2(snn.Module):
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
            self.proj_conv = snn.Conv1D(
                output_channels=channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="SAME",
                name="shortcut_conv")

        channel_div = 4 if bottleneck else 1
        conv_0 = snn.Conv1D(
            output_channels=channels // channel_div,
            kernel_shape=1 if bottleneck else 3,
            stride=1 if bottleneck else stride,
            with_bias=False,
            padding="SAME",
            name="conv_0")

        ln_0 = snn.LayerNorm(name="LayerNorm_0", **ln_config)

        conv_1 = snn.Conv1D(
            output_channels=channels // channel_div,
            kernel_shape=3,
            stride=stride if bottleneck else 1,
            with_bias=False,
            padding="SAME",
            name="conv_1")

        ln_1 = snn.LayerNorm(name="LayerNorm_1", **ln_config)
        layers = ((conv_0, ln_0), (conv_1, ln_1))

        if bottleneck:
            conv_2 = snn.Conv1D(
                output_channels=channels,
                kernel_shape=1,
                stride=1,
                with_bias=False,
                padding="SAME",
                name="conv_2")

            # NOTE: Some implementations of ResNet50 v2 suggest initializing
            # gamma/scale here to zeros.
            ln_2 = snn.LayerNorm(name="LayerNorm_2", **ln_config)
            layers = layers + ((conv_2, ln_2),)

        self.layers = layers

    def __call__(self, inputs):
        # FIXME: figure out what to do with the is_training and test_local_stats
        # logger.debug("ResBlockV2: inputs shape %s", inputs.shape)
        x = shortcut = inputs

        for i, (conv_i, ln_i) in enumerate(self.layers):
            x = ln_i(x)
            x = tf.nn.relu(x)
        if i == 0 and self.use_projection:
            shortcut = self.proj_conv(x)
        x = conv_i(x)

        return x + shortcut


def int2bin(integers_array: tf.Tensor) -> tf.Tensor:
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
    flat_arr = tf.reshape(tf.cast(integers_array, dtype=tf.int32), (-1, 1))
    # bin_mask = jnp.tile(2 ** jnp.arange(32), (flat_arr.shape[0], 1))
    bin_mask = tf.tile(tf.reshape(tf.pow(2, tf.range(32)), (1,32)), [tf.shape(flat_arr)[0], 1])
    masked = (flat_arr & bin_mask) != 0
    return tf.reshape(masked, (*integers_array.shape[:-1], integers_array.shape[-1] * 32))

def bin2int(binary_array: tf.Tensor) -> tf.Tensor:
    """Reverses operation of int2bin."""
    # reshape the binary array to be of shape (S1, S2, ..., SN, 32)
    # i.e. all 32-long binary sequences are separated
    u_binary_array = tf.reshape(binary_array, (*binary_array.shape[:-1], binary_array.shape[-1] // 32, 32)
    )
    # calculate the exponents for each bit
    exponents = tf.pow(2, tf.range(32))
    result = tf.tensordot(tf.cast(u_binary_array, tf.int32), exponents, axes=1)
    return tf.cast(result, dtype=tf.int32)

class RepresentationNet(snn.Module):
    """
    Implemementation of the RepresentationNet based on the AlphaDev pseudocode
    
    https://github.com/google-deepmind/alphadev
    """

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

    def __call__(self, inputs: CPUState):
        # logger.debug("representation_net program shape %s", inputs['program'].shape)
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
            raise NotImplementedError(
                'permutation embedding is not implemented and will not be. keeping for completeness.')
        # aggregate the locations and the program to produce a single output vector
        return self.aggregate_locations_program(
            locations_encoding, permutation_embedding, program_encoding, batch_size
        )

    def _encode_program(self, inputs: CPUState, batch_size):
        # logger.debug("encode_program shape %s", inputs['program'].shape)
        program = inputs.program
        max_program_size = inputs.program.shape[1] # TODO: this might not be a constant
        program_length = tf.cast(inputs.program_length, tf.int32)
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
        locations_embedder = snn.Sequential(
            [
                # input is embedding_dim size, because we already encoded in either one-hot or binary
                snn.Linear(self._embedding_dim),
                snn.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                tf.nn.relu,
                snn.Linear(self._embedding_dim),
            ],
            name='per_locations_embedder',
        )

        # locations_encoding.shape == [B, P, D] so map embedder across locations to
        # share weights
        # logger.debug("aggregate_locations_program: locations_encoding shape %s", locations_encoding.shape)
        
        locations_embedding = tf.vectorized_map(locations_embedder, locations_encoding)
        # logger.debug("aggregate_locations_program: locations_embedding shape %s", locations_embedding.shape)

        # broadcast the program encoding for each example.
        # this way, it matches the size of the observations.
        # logger.debug("aggregate_locations_program: program_encoding shape %s", program_encoding.shape)
        program_encoded_repeat = self.repeat_program_encoding(
            program_encoding, batch_size
        )
        # logger.debug("aggregate_locations_program: program_encoded_repeat shape %s", program_encoded_repeat.shape)

        grouped_representation = tf.concat( # concat the CPU state and the program.
            [locations_embedding, program_encoded_repeat], axis=-1
        )
        # logger.debug("aggregate_locations_program: grouped_representation shape %s", grouped_representation.shape)

        return self.apply_joint_embedder(grouped_representation, batch_size)

    def repeat_program_encoding(self, program_encoding, batch_size):
        # logger.debug("aggregate_locations_program: repeat_program_encoding pre shape %s", program_encoding.shape)
        program_encoding = tf.broadcast_to(
            program_encoding,
            [batch_size, self._task_spec.num_inputs, program_encoding.shape[-1]],
        )
        # logger.debug("aggregate_locations_program: repeat_program_encoding post shape %s", program_encoding.shape)
        return program_encoding

    def apply_joint_embedder(self, grouped_representation, batch_size):
        all_locations_net = snn.Sequential(
            [
                snn.Linear(self._embedding_dim),
                snn.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                tf.nn.relu,
                snn.Linear(self._embedding_dim),
            ],
            name='per_element_embedder',
        )
        joint_locations_net = snn.Sequential(
            [
                snn.Linear(self._embedding_dim),
                snn.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                tf.nn.relu,
                snn.Linear(self._embedding_dim),
            ],
            name='joint_embedder',
        )
        joint_resnet = [
            ResBlockV2(self._embedding_dim, name=f'joint_resblock_{i}')
            for i in range(self._hparams.representation.repr_net_res_blocks)
        ]

        assert grouped_representation[:2] == (batch_size, self._task_spec.num_inputs), \
            f"grouped_representation shape {grouped_representation.shape} does not match expected shape {(batch_size, self._task_spec.num_inputs)}"
        # logger.debug("apply_joint_embedder grouped_rep shape %s", grouped_representation.shape)
        # apply MLP to the combined program and locations embedding
        permutations_encoded = all_locations_net(grouped_representation)
        # logger.debug("apply_joint_embedder permutations_encoded shape %s", permutations_encoded.shape)
        # Combine all permutations into a single vector using a ResNetV2
        joint_encoding = joint_locations_net(tf.reduce_mean(permutations_encoded, axis=1))
        # logger.debug("apply_joint_embedder joint_encoding shape %s", joint_encoding.shape)
        for net in joint_resnet:
            joint_encoding = net(joint_encoding)
        return joint_encoding

    def make_program_onehot(self, program, batch_size, max_program_size):
        # logger.debug("make_program_onehot shape %s", program.shape)
        func = program[:, :, 0] # the opcode -- int
        arg1 = program[:, :, 1] # the first operand -- int 
        arg2 = program[:, :, 2] # the second operand -- int
        func_onehot = tf.one_hot(func, self._task_spec.num_funcs)
        arg1_onehot = tf.one_hot(arg1, self._task_spec.num_locations)
        arg2_onehot = tf.one_hot(arg2, self._task_spec.num_locations)
        # logger.debug("func %s, arg1 %s, arg2 %s", func_onehot.shape, arg1_onehot.shape, arg2_onehot.shape)
        program_onehot = tf.concat(
            [func_onehot, arg1_onehot, arg2_onehot], axis=-1
        )
        assert program_onehot.shape[:2] == (batch_size, max_program_size, None), \
            f"program_onehot shape {program_onehot.shape} does not match expected shape {(batch_size, max_program_size, None)}"
        # logger.debug("program_onehot shape %s", program_onehot.shape)
        return program_onehot

    def pad_program_encoding(
        self, program_encoding, batch_size, program_length, max_program_size
    ):
        """Pads the program encoding to account for state-action stagger."""
        # logger.debug("pad_program_encoding shape %s", program_encoding.shape)
        assert program_encoding.shape[:2] == (batch_size, max_program_size),\
            f"program_encoding shape {program_encoding.shape} does not match expected shape {(batch_size, max_program_size)}"
        assert program_length.shape[:2] == (batch_size, self._task_spec.num_inputs),\
            f"program_length shape {program_length.shape} does not match expected shape {(batch_size, self._task_spec.num_inputs)}"

        empty_program_output = tf.zeros(
            [batch_size, program_encoding.shape[-1]],
        )
        program_encoding = tf.concat(
            [empty_program_output[:, None, :], program_encoding], axis=1
        )

        program_length_onehot = tf.one_hot(program_length, max_program_size + 1)
        # logger.debug("pad_program_encoding pre program_length_onehot shape %s", program_length_onehot.shape)
        # logger.debug("pad_program_encoding pre program_encoding shape %s", program_encoding.shape)
        # two cases here:
        # - program length is a batch of scalars corr. to the program length
        # - program length is a batch of vectors (of len num_inputs) corr. to the state of the program counters
        if len(program_length_onehot.shape) == 3:
            program_encoding = tf.einsum(
                'bnd,bNn->bNd', program_encoding, program_length_onehot
            )
        else:
            program_encoding = tf.einsum(
                'bnd,bn->bd', program_encoding, program_length_onehot
            )
        # logger.debug("pad_program_encoding post program_encoding shape %s", program_encoding.shape)

        return program_encoding

    def apply_program_mlp_embedder(self, program_encoding):
        # input: B x P x (num_funcs + 2 * num_locations)
        # output: B x P x embedding_dim
        # P is the program length
        # logger.debug("apply_program_mlp_embedder program shape %s, embedding_dim %s", program_encoding.shape, self._embedding_dim)
        program_embedder = snn.Sequential(
            [
                snn.Linear(self._embedding_dim), # (nF + 2*nL) x D -- input size is decided automatically
                snn.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                tf.nn.relu,
                snn.Linear(self._embedding_dim), # D x D
            ],
            name='per_instruction_program_embedder',
        )
        # logger.debug("program.shape %s -->", program_encoding.shape)
        # logger.debug("  program_embedder %s", program_embedder)
        program_encoding = program_embedder(program_encoding)
        # logger.debug("program_encoding shape %s <--", program_encoding.shape)
        return program_encoding

    def apply_program_attention_embedder(self, program_encoding):
        # logger.debug("apply_program_attention_embedder program shape %s", program_encoding.shape)
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

        position_encodings = tf.broadcast_to(
            MultiQueryAttentionBlock.sinusoid_position_encoding(
                seq_size, feat_size
            ),
            program_encoding.shape,
        )
        program_encoding = tf.add(program_encoding, position_encodings)

        for e in attention_encoders:
            # logger.debug("apply_program_attention_embedder layer %s", e.name)
            program_encoding = e(program_encoding, encoded_state=None)
        # logger.debug("apply_program_attention_embedder post MQAB %s", program_encoding.shape)

        return program_encoding

    def _make_locations_encoding_onehot(self, inputs: CPUState, batch_size):
        """Creates location encoding using onehot representation."""
        # logger.debug("make_locations_encoding_onehot shapes %s", str({k:v.shape for k,v in inputs.items()}))
        memory = inputs.memory # B x E x M (batch, num_inputs, memory size)
        registers = inputs.registers # B x E x R (batch, num_inputs, register size)
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
        locations = tf.concat([registers, memory], axis=-1) # B x E x (R + M)
        # logger.debug("locations shape %s", locations.shape)
        # to support inputs with sequences of states, we conditinally transpose
        # the locations tensor to have the shape [B, P, H, D]
        if len(locations.shape) == 4:
            # in this case, locations is [B, H, P, D]
            # and we need to transpose it to [B, P, H, D]
            locations = tf.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D]

        # One-hot encode the values in the memory and average everything across
        # permutations.
        locations_onehot = tf.one_hot( # shape is now B x E x num_locations x num_locations
            locations, self._task_spec.num_locations, dtype=tf.float32
        )
        # logger.debug("locations_onehot shape %s", locations_onehot.shape)
        locations_onehot = locations_onehot.reshape(
            [batch_size, self._task_spec.num_inputs, -1]
        )
        # logger.debug("locations_onehot reshaped to %s", locations_onehot.shape)
        return locations_onehot

    def _make_locations_encoding_binary(self, inputs, batch_size):
        """Creates location encoding using binary representation."""

        memory_binary = int2bin(inputs['memory']).astype(tf.float32)
        registers_binary = int2bin(inputs['registers']).astype(tf.float32)
        # Note the extra I dimension for the length of the binary integer (32)
        locations = tf.concat(
            [memory_binary, registers_binary], axis=-1
        )  # [B, H, P, D*I]
        locations = tf.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D*I]

        locations = locations.reshape([batch_size, self._task_spec.num_inputs, -1])

        return locations


def make_head_network(
    embedding_dim: int,
    output_size: int,
    num_hidden_layers: int = 2,
    name: Optional[str] = None,
) -> Callable[[tf.Tensor,], tf.Tensor]:
    return snn.Sequential(
        [ResBlockV2(embedding_dim) for _ in range(num_hidden_layers)]
        + [snn.Linear(output_size)],
        name=name,
    )


class CategoricalHead(snn.Module):
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

    def __call__(self, x: tf.Tensor):
        # For training returns the logits, for inference the mean.
        logits = self._head(x) # project the embedding to the value support's numbeer of bins 
        probs = tf.nn.softmax(logits) # take softmax -- probabilities over the bins
        # logger.debug("CategoricalHead: logits shape %s, probs shape %s", logits.shape, probs.shape)
        mean = self._value_support.mean(probs) # compute the mean, which is probs * [0, max_val/num_bins, 2max_val/num_bins, max_val]
        return dict(logits=logits, mean=mean)


class NetworkOutput(NamedTuple):
    value: float
    correctness_value_logits: tf.Tensor
    latency_value_logits: tf.Tensor
    policy_logits: tf.Tensor

class PredictionNet(snn.Module):
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

    def __call__(self, embedding: tf.Tensor):
        # logger.debug("PredictionNet: embedding shape %s", embedding.shape)
        policy_head = make_head_network(
            self.embedding_dim, self.task_spec.num_actions
        )
        # logger.debug("PredictionNet: policy_head %s", policy_head)
        value_head = CategoricalHead(self.embedding_dim, self.support)
        # logger.debug("PredictionNet: value_head %s", value_head)
        latency_value_head = CategoricalHead(self.embedding_dim, self.support)
        # logger.debug("PredictionNet: latency_value_head %s", latency_value_head)
        correctness_value = value_head(embedding)
        # logger.debug("PredictionNet: correctness_value shape %s", str({k:v.shape for k, v in correctness_value.items()}))
        latency_value = latency_value_head(embedding)
        # logger.debug("PredictionNet: latency_value shape %s", str({k:v.shape for k, v in latency_value.items()}))

        # embedding is B x embedding_dim
        # with an uninitialised network, its distribution should be close to
        # a standard normal distribution with mean 0 
        
        # for debugging, we can check the distribution of the embedding
        # if logger.isEnabledFor(logging.DEBUG):
        #     embedding_mean = jnp.mean(embedding)
        #     embedding_std = jnp.std(embedding)
        #     embedding_min = jnp.min(embedding)
        #     embedding_max = jnp.max(embedding)
        #     logger.debug("PredictionNet.distr_check: embedding min %s, max %s mean %s std %s", embedding_min, embedding_max, embedding_mean, embedding_std)
        
        policy = policy_head(embedding) # B x num_actions
        
        # similarly, the policy should be close to a uniform distribution
        # with a mean of 1/num_actions
        # if logger.isEnabledFor(logging.DEBUG):
        #     policy_mean = jnp.mean(policy)
        #     policy_std = jnp.std(policy)
        #     policy_min = jnp.min(policy)
        #     policy_max = jnp.max(policy)
        #     logger.debug("PredictionNet.distr_check: policy min %s, max %s mean %s std %s", policy_min, policy_max, policy_mean, policy_std)
        
        output = NetworkOutput(
            value=correctness_value['mean'] + latency_value['mean'],
            correctness_value_logits=correctness_value['logits'],
            latency_value_logits=latency_value['logits'],
            policy_logits=policy,
        )
        # logger.debug("PredictionNet: output %s", str({k: v.shape for k, v in output._asdict().items() if isinstance(v, jnp.ndarray)}))
        return output


class AlphaDevNetwork(snn.Module):
    # NOTE: this won't work :( need to convert from jax/haiku to tf/sonnet
    prediction_net = PredictionNet
    representation_net = RepresentationNet
    
    @staticmethod
    def _return_with_reward_logits(prediction: NetworkOutput) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            prediction.policy_logits,
            prediction.value,
            prediction.correctness_value_logits,
            prediction.latency_value_logits,
        )
    @staticmethod
    def _return_without_reward_logits(prediction: NetworkOutput) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            prediction.policy_logits,
            prediction.value,
        )
    
    def __init__(self, hparams, task_spec,
                 name: str = 'AlphaDevNetwork'):
        super().__init__(name=name)
        self._hparams = hparams
        self._task_spec = task_spec
        self._prediction_net = self.prediction_net(
            task_spec=task_spec,
            value_max=hparams.value_max,
            value_num_bins=hparams.value_num_bins,
            embedding_dim=hparams.embedding_dim,
            name=f'{name}_prediction_net',
        )
        self._representation_net = self.representation_net(
            hparams=hparams,
            task_spec=task_spec,
            embedding_dim=hparams.embedding_dim,
            name=f'{name}_representation_net',
        )
        self._return_fn = (
            self._return_with_reward_logits
            if hparams.categorical_value_loss else
            self._return_without_reward_logits
        )
    
    def __call__(self, inputs: CPUState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes and returns the policy and value logits for the AZLearner."""
        # logger.debug("AlphaDevNetwork: inputs %s", str({k:v.shape for k,v in inputs.items()}))
        # inputs is the observation dict
        embedding: tf.Tensor = self._representation_net(inputs)
        # logger.debug("AlphaDevNetwork: embedding shape %s", embedding.shape)
        prediction: NetworkOutput = self._prediction_net(embedding)
        # logger.debug("AlphaDevNetwork: prediction shape %s", str({k:v.shape for k,v in prediction.items()}))
        return self._return_fn(prediction)


# #################
# model definition 
# #################
# wrapper for AlphaGame
class AssemblyGameModel(models.Model):
    def __init__(
        self,
        task_spec: TaskSpec,
        name: str = 'AssemblyGameModel',
    ):
        super().__init__(name=name)
        self._environment = AssemblyGame(
            task_spec=task_spec,
        )
        self._task_spec = task_spec
        self._needs_reset = False
    
    def load_checkpoint(self):
        """Loads a saved model state, if it exists."""
        self.needs_reset = False
        return self._ckpt

    def save_checkpoint(self):
        """Saves the model state so that we can reset it after a rollout."""
        self._ckpt = self._environment.copy() # TODO: implement

    def update(
        self,
        timestep: TimeStep, # prior to executing the action
        action: tf.Tensor, # opcode, operands
        next_timestep: TimeStep, # after executing the action
    ) -> TimeStep:
        """
        Updates the model given an observation, action, reward, and discount.
        
        This is called in the EnvironmentLoop and is used to keep the 
        model and the environment in sync.
        Args:
            timestep: the current timestep
            action: the action taken
            next_timestep: the next timestep after taking the action
            
        Returns:
            next_timestep
        Raises:
            assertion error is the timestep is not aligned with what the model
            expects.
        """
        # environment will change here, so we might want to reset it.
        self._needs_reset = True
        def assert_timestep():
            # to save time, we only compare the program.
            # it deterministically defines the rest.
            tf.assert_equal(
                timestep.observation['program'],
                self._environment._last_ts.observation['program'],
                message=(
                    f"timestep {timestep.observation['program']} does not match "
                    f"environment {self._environment._last_ts.observation['program']}"
                ),
            )
        if assert_timestep():
            self._environment.step(action)
        else:
            # re-executes the program contained in the timestep
            self._environment.reset(next_timestep)
    
    def reset(self, initial_state: Optional[CPUState] = None):
        """Resets the model, optionally to an initial state."""
        self._needs_reset = False
        if initial_state is not None:
            self._environment.reset(initial_state) #TODO
        else:
            self._environment.reset()

    @property
    def needs_reset(self) -> bool:
        """Returns whether or not the model needs to be reset."""
        return self._needs_reset

# predictor net from JEPA

# #################
# factory functions
# #################
config: AlphaDevConfig = AlphaDevConfig.from_yaml('alphadev_config.yaml')

env_spec = make_environment_spec(AssemblyGame(config.task_spec))

def environment_factory(): # no args
    return AssemblyGameModel(
        task_spec=config.task_spec,
        name='AssemblyGameEnvironment',
    )

def network_factory(env_spec: DiscreteArray): 
    # env_spec here is env.actions 
    # (i.e. enumeration of available actions). we ignore it
    return AlphaDevNetwork(
        hparams=config.hparams,
        task_spec=config.task_spec,
    )

def model_factory(env_spec: EnvironmentSpec): # env_spec
    # again. we ignore the env_spec.
    return AssemblyGameModel(
        task_spec=config.task_spec,
        name='AssemblyGameModel',
    )

# #################
# Agent definition
# #################

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
