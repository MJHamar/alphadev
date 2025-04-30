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

import collections
import functools
import math
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Union, Tuple, List

import time
import chex
import haiku as hk
import jax
import jax.lax
import jax.numpy as jnp
import ml_collections
import numpy
import optax

from tinyfive.machine import machine as riscv_machine

############################
###### 1. Environment ######


class CPUState(NamedTuple):
    registers: jnp.ndarray
    memory: jnp.ndarray
    register_mask: jnp.ndarray
    memory_mask: jnp.ndarray
    program: jnp.ndarray
    program_length: int
    
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

class RiscvAction(NamedTuple):
    opcode: str
    operands: Tuple[int, ...]

    def __repr__(self):
        return f"RiscvAction(opcode={self.opcode}, operands={self.operands}"

    def __str__(self):
        return f"{self.opcode} {', '.join(map(str, self.operands))}"

def x86_to_riscv(x86_opcode, x86_operands, state):
    """
    Convert x86 opcode and operands to RISC-V opcode and operands.
    Args:
        x86_opcode (str): The x86 opcode.
        x86_operands (list): The x86 operands.
    Returns:
        list: A list of RISC-V instructions. Using the conversion described above
    """
    if x86_opcode == "mv": # move between registers
        return [RiscvAction("ADD", (x86_operands[0], X0, x86_operands[1]))]
    elif x86_opcode == "lw": # load word from memory to register
        return [RiscvAction("SW", (x86_operands[1], x86_operands[0]))]
    elif x86_opcode == "sw": # store word from register to memory
        return [RiscvAction("LW", (x86_operands[0], x86_operands[1]))]
    elif x86_opcode == "cmp": # compare two registers
        return [RiscvAction("SUB", (X1, x86_operands[0], x86_operands[1]))]
    elif x86_opcode == "cmovg": # conditional move if greater than
        return [
            RiscvAction("BLE", (X1, 0, state.pc+8)),  # skip next instruction if A < B
            RiscvAction("ADD", (x86_operands[1], X0, x86_operands[0]))  # copy C to D
        ]
    elif x86_opcode == "cmovle": # conditional move if less than or equal
        return [
            RiscvAction("BGT", (X1, 0, state.pc+8)),  # skip next instruction if A > B
            RiscvAction("ADD", (x86_operands[1], X0, x86_operands[0]))  # copy E to F
        ]
    else:
        raise ValueError(f"Unknown opcode: {x86_opcode}")

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
    def enum_actions(r1: int, r2: int, m: int) -> List[Tuple[str, Tuple[int, int]]]:
        if r1 == 1 and r2 == 1 and m == 1:
            return [apply_opcode(opcode, (r1, r2, m)) for opcode in x86_signatures.keys()]
        actions = []
        if r1 > 1:
            actions.extend(enum_actions(r1 - 1, r2, m))
        if r2 > 1:
            actions.extend(enum_actions(r1, r2 - 1, m))
        if m > 1:
            actions.extend(enum_actions(r1, r2, m - 1))
        actions.extend([apply_opcode(opcode, (r1, r2, m)) for opcode in x86_signatures.keys()])
        return actions
    actions = enum_actions(max_reg, max_reg, max_mem)
    return actions

# the emulator should call ActionSpace.get(action) to get the action
# the action space can use the emulator to get the action
# the action space also needs to define a masking function that masks invalid actions based on the emulator state
class ActionSpace(object):
    # placeholder for now.
    pass

class x86ActionSpace(ActionSpace):
    def __init__(self,
            actions: List[Callable[[Any],Tuple[str, Tuple[int, int]]]],
            state: 'CPUState'):
        self.actions = actions
        self.state = state
        
    def get(self, action_id: int) -> List[RiscvAction]:
        return x86_to_riscv(*self.actions[action_id], self.state) # convert to RISC-V

class ActionSpaceStorage(object):
    # placeholder for now.
    pass

class x86ActionSpaceStorage(ActionSpaceStorage):
    def __init__(self, max_reg: int, max_mem: int):
        self.max_reg = max_reg
        self.max_mem = max_mem
        self.actions = x86_enumerate_actions(max_reg, max_mem)
        # there is a single action space for the given task
        self.action_space = x86ActionSpace # these are still x86 instructions
        # TODO: make sure we don't flood the memory with this
        self.masks = {}
        # build mask lookup tables
        self.reg_masks = None
        self.mem_masks = None
        self._build_masks()
        # for pruning the action space (one read and one write per memory location)
        self._history_cache = None
        self._mems_read = set()
        self._mems_written = set()
        
    def _build_masks(self):
        """
        Build masks over the action space for each register and memory location.
        At runtime, we can dynamically take the union of a subset of these masks
        to efficiently mask the action space. 
        """
        # we create a max_reg x action_space_size and 
        # max_mem x action_space_size masks
        action_space_size = len(self.actions)
        reg_masks = jnp.zeros((self.max_reg, action_space_size), dtype=jnp.bool)
        mem_masks = jnp.zeros((self.max_mem, action_space_size), dtype=jnp.bool)
        for i, action in enumerate(self.actions):
            # iterate over the x86 instructions currently under consideration
            x86_opcode, x86_operands = action
            signature = x86_signatures[x86_opcode]
            if signature == (REG_T, REG_T):
                # move between registers
                reg_masks[x86_operands[0], i] = True
                reg_masks[x86_operands[1], i] = True
            elif signature == (MEM_T, REG_T):
                # load word from memory to register
                mem_masks[x86_operands[0], i] = True
                reg_masks[x86_operands[1], i] = True
            elif signature == (REG_T, MEM_T):
                # store word from register to memory
                reg_masks[x86_operands[0], i] = True
                mem_masks[x86_operands[1], i] = True
            else:
                assert False, f"No signature of type {signature} should be used. fix this."
        
        self.reg_masks = reg_masks
        self.mem_masks = mem_masks

    def get_masks(self, state, history=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        def update_history(action):
            # update the history with the action
            if action.opcode == "lw": # (MEM_T, _)
                self._mems_read.add(action.operands[0])
            elif action.opcode == "sw": # (_, MEM_T)
                self._mems_written.add(action.operands[1])
        
        if state not in self.masks:
            active_registers = state.register_mask
            active_memory = state.memory_mask
            # increment the active registers and memory by 1 to allow accessing locations
            # find the last non-zero index
            last_reg = jnp.argmax(active_registers[::-1])
            last_mem = jnp.argmax(active_memory[::-1])
            reg_mask = jnp.any(self.reg_masks[active_registers, last_reg], axis=0)
            mem_mask = jnp.any(self.mem_masks[active_memory, last_mem], axis=0)
            self.masks[state] = (reg_mask, mem_mask)
        # now we need to make sure that only one read and one write is allowed for each memory location
        # this should be computed on the fly, as the program keeps changing in crazy ways.
        # BUT, we can cache the histor so we don't have to iterate over it every time
        # since we only use this in MCTS which is depth-first

        reg_mask, mem_mask = self.masks[state]
        reg_mask = reg_mask.copy()
        mem_mask = mem_mask.copy()

        # check if the history seems to be a continuation of the current state
        if history is not None:
            if self.history_cache is not None and\
                len(self._history_cache) + 1 == len(history) and \
                self._history_cache[-1] == history[-2]: # check only the last action to save time
                # we can use the cached history
                update_history(history[-1])
            else:
                # we need to recompute the history
                self._mems_read = set()
                self._mems_written = set()
                # iterate over the history
                for action in history:
                    update_history(action)
            # update the history cache
            self._history_cache = history
            # update the masks
            mem_mask |= jnp.any(self.mem_masks[self._mems_read], axis=0)
            mem_mask |= jnp.any(self.mem_masks[self._mems_written], axis=0)

        return reg_mask, mem_mask

    def get_space(self, state) -> x86ActionSpace:
        return self.action_space(self.actions, state)

class IOExample(NamedTuple):
    inputs: jnp.ndarray
    outputs: jnp.ndarray

class TaskSpec(NamedTuple):
    max_program_size: int
    num_inputs: int # TODO: ??? wtf is this
    num_funcs: int # number of x86 instructions to consider
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

    class AssemblySimulator(riscv_machine):
        
        def __init__(self, task_spec, example:IOExample=None):
            """Initialize the simulator with the task specification."""
            super().__init__(mem_size=task_spec.num_memory_locs + task_spec.max_program_size)
            self.task_spec = task_spec
            self.example = example
            self.reset()
        
        def reset(self):
            """Reset the simulator to an initial state."""
            self.clear_cpu(); self.clear_mem()
            # initialise write heads and program counter.
            self.mem_write_head = 0
            self.pc = self.task_spec.num_memory_locs
            if self.example is not None:
                self._populate_memory(self.example.inputs)

        def _populate_memory(self, inputs: Union[jnp.ndarray, int, float]):
            # overflow checks
            if (self.mem_write_head >= self.task_spec.num_memory_locs or
                # isinstance(inputs, (int, float)) and 
                (isinstance(inputs, jnp.ndarray) and
                    self.mem_write_head + inputs.size >= self.task_spec.num_memory_locs
                )):
                raise ValueError("Memory overflow: cannot write to memory.")
            if isinstance(inputs, int):
                self.write_i32(inputs, self.mem_write_head)
                self.mem_write_head += 4 # increment
            elif isinstance(inputs, float):
                self.write_f32(inputs, self.mem_write_head)
                self.mem_write_head += 4
            elif isinstance(inputs, jnp.ndarray):
                if inputs.dtype == jnp.int32:
                    self.write_i32_vec(inputs, self.mem_write_head)
                    self.mem_write_head += inputs.size * 4
                elif inputs.dtype == jnp.float32:
                    self.write_f32_vec(inputs, self.mem_write_head)
                    self.mem_write_head += inputs.size * 4
                else:
                    raise ValueError(f"Unsupported data type: {inputs.dtype}")

        # pylint: disable-next=unused-argument
        def apply(self, instruction):
            """Apply an assembly instruction to the simulator."""
            # NOTE: to avoid the overhead of encoding and then decoding the instruction (O(n) time),
            # we use the uppercase methods of the machine class directly.
            op_str = instruction.opcode
            op_fn = getattr(self, op_str)
            op_fn(*instruction.oprands) # NOTE: we need to ensure that `operands` is correct.
            return self.get_state()

        def get_state(self) -> CPUState:
            """Get the current state of the simulator."""
            return CPUState(
                registers=self.registers,
                memory=self.memory,
                register_mask=self.register_mask,
                memory_mask=self.memory_mask,
                program=self.program,
                program_length=len(self.program)
            )

        def measure_latency(self, program) -> float:
            crnt_state = self.copy()
            self.reset()
            start_time = time.time()
            for instruction in program:
                op_str = instruction.opcode
                op_fn = getattr(self, op_str)
                op_fn(*instruction.operands)
            end_time = time.time()
            latency = end_time - start_time
            self.reset()
            # restore state
            self = crnt_state
            return float(latency)

        @property
        def registers(self): return jnp.concat([self.x.astype(self.dtype), self.f.astype(self.dtype)], axis=0)
        @property
        def memory(self): return jnp.array(self.mem[:self.task_spec.num_memory_locs], dtype=self.dtype)
        
        @property
        def register_mask(self): return jnp.concat(self.x_usage != 0, self.f_usage != 0, axis=0)
        @property
        def memory_mask(self): return jnp.array(self.mem[:self.task_spec.num_memory_locs] != 0)
        
        def invalid(self) -> bool:
            # FIXME: need to check if the program can be invalid at any point in time.
            return False

    def __init__(self, task_spec, example: IOExample, action_space_storage: ActionSpaceStorage):
        self.task_spec = task_spec
        self.example = example
        self.storage = action_space_storage
        self.program = [] # program here is an array, which suggests that there are only append actions
        self.simulator = self.AssemblySimulator(task_spec, example)
        self.previous_correct_items = 0
        self.expected_outputs = self.make_expected_outputs()

    def step(self, action:'Action'):
        action_space = self.storage.get_space(self.simulator.get_state())
        instructions = action_space.get(action.index) # lookup x86 instructions and convert to riscv
        # there might be multiple instructions in a single action
        if not isinstance(instructions, list):
            instructions = [instructions]
        for riscv_action in instructions:
            insn = self.AssemblyInstruction(riscv_action, self.storage)
            self.program.append(insn) # append the action (no swap moves)
            self.execution_state = self.simulator.apply(insn)
        return self.observation(), self.correctness_reward()

    def observation(self):
        return {
            'program': self.program,
            'program_length': len(self.program),
            'memory': self.execution_state.memory,
            'registers': self.execution_state.registers,
            'register_mask': self.execution_state.register_mask,
            'memory_mask': self.execution_state.memory_mask,
        }

    def make_expected_outputs(self):
        # TODO: this might need refining.
        return jnp.array(
            self.example.outputs
        )

    def correctness_reward(self) -> float:
        """Computes a reward based on the correctness of the output."""
        state = self.execution_state

        # Weighted sum of correctly placed items
        correct_items = 0
        for output, expected in zip(state.memory, self.expected_outputs):
            correct_items += output.weight * sum(
                output[i] == expected[i] for i in range(len(output))
            )
            reward = self.task_spec.correctness_reward_weight * (
                correct_items - self.previous_correct_items
            )
        self.previous_correct_items = correct_items

        # Bonus for fully correct programs
        all_correct = jnp.all(state.memory == self.expected_outputs)
        reward += self.task_spec.correct_reward * all_correct

        return reward

    def latency_reward(self) -> float:
        latency_samples = [
            # NOTE: measure latency n times
            self.simulator.measure_latency(self.program)
            for _ in range(self.task_spec.num_latency_simulation)
        ]
        return (
            numpy.quantile(latency_samples, self.task_spec.latency_quantile)
            * self.task_spec.latency_reward_weight
        )

    def clone(self):
        pass

    def terminal(self) -> bool: # TODO: ? when else
        return self.simulator.invalid() or self.correct()

    def correct(self) -> bool:
        state = self.execution_state
        return jnp.all(state.memory == self.expected_outputs)

######## End Environment ########
#################################

#####################################
############ 2. Networks ############

######## 2.1 Network helpers ########


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

class NetworkOutput(NamedTuple):
    value: float
    correctness_value_logits: jnp.ndarray
    latency_value_logits: jnp.ndarray
    policy_logits: Dict[Action, float]


class Network(object):
    """Wrapper around Representation and Prediction networks."""

    def __init__(self, hparams: ml_collections.ConfigDict, task_spec: TaskSpec):
        self.representation = hk.transform(RepresentationNet(
            hparams, task_spec, hparams.embedding_dim
        ))
        self.prediction = hk.transform(PredictionNet(
            task_spec=task_spec,
            value_max=hparams.value.max,
            value_num_bins=hparams.value.num_bins,
            embedding_dim=hparams.embedding_dim,
        ))
        rep_key, pred_key = jax.random.PRNGKey(42).split()
        self.params = {
            'representation': self.representation.init(rep_key),
            'prediction': self.prediction.init(pred_key),
        }

    def inference(self, params: Any, observation: jnp.array) -> NetworkOutput:
        # NOTE: observation here is an array, not a dict like in AssemblyGame. Where do we convert
        # representation + prediction function
        # NOTE: the representation net actually expects a dict, so only the type annotation is wrong.
        embedding = self.representation.apply(params['representation'], observation)
        return self.prediction.apply(params['prediction'], embedding)

    def get_params(self):
        # Returns the weights of this network.
        return self.params

    def update_params(self, updates: Any) -> None:
        # Update network weights internally.
        self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0


class UniformNetwork(object):
    # NOTE: what does this do?
    """Network representation that returns uniform output."""

    # pylint: disable-next=unused-argument
    def inference(self, observation) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0, 0, 0, {})

    def get_params(self):
        # Returns the weights of this network.
        return self.params

    def update_params(self, updates: Any) -> None:
        # Update network weights internally.
        self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0


######## 2.2 Representation Network ########


class MultiQueryAttentionBlock:
    """Attention with multiple query heads and a single shared key and value head.

    Implementation of "Fast Transformer Decoding: One Write-Head is All You Need",
    see https://arxiv.org/abs/1911.02150.
    """


class ResBlockV2:
    """Layer-normed variant of the block from https://arxiv.org/abs/1603.05027."""


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
        #NOTE: no init here, haiku takes care of it.

    def __call__(self, inputs):
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

        return self.aggregate_locations_program(
            locations_encoding, permutation_embedding, program_encoding, batch_size
        )

    def _encode_program(self, inputs, batch_size):
        program = inputs['program']
        max_program_size = inputs['program'].shape[1]
        program_length = inputs['program_length'].astype(jnp.int32)
        program_onehot = self.make_program_onehot(
            program, batch_size, max_program_size
        )
        program_encoding = self.apply_program_mlp_embedder(program_onehot)
        program_encoding = self.apply_program_attention_embedder(program_encoding)
        return self.pad_program_encoding(
            program_encoding, batch_size, program_length, max_program_size
        )

    def aggregate_locations_program(
        self,
        locations_encoding,
        unused_permutation_embedding,
        program_encoding,
        batch_size,
    ):
        locations_embedder = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name='per_locations_embedder',
        )

        # locations_encoding.shape == [B, P, D] so map embedder across locations to
        # share weights
        locations_embedding = hk.vmap(
            locations_embedder, in_axes=1, out_axes=1, split_rng=False
        )(locations_encoding)

        program_encoded_repeat = self.repeat_program_encoding(
            program_encoding, batch_size
        )

        grouped_representation = jnp.concatenate(
            [locations_embedding, program_encoded_repeat], axis=-1
        )

        return self.apply_joint_embedder(grouped_representation, batch_size)

    def repeat_program_encoding(self, program_encoding, batch_size):
        return jnp.broadcast_to(
            program_encoding,
            [batch_size, self._task_spec.num_inputs, program_encoding.shape[-1]],
        )

    def apply_joint_embedder(self, grouped_representation, batch_size):
        all_locations_net = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name='per_element_embedder',
        )
        joint_locations_net = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1),
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
        permutations_encoded = all_locations_net(grouped_representation)
        # Combine all permutations into a single vector.
        joint_encoding = joint_locations_net(jnp.mean(permutations_encoded, axis=1))
        for net in joint_resnet:
            joint_encoding = net(joint_encoding)
        return joint_encoding

    def make_program_onehot(self, program, batch_size, max_program_size):
        func = program[:, :, 0] # the opcode -- int
        arg1 = program[:, :, 1] # the first operand -- int 
        arg2 = program[:, :, 2] # the second operand -- int
        func_onehot = jax.nn.one_hot(func, self._task_spec.num_funcs)
        arg1_onehot = jax.nn.one_hot(arg1, self._task_spec.num_locations)
        arg2_onehot = jax.nn.one_hot(arg2, self._task_spec.num_locations)
        program_onehot = jnp.concatenate(
            [func_onehot, arg1_onehot, arg2_onehot], axis=-1
        )
        chex.assert_shape(program_onehot, (batch_size, max_program_size, None))
        return program_onehot

    def pad_program_encoding(
        self, program_encoding, batch_size, program_length, max_program_size
    ):
        """Pads the program encoding to account for state-action stagger."""
        chex.assert_shape(program_encoding, (batch_size, max_program_size, None))

        empty_program_output = jnp.zeros(
            [batch_size, program_encoding.shape[-1]],
        )
        program_encoding = jnp.concatenate(
            [empty_program_output[:, None, :], program_encoding], axis=1
        )

        program_length_onehot = jax.nn.one_hot(program_length, max_program_size + 1)

        program_encoding = jnp.einsum(
            'bnd,bNn->bNd', program_encoding, program_length_onehot
        )

        return program_encoding

    def apply_program_mlp_embedder(self, program_encoding):
        program_embedder = hk.Sequential(
            [
                hk.Linear(self._embedding_dim),
                hk.LayerNorm(axis=-1),
                jax.nn.relu,
                hk.Linear(self._embedding_dim),
            ],
            name='per_instruction_program_embedder',
        )

        program_encoding = program_embedder(program_encoding)
        return program_encoding

    def apply_program_attention_embedder(self, program_encoding):
        attention_params = self._hparams.representation.attention
        make_attention_block = functools.partial(
            MultiQueryAttentionBlock, attention_params, causal_mask=False
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
            program_encoding, _ = e(program_encoding, encoded_state=None)

        return program_encoding

    def _make_locations_encoding_onehot(self, inputs, batch_size):
        """Creates location encoding using onehot representation."""
        memory = inputs['memory']
        registers = inputs['registers']
        locations = jnp.concatenate([memory, registers], axis=-1)  # [B, H, P, D]
        locations = jnp.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D]

        # One-hot encode the values in the memory and average everything across
        # permutations.
        locations_onehot = jax.nn.one_hot(
            locations, self._task_spec.num_location_values, dtype=jnp.int32
        )
        locations_onehot = locations_onehot.reshape(
            [batch_size, self._task_spec.num_inputs, -1]
        )

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
        pass

    def scalar_to_two_hot(self, scalar: float) -> jnp.ndarray:
        pass


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
        logits = self._head(x)
        probs = jax.nn.softmax(logits)
        mean = jax.vmap(self._value_support.mean)(probs)
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
        self.support = DistributionSupport(self.value_max, self.value_num_bins)
        self.embedding_dim = embedding_dim

    def __call__(self, embedding: jnp.ndarray):
        policy_head = make_head_network(
            self.embedding_dim, self.task_spec.num_actions
        )
        value_head = CategoricalHead(self.embedding_dim, self.support)
        latency_value_head = CategoricalHead(self.embedding_dim, self.support)
        correctness_value = value_head(embedding)
        latency_value = latency_value_head(embedding)

        return NetworkOutput(
            value=correctness_value['mean'] + latency_value['mean'],
            correctness_value_logits=correctness_value['logits'],
            latency_value_logits=latency_value['logits'],
            policy=policy_head(embedding),
        )


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
        self.num_simulations = 800
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
            max_program_size=100,
            num_inputs=17,
            num_funcs=14,
            num_locations=19,
            num_actions=271,
            correct_reward=1.0,
            correctness_reward_weight=2.0,
            latency_reward_weight=0.5,
            latency_quantile=0,
        )

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
        self.batch_size = 512
        self.td_steps = 5
        self.lr_init = 2e-4
        self.momentum = 0.9

    def new_game(self):
        return Game(self.task_spec.num_actions, self.discount, self.task_spec)


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
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


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
        self, action_space_size: int, discount: float, task_spec: TaskSpec, example: IOExample = None
    ):
        self.task_spec = task_spec
        # TODO: also pass io example and action space storage
        self.action_space_storage = ActionSpaceStorage(max_reg=task_spec.num_funcs, max_mem=task_spec.)
        self.environment = AssemblyGame(task_spec, example, action_space_storage)
        self.history = []
        self.rewards = []
        self.latency_reward = 0
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        # Game specific termination rules.
        # For sorting, a game is terminal if we sort all sequences correctly or
        # we reached the end of the buffer.
        return self.environment.terminal()

    def is_correct(self) -> bool:
        # Whether the current algorithm solves the game.
        pass

    def legal_actions(self) -> Sequence[Action]:
        # Game specific calculation of legal actions.
        # NOTE: implement as an enumeration of the allowed actions,
        # instead of filtering the pre-defined action space.
        return []

    def apply(self, action: Action):
        _, reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)
        if self.terminal() and self.is_correct():
            self.latency_reward = self.environment.latency_reward()

    def store_search_statistics(self, root: Node):
        # NOTE: this function is used to store statistics about the observed trajectory (outside of MCTS)
        sum_visits = sum(child.visit_count for child in root.children.values())
        # NOTE: this is a problem. MCTS assumes a fixed action space, 
        # which i don't know how to enumerate.
        # fair point tho that we CAN compute this on the fly,
        # and store the available actions in expanded nodes.
        # TODO: add a shared action storage to cache the action spaces based on
        # active register and memory locations.
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits
                if a in root.children
                else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def make_observation(self, state_index: int):
        # NOTE: returns the observation corresponding to the last action
        if state_index == -1:
            return self.environment.observation()
        # NOTE: re-play the game from the initial state.
        # FIXME: initial state depends on the current task we are solving
        # unlike in AlphaDev, where the initial state is always the same.
        # TODO: also pass the action space storage to the game.
        env = AssemblyGame(self.task_spec)
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
        bootstrap_index = state_index + td_steps

        for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
            value += reward * self.discount**i  # pytype: disable=unsupported-operands

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

    def sample_batch(self, td_steps: int) -> Sequence[Sample]:
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        # pylint: disable=g-complex-comprehension
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
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[0]

    # pylint: disable-next=unused-argument
    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return -1


class SharedStorage(object):
    """Controls which network is used at inference."""

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
        # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


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

    # TODO: this should handled by a multiprocessing pool
    # or a job queue.
    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

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
        root = Node(0)
        current_observation = game.make_observation(-1)
        network_output = network.inference(current_observation)
        _expand_node( # expand root
            root, game.to_play(), game.legal_actions(), network_output, reward=0
        )
        _backpropagate(
            [root], # only the root
            network_output.value, # predicted value at the root
            game.to_play(), # dummy player
            config.discount,
            min_max_stats,
        )
        _add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using the environment.
        run_mcts(
            config,
            root,
            game.action_history(),
            network,
            min_max_stats,
            game.environment,
        )
        # NOTE: make a move after the MCTS policy improvement step
        action = _select_action(config, len(game.history), root, network)
        game.apply(action) # step the environment
        game.store_search_statistics(root)
    return game


def run_mcts(
    config: AlphaDevConfig,
    root: Node,
    action_history: ActionHistory,
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

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]
        sim_env = env.clone() # start from the current state of the environment
        # Traverse the tree until we reach a leaf node.

        while node.expanded():
            action, node = _select_child(config, node, min_max_stats) # based on UCB
            sim_env.step(action) # step the environment
            history.add_action(action) # update histroy 
            search_path.append(node) # append to the current trajectory

        # Inside the search tree we use the environment to obtain the next
        # observation and reward given an action.
        observation, reward = sim_env.step(action) # step the environment
        network_output = network.inference(observation) # get the priors
        _expand_node(
            node, history.to_play(), history.action_space(), network_output, reward
        )

        _backpropagate(
            search_path,
            network_output.value,
            history.to_play(),
            config.discount,
            min_max_stats,
        )
        # NOTE: excuse me but how is this supposed to reach a leaf node? 
        # it seems like we are expanding exactly one node in each iteration.
        # there should be an exit condition upon terminations of the game.


def _select_action(
    # pylint: disable-next=unused-argument
    config: AlphaDevConfig, num_moves: int, node: Node, network: Network
) -> Action:
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        training_steps=network.training_steps()
    )
    _, action = softmax_sample(visit_counts, t)
    return action


def _select_child(
    config: AlphaDevConfig, node: Node, min_max_stats: MinMaxStats
):
    """Selects the child with the highest UCB score."""
    _, action, child = max(
        (_ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    )
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
    node.hidden_state = network_output.hidden_state
    node.reward = reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


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
    network = Network(config.hparams, config.task_spec) # the one we train
    target_network = Network(config.hparams, config.task_spec) # target network is updated periodically (bootstrap)
    optimizer = optax.sgd(config.lr_init, config.momentum)
    optimizer_state = optimizer.init(network.get_params())

    for i in range(config.training_steps): # insertion point for training pipeline
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network) # save the current network
        if i % config.target_network_interval == 0:
            target_network = network.copy() # update the bootstrap network
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        optimizer_state = _update_weights(
            optimizer, optimizer_state, network, target_network, batch)
    storage.save_network(config.training_steps, network)

def scale_gradient(tensor: Any, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + jax.lax.stop_gradient(tensor) * (1 - scale)

def _loss_fn(
    network_params: jnp.array,
    target_network_params: jnp.array,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample]
) -> float:
    """Computes loss."""
    loss = 0
    for observation, bootstrap_obs, target in batch: # processes batches
        # NOTE: re-compute the priors instead of using the cached ones
        # which is fine, we want the updated network to be used for each batch.
        predictions = network.inference(network_params, observation)
        bootstrap_predictions = target_network.inference(
            target_network_params, bootstrap_obs)
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
) -> Any:
    """Updates the weight of the network."""
    updates = _loss_grad(
        network.get_params(),
        target_network.get_params(),
        network,
        target_network,
        batch)

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
    # TODO
    return 0, 0


def launch_job(f, *args):
    # NOTE: a simple wrapper to launch a job in a separate thread.
    f(*args)


def make_uniform_network():
    return UniformNetwork()

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
