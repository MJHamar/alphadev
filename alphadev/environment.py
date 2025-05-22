from typing import Dict, Any, Tuple, List, Callable, Union, Optional
import numpy as np
import tensorflow as tf
from dm_env import Environment, TimeStep, StepType
from acme.specs import EnvironmentSpec, make_environment_spec as acme_make_environment_spec, Array, BoundedArray, DiscreteArray
from acme.agents.tf.mcts import models

from tinyfive.multi_machine import multi_machine
from .config import AlphaDevConfig
from .utils import (
    TaskSpec,
    x86_enumerate_actions, x86_opcode2int, x86_signatures, x86_to_riscv,
    REG_T, MEM_T, IMM_T, Program, CPUState,
)
import multiprocessing as mp

# #################
# Action Spaces 
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
        act_loc_table = np.zeros(((self.max_reg + self.max_mem), action_space_size), dtype=np.bool_)
        # mask for register locations
        reg_locs = np.zeros((self.max_reg + self.max_mem), dtype=np.bool_)
        reg_locs[:self.max_reg] = True
        # mask for memory locations
        mem_locs = np.zeros((self.max_reg + self.max_mem), dtype=np.bool_)
        mem_locs[self.max_reg:] = True
        # boolean mask for actions that only use register locations
        reg_only_actions = np.zeros((action_space_size,), dtype=np.bool_)
        # boolean mask for actions that read from memory locations
        mem_read_actions = np.zeros((action_space_size,), dtype=np.bool_)
        # boolean mask for actions that write to memory locations
        mem_write_actions = np.zeros((action_space_size,), dtype=np.bool_)
        for i, action in enumerate(self.actions.values()):
            # iterate over the x86 instructions currently under consideration
            x86_opcode, x86_operands = action
            signature = x86_signatures[x86_opcode]
            if signature == (REG_T, REG_T):
                act_loc_table[x86_operands, i] = True
                reg_only_actions[i] = True
            else: # action that accesses memory.
                mem_loc, reg_loc = x86_operands if signature == (MEM_T, REG_T) else reversed(x86_operands)
                
                act_loc_table[reg_loc, i] = True
                act_loc_table[mem_loc, i] = True
                if x86_opcode.startswith("l"): # load action
                    mem_read_actions[i] = True
                else:
                    mem_write_actions[i] = True
        
        assert (reg_only_actions & mem_read_actions & mem_write_actions == 0).any(), \
            "Action space was not partitioned correctly"
        assert (reg_only_actions | mem_read_actions | mem_write_actions).all(), \
            "Action space was not partitioned correctly"
        
        self.act_loc_table = tf.constant(act_loc_table)
        self.reg_locs = tf.constant(reg_locs)
        self.mem_locs = tf.constant(mem_locs)
        self.reg_only_actions = tf.constant(reg_only_actions)
        self.mem_read_actions = tf.constant(mem_read_actions)
        self.mem_write_actions = tf.constant(mem_write_actions)

    def get_mask(self, state: Dict[str, tf.Tensor], history: List[Tuple[str, Tuple[int, int]]]) -> tf.Tensor:
        """
        Get the mask over the action space for the given state and history.

        Returns a boolean tensor over the action space, with True values indicating
        valid actions.
        """
        _mems_read = set()
        _mems_written = set()

        def update_history(opcode, operands):
            # update the history with the action.
            # NOTE: history at this point is in RISC-V format.
            # both load and store actions have (absolute) address as position 2
            # rd/rs1 imm rs1/2. It is also in bytes so we divide by 4
            if opcode.startswith("LW"):  # (MEM_T, _)
                _mems_read.add(operands[1]//4)
            elif opcode.startswith("SW"):  # (_, MEM_T)
                _mems_written.add(operands[1]//4)

        if history:
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
        active_registers = state['active_registers']  # shape E x R+(unused) # TODO: we might want to let the emulator know
        active_registers = tf.reduce_any(active_registers, axis=0)  # shape R
        active_memory = state['active_memory']  # shape E x M
        active_memory = tf.reduce_any(active_memory, axis=0)  # shape M

        assert active_registers.shape[0] == self.max_reg, \
            "active registers and max_reg do not match."
        assert active_memory.shape[0] == self.max_mem, \
            "active memory and max_mem do not match."

        # find windows of locations that are valid
        last_reg = tf.cast(tf.argmax(tf.reverse(tf.cast(active_registers, tf.int64), axis=[0])), tf.int32)
        if last_reg != 0: # 0 means window is full
            last_reg = self.max_reg - last_reg
            active_registers = tf.tensor_scatter_nd_update(active_registers, [[last_reg]], [True])
        reg_window = tf.concat([active_registers, tf.zeros((self.max_mem,), dtype=active_registers.dtype)], axis=-1)

        # same for the memory locations
        last_mem = tf.cast(tf.argmax(tf.reverse(tf.cast(active_memory, tf.int64), axis=[0])), tf.int32)
        if last_mem != 0: # 0 means window is full
            last_mem = self.max_mem - last_mem
            active_memory = tf.tensor_scatter_nd_update(active_memory, [[last_mem]], [True])
        mem_window = tf.concat([tf.zeros((self.max_reg,), dtype=active_memory.dtype), active_memory], axis=-1)

        # Identify register-only actions that access *any* location *outside* the active register window.
        # 1. Get access pattern for locations outside the window:
        inactive_loc_access = tf.boolean_mask(act_loc_table, ~reg_window, axis=0)  # Shape (N_inactive_locs, N_actions)
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
        return tf.logical_or(tf.logical_or(reg_only_mask, mem_read_mask), mem_write_mask).numpy()

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
            if tf.reduce_all(insn == 0):
                # reached the end of the program
                break
            insn_idx = self._npy_reversed.get(tuple(insn))
            asm_insn = self.asm_actions.get(insn_idx)
            asm_program.extend(asm_insn)
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
        print(mp.current_process().pid, 'AssemblyGame.__init__')
        self._task_spec = task_spec
        self._inputs = task_spec.inputs.inputs
        self._output_mask = task_spec.inputs.output_mask
        self._outputs = task_spec.inputs.outputs
        self._max_num_hits = tf.math.count_nonzero(self._output_mask)
        # whether to return the correctness and latency components of the reward
        # in the TimeSteps
        self._observe_reward_components = task_spec.observe_reward_components
        
        self._emulator = multi_machine(
            mem_size=task_spec.num_mem*4, # 4 bytes per memory location
            num_machines=task_spec.num_inputs,
            initial_state=self._inputs,
            special_x_regs=np.array([1], dtype=np.int32), # TODO: this is the hard-coded X1 register for now.
        )
        # TODO: make this distributed
        self._action_space_storage = x86ActionSpaceStorage(
            max_reg=task_spec.num_regs,
            max_mem=task_spec.num_mem
        )
        self.reset()

    def _reset_program(self):
        self._program = Program(
            npy_program=np.zeros((self._task_spec.max_program_size, 3), dtype=np.int32),
            asm_program=[],
        )
    
    def _eval_output(self, output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        print(mp.current_process().pid, 'AssemblyGame._eval_output output:', output.shape)
        masked_output = tf.multiply(output, self._output_mask)
        hits = tf.equal(masked_output, self._outputs)
        num_hits = tf.math.count_nonzero(hits)
        all_hits = tf.equal(num_hits, self._max_num_hits)
        print(mp.current_process().pid, 'AssemblyGame._eval_output all_hits:', all_hits, 'num_hits:', num_hits)
        return tf.cast(all_hits, dtype=tf.float32), tf.cast(num_hits, dtype=tf.float32)
    
    def _eval_latency(self) -> tf.Tensor:
        """Returns a scalar latency for the program."""
        latencies = tf.constant([
            self._emulator.measure_latency(self._program.asm_program)
            for _ in range(self._task_spec.num_inputs)], dtype=tf.float32
        )
        return tf.cast(latencies, dtype=tf.float32)
    
    def _compute_reward(self, include_latency: float) -> float:
        # compute the reward based on the latency and correctness
        correctness_reward = self._task_spec.correctness_reward_weight * (
            self._num_hits - self._prev_num_hits
        )
        if correctness_reward < 0:
            correctness_reward = 0.0 # avoid negative rewards.
        # NOTE: _is_correct is nonzero only if num_hits == max_num_hits.
        # so in that case, correctness_reward is always positive
        correctness_reward += self._task_spec.correct_reward * self._is_correct
        
        # update the previous correct items
        latency_reward = 0.0
        if include_latency: # cannot be <0 btw
            latencies = self._eval_latency()
            latency_reward = np.quantile(
                latencies, self._task_spec.latency_quantile
            ) * self._task_spec.latency_reward_weight
        reward = max(correctness_reward - latency_reward, 0.0)
        # if self._num_hits != self._prev_num_hits:
        #     logger.debug(
        #         "AssemblyGame._compute_reward: nh %s, pnh %s, r %s, l %s, c %s",
        #         self._num_hits, self._prev_num_hits, reward, latency_reward, correctness_reward
        #     )
        self._prev_num_hits = self._num_hits
        return reward, latency_reward, correctness_reward
    
    def _make_observation(self) -> Dict[str, tf.Tensor]:
        # get the current state of the CPU
        return CPUState(
            registers=tf.constant(self._emulator.registers[:, :self._task_spec.num_regs], dtype=tf.int32),
            active_registers=tf.constant(self._emulator.register_mask[:, :self._task_spec.num_regs], dtype=tf.bool),
            memory=tf.constant(self._emulator.memory, dtype=tf.int32),
            active_memory=tf.constant(self._emulator.memory_mask, dtype=tf.bool),
            program=tf.constant(self._program.npy_program, dtype=tf.int32),
            program_length=tf.constant(len(self._program), dtype=tf.int32),
            program_counter=tf.constant(self._emulator.program_counter, dtype=tf.int32)
        )._asdict()
    
    def _check_invalid(self) -> bool:
        # either too long or the emulator is in an invalid state
        # logger.debug("AssemblyGame._check_invalid: len %s, inval %s", len(self._program), len(self._program) >= self._task_spec.max_program_size)
        return len(self._program) >= self._task_spec.max_program_size or \
            False # TODO: self._emulator.invalid()
    
    def _update_state(self):
        # first we make an observation
        print(mp.current_process().pid, 'AssemblyGame. make observation')
        observation = self._make_observation()
        # then we check if the program is correct
        print(mp.current_process().pid, 'AssemblyGame._check_invalid')
        self._is_invalid = self._check_invalid()
        if self._is_invalid:
            self._is_correct = False; self._num_hits = 0
        else:
            print(mp.current_process().pid, 'AssemblyGame._eval_output')
            self._is_correct, self._num_hits = self._eval_output(observation['memory'])
        # terminality check
        is_terminal = self._is_correct or self._is_invalid
        
        # we can now compute the reward
        print(mp.current_process().pid, 'AssemblyGame._compute_reward')
        reward, latency, correctness = self._compute_reward(include_latency=is_terminal)
        
        step_type = StepType.FIRST if len(self._program) == 0 else (
                        StepType.MID if not is_terminal else
                            StepType.LAST)
        print(mp.current_process().pid, 'AssemblyGame.make timestep')
        ts = TimeStep(
            step_type=step_type,
            # too many components in acme hard-code the structure of TimeStep, and not
            # everything supports reward to be a dictionary, so we concatenate
            # the reward components into a single tensor
            reward=(tf.constant(reward, dtype=tf.float32) 
                        if not self._observe_reward_components else 
                            tf.constant(np.asarray([reward, correctness, latency]), dtype=tf.float32)),
            discount=tf.constant(1.0, dtype=tf.float32), # NOTE: not sure what discount here means.
            observation=observation,
            # skip latency and correctness
        )
        self._last_ts = ts
        print(mp.current_process().pid, 'AssemblyGame._update_state done')
        return ts
    
    def reset(self, state: Union[TimeStep, CPUState, None]=None) -> TimeStep:
        print(mp.current_process().pid, 'AssemblyGame.reset state is none:', state is None)
        # deletes the program and resets the
        # CPU state to the original inputs
        # logger.debug("AssemblyGame.reset: state is None %s", state is None)
        if state is None:
            print(mp.current_process().pid, 'AssemblyGame.reset: reset emulator')
            self._emulator.reset_state()
            print(mp.current_process().pid, 'AssemblyGame.reset: reset program')
            self._reset_program()
        else:
            # decode the program and execute it.
            # basically the same overhead as copying everything
            # but copying is also not fully possible
            # and program numpy -> asm is unavoidable anyway
            if isinstance(state, TimeStep):
                ts_program = state.observation['program']
            else: # then it is a CPUState._asdict()
                ts_program = state['program']
            # logger.debug("AssemblyGame.reset: ts_program shape %s", ts_program.shape)
            # either B x num_inputs x 3 or no batch dimension
            if len(ts_program.shape) > 2:
                # we need to remove the batch dimension
                assert ts_program.shape[0] == 1, "Batch dimension is not 1, resetting is ambigouous."
                ts_program = tf.squeeze(ts_program, axis=0)

            # convert the numpy program to a list of assembly instructions
            asm_program = self._action_space_storage.npy_to_asm(ts_program.numpy())
            self._program = Program(
                npy_program=ts_program.numpy(),
                asm_program=asm_program,
            )
            self._emulator.reset_state()
            # execute the program only if nonempty
            if len(self._program) > 0:
                # execute the program
                self._emulator.exe(program=self._program.asm_program)
        # calculate the number of hits we have currently,
        # so reset doesn't accidentally return positive reward
        print(mp.current_process().pid, 'AssemblyGame.reset: _eval_output')
        self._prev_num_hits, _ = self._eval_output(self._emulator.memory)
        # update the state
        print(mp.current_process().pid, 'AssemblyGame.reset: _update_state')
        return self._update_state()
    
    def step(self, actions:Union[List[int], int]) -> TimeStep:
        # logger.debug("AssemblyGame.step: action %s", action)
        action_space = self._action_space_storage.get_space()
        if not isinstance(actions, list):
            # single action
            actions = [actions]
        assert len(self._program) + len(actions) <= self._task_spec.max_program_size, \
            "Program size exceeded. Current size: %d, action size: %d" % (len(self._program), len(actions))
        updated_program = self._program.npy_program.copy()
        new_asm_program = []
        for i, action in enumerate(actions):
            # append the action to the program
            action_np = action_space.get_np(action)
            action_asm = action_space.get_asm(action)
            if not isinstance(action_asm, list):
                action_asm = [action_asm]
            updated_program[len(self._program)+i,:] = action_np
            new_asm_program.extend(action_asm)
        
        self._program = Program(
            npy_program=updated_program,
            asm_program=self._program.asm_program + new_asm_program
        )
        # reset the emulator
        self._emulator.reset_state()
        # execute the program
        self._emulator.exe(program=self._program.asm_program)
        # update observation and cached values
        # and return the updated timestep
        return self._update_state()
    
    def legal_actions(self) -> np.ndarray:
        return self._action_space_storage.get_mask(self._last_ts.observation, self._program.asm_program)
    
    def reward_spec(self):
        return Array(shape=(), dtype=np.float32) if not self._observe_reward_components else Array(shape=(3,), dtype=np.float32)
    def discount_spec(self):
        return Array(shape=(), dtype=np.float32)
    def observation_spec(self):
        return CPUState(
            registers=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_regs), dtype=np.int32),
            active_registers=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_regs), dtype=np.bool_),
            memory=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_mem), dtype=np.int32),
            active_memory=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_mem), dtype=np.bool_),
            program=Array(shape=(self._task_spec.max_program_size, 3), dtype=np.int32),
            program_length=Array(shape=(), dtype=np.int32),
            program_counter=Array(shape=(self._task_spec.num_inputs,), dtype=np.int32)
        )._asdict()
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
        new_game._observe_reward_components = self._observe_reward_components
        # copy the two mutable parts of the state
        new_game._emulator = self._emulator.clone()
        # these are also 'immmutable'
        new_game._program = self._program
        new_game._last_ts = self._last_ts
        new_game._is_correct = self._is_correct
        new_game._num_hits = self._num_hits
        new_game._is_invalid = self._is_invalid
        new_game._prev_num_hits = self._prev_num_hits
        return new_game

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
        super().__init__()
        self._environment = AssemblyGame(
            task_spec=task_spec,
        )
        self._task_spec = task_spec
        self._needs_reset = False
    
    def load_checkpoint(self):
        """Loads a saved model state, if it exists."""
        self._needs_reset = False
        self._environment = self._ckpt.copy()

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
        # logger.debug("AssemblyGameModel: update")
        self._needs_reset = True
        def assert_timestep():
            # to save time, we only compare the program.
            # it deterministically defines the rest.
            try:
                tf.assert_equal(
                    timestep.observation['program'],
                    self._environment._last_ts.observation['program'],
                    message=(
                        f"timestep {timestep.observation['program']} does not match "
                        f"environment {self._environment._last_ts.observation['program']}"
                    ),
                )
                return True
            except tf.errors.InvalidArgumentError as e:
                # logger.error("AssemblyGameModel: timestep assertion error %s", e)
                return False
        if assert_timestep():
            return self._environment.step(action)
        else:
            # re-executes the program contained in the timestep
            return self._environment.reset(next_timestep)
    
    def reset(self, initial_state: Optional[CPUState] = None):
        """Resets the model, optionally to an initial state."""
        print(mp.current_process().pid, "AssemblyGameModel: reset")
        self._needs_reset = False
        self._environment.reset(initial_state)
        print(mp.current_process().pid, "AssemblyGameModel: reset done")

    @property
    def needs_reset(self) -> bool:
        """Returns whether or not the model needs to be reset."""
        return self._needs_reset

    def legal_actions(self):
        """Returns the legal actions for the current state."""
        return self._environment.legal_actions()

    def action_spec(self):
        return self._environment.action_spec()
    def reward_spec(self):
        return self._environment.reward_spec()
    def discount_spec(self):
        return self._environment.discount_spec()
    def observation_spec(self):
        return self._environment.observation_spec()
    def step(self, action):
        # logger.debug("AssemblyGameModel: step") 
        return self._environment.step(action)

class EnvironmentFactory:
    def __init__(self, config: AlphaDevConfig): self._task_spec = config.task_spec
    def __call__(self): return AssemblyGame(task_spec=self._task_spec)

class ModelFactory:
    def __init__(self, config: AlphaDevConfig): self._task_spec = config.task_spec
    def __call__(self, env_spec: EnvironmentSpec): return AssemblyGameModel(task_spec=self._task_spec, name='AssemblyGameModel')

def environment_spec_from_config(config: AlphaDevConfig) -> EnvironmentSpec:
    return acme_make_environment_spec(EnvironmentFactory(config)())
