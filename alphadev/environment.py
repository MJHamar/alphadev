from typing import Dict, Any, Tuple, List, Callable, Union, Optional, Literal
import functools
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
    def __init__(self, max_reg: int, max_mem: int, mode: Literal['u8', 'i32'] = 'i32'):
        self.max_reg = max_reg
        self.max_mem = max_mem
        self.mode = mode
        self.actions: Dict[int, Dict[int, Tuple[str, Tuple[int,int]]]] =\
            x86_enumerate_actions(max_reg, max_mem)
        self.all_actions = tf.constant(tf.range(len(self.actions), dtype=tf.int32))
        # pre-compute the assembly representation of the actions
        self.asm_actions = {
            i: x86_to_riscv(action[0], action[1], self.max_reg, mode=mode) for i, action in self.actions.items()
        }
        self.np_actions = {
            i: np.array([x86_opcode2int[action[0]], action[1][0], action[1][1]])
            for i, action in self.actions.items()
        }
        self._npy_reversed = {
            tuple(v): k for k, v in self.np_actions.items()
        }
        self.np_action_space = np.stack(list(self.np_actions.values()))
        # there is a single action space for the given task
        self.action_space_cls = ActionSpace # these are still x86 instructions
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
        reg_access = np.zeros((len(self.actions),), dtype=np.int32) # largest register accessed by action
        mem_access = np.zeros((len(self.actions),), dtype=np.int32) # largest memory accessed by action
        is_mem_read = np.zeros((len(self.actions),), dtype=np.bool_) # 1 if action is a memory read, 0 otherwise
        is_mem_write = np.zeros((len(self.actions),), dtype=np.bool_) # 1 if action is a memory write, 0 otherwise
        for idx, action in self.actions.items():
            opcode, operands = action[0], action[1] # NOTE: mem locs are offset by max_reg
            if opcode == "lw": # mv mem, reg
                reg_access[idx] = operands[1]
                mem_access[idx] = operands[0] - self.max_reg
                is_mem_read[idx] = True
                is_mem_write[idx] = False
            elif opcode == "sw": # mv reg, mem
                reg_access[idx] = operands[0]
                mem_access[idx] = operands[1] - self.max_reg
                is_mem_read[idx] = False
                is_mem_write[idx] = True
            else: # all others are register-only actions
                reg_access[idx] = max(operands)  # rd or rs1
                mem_access[idx] = self.max_mem  # no memory access
                is_mem_read[idx] = False; is_mem_write[idx] = False
        # convert to tensors
        self.reg_access = tf.constant(reg_access, dtype=tf.int32)  # shape (N_actions,)
        self.mem_access = tf.constant(mem_access, dtype=tf.int32)  # shape (N_actions,)
        self.mem_access_1hot = tf.one_hot(self.mem_access, depth=self.max_mem, on_value=True, off_value=False, dtype=tf.bool)
        self.is_mem_read = tf.constant(is_mem_read, dtype=tf.bool)  # shape (N_actions,)
        self.is_mem_write = tf.constant(is_mem_write, dtype=tf.bool)  # shape (N_actions,)

    def get_mask(self, state: Dict[str, tf.Tensor], history: List[Tuple[str, Tuple[int, int]]]) -> tf.Tensor:
        """
        Get the mask over the action space for the given state and history.

        Returns a boolean tensor over the action space, with True values indicating
        valid actions.
        """
        _mems_read = [False] * (self.max_mem) # add an extra entry so reg-only actions can index it.
        _mems_written = [False] * (self.max_mem) # add an extra entry so reg-only actions can index it.
        blen = 4 if self.mode == 'u8' else 2 if self.mode == 'i16' else 1

        def update_history(opcode, operands):
            # update the history with the action.
            # NOTE: history at this point is in RISC-V format.
            # both load and store actions have (absolute) address at position 2
            # rd/rs1 imm rs1/2. It is also in bytes so we divide by 4
            # NOTE: memory locations are not offset by max_reg here
            if opcode.startswith("LW"):  # (MEM_T, _)
                _mems_read[operands[1]//blen] = True
            elif opcode.startswith("SW"):  # (_, MEM_T)
                _mems_written[operands[1]//blen] = True

        if history:
            # iterate over the history
            for action in history:
                update_history(*action)
        # convert the sets to tensors
        mem_reads = tf.constant(_mems_read, dtype=tf.bool)
        mem_writes = tf.constant(_mems_written, dtype=tf.bool)
        # get the maximum register and memory indices
        reg_access = state['active_registers']
        reg_access = tf.reduce_any(reg_access, axis=0)  # N_inputs x N_regs -> N_regs
        mem_access = state['active_memory']
        mem_access = tf.reduce_any(mem_access, axis=0)  # N_inputs x N_mem -> N_mem
        reg_max_idx = tf.reduce_sum(tf.cast(reg_access, tf.int32))
        mem_max_idx = tf.reduce_sum(tf.cast(mem_access, tf.int32)) + self.max_reg # offset
        # create a mask over the action space
        # action is valid if it is
        # within the register window
        reg_ok = self.reg_access <= reg_max_idx
        # within the memory window
        mem_ok = self.mem_access <= mem_max_idx
        # either is isn't a memory read or the memory at this location was not read before
        mem_r_ok = tf.reduce_any(
            [
            ~self.is_mem_read, 
            ~tf.reduce_any(tf.boolean_mask(self.mem_access_1hot, mem_reads, axis=1),axis=1)
            ],
            axis=0
        )
        # either is isn't a memory write or the memory at this location was not written before
        mem_w_ok = tf.reduce_any(
            [
            ~self.is_mem_write,
            ~tf.reduce_any(tf.boolean_mask(self.mem_access_1hot, mem_writes, axis=1),axis=1)
            ],
            axis=0
        )
        # assert not tf.reduce_any(mem_reads) or tf.reduce_any(~mem_r_ok), \
        #     "There are memory reads but no actions are filtered"
        # assert not tf.reduce_any(mem_writes) or tf.reduce_any(~mem_w_ok), \
        #     "There are memory writes but no actions are filtered"
        return tf.reduce_all([reg_ok, mem_ok, mem_r_ok, mem_w_ok], axis=0)


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
        self._task_spec = task_spec
        self._inputs = task_spec.inputs.inputs
        self._output_mask = task_spec.inputs.output_mask
        self._outputs = task_spec.inputs.outputs
        self._max_num_hits = tf.math.count_nonzero(self._output_mask)
        # whether to return the correctness and latency components of the reward
        # in the TimeSteps
        self._observe_reward_components = task_spec.observe_reward_components
        
        self._emulator = multi_machine(
            mem_size=task_spec.num_mem,
            num_machines=task_spec.num_inputs,
            initial_state=self._inputs,
            # TODO: this is the hard-coded X1 register for now.
            special_x_regs=np.array([1], dtype=np.int32),
            mode=task_spec.emulator_mode, # use 32-bit mode
        )
        # TODO: make this distributed
        self._action_space_storage = x86ActionSpaceStorage(
            max_reg=task_spec.num_regs,
            max_mem=task_spec.num_mem,
            mode=task_spec.emulator_mode, # use 32-bit mode
        )
        self.reset()

    def _reset_program(self):
        self._program = Program(
            npy_program=np.zeros((self._task_spec.max_program_size, 3), dtype=np.int32),
            asm_program=[],
        )
    
    def _eval_output(self, output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        masked_output = tf.multiply(output, self._output_mask)
        hits = tf.equal(masked_output, self._outputs)
        num_hits = tf.math.count_nonzero(hits)
        all_hits = tf.equal(num_hits, self._max_num_hits)
        return tf.cast(all_hits, dtype=tf.float32), tf.cast(num_hits, dtype=tf.float32)
    
    def _eval_latency(self) -> tf.Tensor:
        """Returns a scalar latency for the program."""
        latencies = tf.constant([
            self._emulator.measure_latency(self._program.asm_program)
            for _ in range(self._task_spec.num_latency_simulations)], dtype=tf.float32
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
        observation = self._make_observation()
        # then we check if the program is correct
        self._is_invalid = self._check_invalid()
        if self._is_invalid:
            self._is_correct = False; self._num_hits = 0
        else:
            self._is_correct, self._num_hits = self._eval_output(observation['memory'])
        # terminality check
        is_terminal = self._is_correct or self._is_invalid
        
        # we can now compute the reward
        reward, latency, correctness = self._compute_reward(include_latency=is_terminal)
        
        step_type = StepType.FIRST if len(self._program) == 0 else (
                        StepType.MID if not is_terminal else
                            StepType.LAST)
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
        return ts
    
    def reset(self, state: Union[TimeStep, CPUState, None]=None) -> TimeStep:
        # deletes the program and resets the
        # CPU state to the original inputs
        # logger.debug("AssemblyGame.reset: state is None %s", state is None)
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
        self._prev_num_hits, _ = self._eval_output(self._emulator.memory)
        # update the state
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
        tf_mask = self._action_space_storage.get_mask(self._last_ts.observation, self._program.asm_program)
        # convert the mask to a numpy array
        legal_actions = tf_mask.numpy().astype(np.bool_)
        return legal_actions
    
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
        self._needs_reset = False
        self._environment.reset(initial_state)

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
