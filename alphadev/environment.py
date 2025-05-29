from typing import Dict, Any, Tuple, List, Callable, Union, Optional, Literal
import functools
import numpy as np
import tensorflow as tf
import tree
from dm_env import Environment, TimeStep, StepType
from acme.specs import EnvironmentSpec, make_environment_spec as acme_make_environment_spec, Array, BoundedArray, DiscreteArray
from acme.agents.tf.mcts import models

from tinyfive.multi_machine import multi_machine
from .config import AlphaDevConfig
from .utils import (
    TaskSpec,
    x86_enumerate_actions, x86_opcode2int, x86_signatures, x86_to_riscv, x86_source_source, x86_source_dest, x86_dest_source,
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
    def __init__(self, max_reg: int, max_mem: int, mode: Literal['u8', 'i32'] = 'i32',
                 init_active_registers: Optional[tf.Tensor] = None,
                 init_active_memory: Optional[tf.Tensor] = None):
        """
        Create an action space storage for x86 instructions.
        Args:
            max_reg: Maximum number of registers.
            max_mem: Maximum number of memory locations.
            mode: The mode to use for the instructions, either 'u8' or 'i32'.
            init_active_registers: Initial active registers mask, if available.
            init_active_memory: Initial active memory mask, if available.
        """
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
        self.init_stats()
        self._build_masks(init_active_registers=init_active_registers,
                          init_active_memory=init_active_memory)
        # create cache for the masks
        self._hash_base = tf.zeros(len(self.actions), dtype=tf.bool)
        self._mask_cache = {}
    
    def init_stats(self):
        self._stats = {
            'hashes': 0,
            'mask_calls': 0,
            'mask_history_hitmiss': [],
            'mask_updates': [],
            'mask_empty': 0,
            'mask_nonempty': 0,
        }
    
    def _hash_program(self, program: List[int]) -> str:
        self._stats['hashes'] = self._stats['hashes'] + 1
        bagged_program = np.zeros(len(self.actions), dtype=np.bool_)
        bagged_program[program] = True
        # convert to a bytearray
        return hash(bagged_program.tobytes())
    
    def _build_masks(self, init_active_registers, init_active_memory):
        """
        Build masks over the action space for each register and memory location.
        At runtime, we can dynamically take the union of a subset of these masks
        to efficiently mask the action space. 
        
        Each row in a mask is a boolean array over the action space, indicating whether
        the action uses the register or memory location. 
        """
        
        # for each action in the action space, create two sequences:
        # 1. a collection of indices that the action directly allows
        #   - these are the indices of actions which read from a register that this action writes to
        #   - or writes to a register that is the next register to which this action writes.
        # 2. a collection of indices that the action directly disallows
        #   - these are all the indices of actions that read from a memory location that this action reads from
        #   - or write to a memory location that this action writes to.
        reg_read_by_action = {i: set() for i in range(self.max_reg)}
        reg_written_by_action = {i: set() for i in range(self.max_reg)}
        lw_by_reg = {i: set() for i in range(self.max_reg)}
        lw_by_mem = {i: set() for i in range(self.max_mem)}
        sw_by_reg = {i: set() for i in range(self.max_reg)}
        sw_by_mem = {i: set() for i in range(self.max_mem)}
        action_ops = {} # destination register + memory location or -1 if either is not applicable
        for i, action in self.actions.items():
            opcode, operands = action
            if opcode == 'lw':
                mem, rd = operands[0]-self.max_reg, operands[1]
                action_ops[i] = (rd, mem) # store the action index and memory location
                lw_by_reg[rd].add((i, mem)) # store the action index and memory location
                lw_by_mem[mem].add((i, rd)) # store the action index and register
            elif opcode == 'sw':
                rs, mem = operands[0], operands[1]-self.max_reg
                action_ops[i] = (-1, mem) # store the action index and memory location
                sw_by_reg[rs].add((i, mem)) # store the action index and register
                sw_by_mem[mem].add((i, rs)) # store the action index and memory location
            elif opcode in x86_source_source:
                rs1, rs2 = operands
                maxreg = max(rs1, rs2)
                action_ops[i] = (-1, -1) # no memory location for source-source ops
                reg_read_by_action[maxreg].add(i)
            elif opcode in x86_source_dest:
                rs, rd = operands
                action_ops[i] = (rd, -1)
                if rd > rs:
                    reg_written_by_action[rd].add(i)
                elif rd <= rs: # equality only makes sense here, because writing is less restrictive
                    reg_read_by_action[rs].add(i)
            elif opcode in x86_dest_source:
                rd, rs = operands
                action_ops[i] = (max(rd, rs), -1)
                if rd > rs:
                    reg_written_by_action[rd].add(i)
                elif rd <= rs: # equality only makes sense here, because writing is less restrictive
                    reg_read_by_action[rs].add(i)
        
        allows_map = {i: set() for i in range(len(self.actions))}
        allows_lw = {}
        allows_lw_mem = {}
        allows_sw = {}
        allows_sw_mem = {}
        disallows_map = {i: set() for i in range(len(self.actions))}
        for i, action in self.actions.items():
            # a register only action allows all actions that read 
            # from the register it writes to and
            # all actions that write to the next register
            opcode, operands = action
            if opcode == 'lw':
                mem, rd = operands[0]-self.max_reg, operands[1]
                # lw allows actions that read from rd or write to rd+1
                allows_map[i] = reg_read_by_action[rd].union(
                    reg_written_by_action.get(rd + 1, set()) # avoid overflow
                )
                # it also allows sw ops for the current register
                # and lw ops for the next register
                # both of these need to be ordered by memory location
                allows_lw[i] = [act for act, _ in sorted(lw_by_reg.get(rd+1, []), key=lambda x: x[1])]
                allows_sw[i] = [act for act, _ in sorted(sw_by_reg[rd]          , key=lambda x: x[1])]
                # further, lw ops disallow all lw ops that read from the current memory
                disallows_map[i] = set([act for act, _ in lw_by_mem[mem]])
            elif opcode == 'sw':
                rs, mem = operands = operands[0], operands[1]-self.max_reg
                allows_map[i] = set() # sw does not allow any register reads
                # it does allow lw ops from mem and sw ops to mem+1
                allows_lw_mem[i] = [act for act, _ in sorted(lw_by_mem[mem]          , key=lambda x: x[1])]
                allows_sw_mem[i] = [act for act, _ in sorted(sw_by_mem.get(mem+1, []), key=lambda x: x[1])]
                # it also disallows all sw ops that write to the current memory
                disallows_map[i] = set([act for act, _ in sw_by_mem[mem]])
            elif opcode in x86_source_source:
                continue # no register or memory writes, so no allows or disallows either
            else:
                if opcode in x86_source_dest:
                    rs, rd = operands
                else:
                    rd, rs = operands
                allows_map[i] = reg_read_by_action[rd].union(
                    reg_written_by_action.get(rd + 1, set())
                )
                # it also allows lw ops for the next register
                # and sw ops for the current register
                # both of these need to be ordered by memory location 
                # so we can slice them later
                allows_lw[i] = [act for act, _ in sorted(lw_by_reg.get(rd+1, []), key=lambda x: x[1])]
                allows_sw[i] = [act for act, _ in sorted(lw_by_reg[rd]          , key=lambda x: x[1])]
                # reg-only actions do not disallow anything
        
        self._action_ops = action_ops
        self._allows_map = allows_map
        self._allows_lw = allows_lw
        self._allows_lw_mem = allows_lw_mem
        self._allows_sw = allows_sw
        self._allows_sw_mem = allows_sw_mem
        self._disallows_map = disallows_map

        # all elements allows_lw and allows_sw should be either num_regs or 0. if 0,we remove it
        for i in self.actions: 
            if i in allows_lw and len(allows_lw[i]) == 0: allows_lw.pop(i)
        assert all(len(v) == self.max_reg for v in allows_lw.values()), \
            "Not all lw actions have the same number of allowed actions."
        for i in self.actions: 
            if i in allows_sw and len(allows_sw[i]) == 0: allows_sw.pop(i)
        assert all(len(v) == self.max_reg for v in allows_sw.values()), \
            "Not all sw actions have the same number of allowed actions."
        
        # also, all elements in allows_lw_mem and allows_sw_mem should be either num_mem or 0
        for i in self.actions: 
            if i in allows_lw_mem and len(allows_lw_mem[i]) == 0: allows_lw_mem.pop(i)
        assert all(len(v) == self.max_mem for v in allows_lw_mem.values()), \
            "Not all lw_mem actions have the same number of allowed actions."
        for i in self.actions: 
            if i in allows_sw_mem and len(allows_sw_mem[i]) == 0: allows_sw_mem.pop(i)
        assert all(len(v) == self.max_mem for v in allows_sw_mem.values()), \
            "Not all sw_mem actions have the same number of allowed actions."

        # finally, compute initial masks
        # there are no actions so nothing is disallowed.
        # we need to allow all actions that read from the initial registers
        # and memory locations.
        # also all actions that write to the next register
        # or next memory location.
        last_reg = max(init_active_registers)
        last_mem = max(init_active_memory)
        mask = tf.zeros(len(self.actions), dtype=tf.bool)
        # take the union of the different lookup tables
        allowed_actions = set()
        # reg-only actions
        for reg in init_active_registers:
            # reg reads
            allowed_actions.update(reg_read_by_action[reg])
            # reg writes
            allowed_actions.update(reg_written_by_action.get(reg + 1, set()))
            # lw ops that write to next register
            lw_candidates = lw_by_reg.get(reg + 1, set())
            lw_candidates = [act for act, mem in lw_candidates if mem <= last_mem]
            allowed_actions.update(lw_candidates)
            # sw ops that read from current register and write to <= next memory
            sw_candidates = sw_by_reg[reg]
            sw_candidates = [act for act, mem in sw_candidates if mem <= last_mem+1]
            allowed_actions.update(sw_candidates)
            # sw|lw_by_mem contain the same actions as sw|lw_by_reg
        for mem in init_active_memory:
            # lw ops that read from current memory and write to next register
            lw_candidates = lw_by_mem[mem]
            lw_candidates = [act for act, reg in lw_candidates if reg <= last_reg+1]
            allowed_actions.update(lw_candidates)
            # sw ops that write to current memory and read from <= next register
            sw_candidates = sw_by_mem.get(mem + 1, set())
            sw_candidates = [act for act, reg in sw_candidates if reg <= last_reg]
            allowed_actions.update(sw_candidates)
        
        # update the mask
        allowed_actions = tf.constant(list(allowed_actions), dtype=tf.int32)
        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=tf.expand_dims(allowed_actions, axis=1),
            updates=tf.ones_like(allowed_actions, dtype=tf.bool)
        )
        self._base_mask = (mask, (last_reg, last_mem))
        # done.
        # summary
        # print("Done building initial mask.")
        # print(f"  allowed actions: {allowed_actions.shape}")
        # print(f"  active registers: {init_active_registers}")
        # print(f"  active memory: {init_active_memory}")
        # print(f"  register mask: >= {last_reg}, memory mask: <= {last_mem}")
        # print(f"allowed actions:")
        # for i, act in self.actions.items():
        #     isok = False
        #     if act[0] == 'lw':
        #         mem, rd = act[1][0]-self.max_reg, act[1][1]
        #         isok = mem <= last_mem and rd <= last_reg+1
        #     elif act[0] == 'sw':
        #         rs, mem = act[1][0], act[1][1]-self.max_reg
        #         isok = rs <= last_reg and mem <= last_mem+1
        #     elif act[0] in x86_source_source:
        #         rs, rs1 = act[1]
        #         isok = rs <= last_reg and rs1 <= last_reg
        #     else:
        #         rs, rd = act[1] if act[0] in x86_source_dest else (act[1][1], act[1][0])
        #         isok = rs <= last_reg and rd <= last_reg+1
        #     if mask[i]:
        #         print(f"{'OK ' if isok else 'NOK'}  {i}: {act} -> {self.asm_actions[i]} (np: {self.np_actions[i]})")
    
    def get_mask(self, history: List[int]) -> tf.Tensor:
        """
        Get the mask over the action space for the given state and history.

        Returns a boolean tensor over the action space, with True values indicating
        valid actions.
        """
        self._stats['mask_calls'] = self._stats['mask_calls'] + 1
        history = history.copy()  # don't modify the original history
        steps_to_eval = []
        prog_hash = self._hash_program(history)
        crnt_hash = prog_hash
        while len(history) > 0 and crnt_hash not in self._mask_cache:
            last_step = history.pop()
            new_hash = self._hash_program(history)
            if new_hash != crnt_hash:
                steps_to_eval.append(last_step)
            crnt_hash = new_hash
        if crnt_hash not in self._mask_cache:
            self._stats['mask_history_hitmiss'].append({'hit': 0, 'miss': len(steps_to_eval)})
            mask, (last_reg, last_mem) = self._base_mask
        else:
            self._stats['mask_history_hitmiss'].append({'hit': len(history), 'miss': len(steps_to_eval)} )
            mask, (last_reg, last_mem) = self._mask_cache[crnt_hash]
        # now  that we have the mask, we need to update it
        # being overly optimistic about the register and memory accesses is still a problem.
        # we need to compute register and memory masks separately and take their union
        allows = set()
        disallows = set()
        for step in steps_to_eval:
            step_reg, step_mem = self._action_ops[step] # mem=-1 if step is not a lw or sw
            # gather the allows and disallows for the step
            if step_mem > last_mem:
                # get LW acts that this step allows, filtered by the current memory
                allows.update([self._allows_lw[step][step_mem  ]] if step in self._allows_lw and step_mem < self.max_mem else [])
                allows.update([self._allows_sw[step][step_mem+1]] if step in self._allows_sw and step_mem+1 < self.max_mem else [])
                last_mem = step_mem
            if step_reg > last_reg:
                assert step not in x86_source_source,\
                    f"Source-to-source step _reads_ from register>last_reg, this should not happen"
                allows.update(self._allows_map[step])
                # get SW acts that this step allows, filtered by the current register
                allows.update([self._allows_lw_mem[step][step_reg  ]] if step in self._allows_lw_mem and step_reg < self.max_reg else [])
                allows.update([self._allows_sw_mem[step][step_reg+1]] if step in self._allows_sw_mem and step_reg+1 < self.max_reg else [])
                last_reg = step_reg
            # if step is an lw of sw, get the list of disallowed actions
            disallows = self._disallows_map.get(step, set())
        # update the mask with the allows and disallows
        update = allows - disallows
        self._stats['mask_updates'].append(len(update))
        update = tf.constant(list(update), dtype=tf.int32)
        if len(update) == 0:
            self._stats['mask_empty'] = self._stats['mask_empty'] + 1
            return mask
        self._stats['mask_nonempty'] = self._stats['mask_nonempty'] + 1
        # update the mask with the new allows
        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=tf.expand_dims(update, axis=1),
            updates=tf.ones_like(update, dtype=tf.bool)
        )
        # cache the mask
        self._mask_cache[prog_hash] = (mask, (last_reg, last_mem))
        return mask

    def get_space(self) -> ActionSpace:
        return self.action_space_cls(self.actions, self.asm_actions, self.np_actions)

    def npy_to_asm_int(self, npy_program):
        # convert the numpy program to a list of assembly instructions
        asm_program = []
        int_program = []
        for insn in npy_program:
            if (insn == 0).all():
                # reached the end of the program
                break
            insn_idx = self._npy_reversed.get(tuple(insn))
            asm_insn = self.asm_actions.get(insn_idx)
            asm_program.extend(asm_insn)
            int_program.append(insn_idx)
        return asm_program, int_program

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

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
            track_usage=False,
        )
        # TODO: make this distributed
        self._action_space_storage = x86ActionSpaceStorage(
            max_reg=task_spec.num_regs,
            max_mem=task_spec.num_mem,
            mode=task_spec.emulator_mode, # use 32-bit mode
            init_active_registers=[
                i for i, b in enumerate(self._emulator.register_mask[:, :task_spec.num_regs].any(axis=0)) if b != 0],
            init_active_memory=[
                i for i, b in enumerate(self._emulator.memory_mask.any(axis=0)) if b != 0]
        )
        self.reset()

    def _reset_program(self):
        self._program = Program(
            npy_program=np.zeros((self._task_spec.max_program_size, 3), dtype=np.int32),
            asm_program=[],
            int_program=[]
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
            # TODO: active registers and memory are no longer used.
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
            np_program = ts_program.numpy()
            asm_program, int_program = self._action_space_storage.npy_to_asm_int(np_program)
            self._program = Program(
                npy_program=np_program,
                asm_program=asm_program,
                int_program=int_program
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
            asm_program=self._program.asm_program + new_asm_program,
            int_program=self._program.int_program + actions
        )
        # reset the emulator
        self._emulator.reset_state()
        # execute the program
        self._emulator.exe(program=self._program.asm_program)
        # update observation and cached values
        # and return the updated timestep
        return self._update_state()
    
    def legal_actions(self) -> np.ndarray:
        tf_mask = self._action_space_storage.get_mask(self._program.int_program)
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
            # TODO: active registers and memory are no longer used.
            active_registers=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_regs), dtype=np.bool_),
            memory=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_mem), dtype=np.int32),
            active_memory=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_mem), dtype=np.bool_),
            program=Array(shape=(self._task_spec.max_program_size, 3), dtype=np.int32),
            program_length=Array(shape=(), dtype=np.int32),
            program_counter=Array(shape=(self._task_spec.num_inputs,), dtype=np.int32),
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
