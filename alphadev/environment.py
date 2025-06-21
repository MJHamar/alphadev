from typing import Dict, Any, Tuple, List, Callable, Union, Optional, Literal
import functools
import numpy as np
from dm_env import Environment, TimeStep, StepType
from acme.specs import EnvironmentSpec, make_environment_spec as acme_make_environment_spec, Array, BoundedArray, DiscreteArray
from acme.agents.tf.mcts import models
import tree

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

    def get_mask(self, state, history:list=None) -> np.ndarray:
        """
        Get the mask over the action space for the given state and history.
        
        Returns a boolean array over the action space, with True values indicating
        valid actions.
        """
        raise NotImplementedError()

    def npy_to_asm(self, npy_program: np.ndarray) -> List[Callable[[int], Any]]:
        raise NotImplementedError()

class x86ActionSpaceStorage(ActionSpaceStorage):
    def __init__(self, max_reg: int, max_mem: int, mode: Literal['u8', 'i32'] = 'i32',
                 init_active_registers: List[int] = None,
                 init_active_memory: List[int] = None):
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
        self._build_masks(init_active_registers=init_active_registers,
                          init_active_memory=init_active_memory)
        # create cache for the masks
        self._mask_cache = {}
        self._mask_max_size = 1000000
        self.init_stats()
    
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
        return bagged_program.tobytes()
    
    def _build_masks(self, init_active_registers, init_active_memory):
        """
        Build masks over the action space for each register and memory location.
        At runtime, we can dynamically take the union of a subset of these masks
        to efficiently mask the action space. 
        
        Each row in a mask is a boolean array over the action space, indicating whether
        the action uses the register or memory location. 
        """
        if init_active_registers is None:
            init_active_registers = [0,1] # X0 and the next one.
        if init_active_memory is None:
            init_active_memory = [0]
        
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
        print('max_reg', self.max_reg, "max_mem", self.max_mem)
        assert all(len(v) == self.max_mem for v in allows_lw.values()), \
            f"Not all lw actions have the same number of allowed actions. {[len(v) for v in allows_lw.values()]}"
        for i in self.actions: 
            if i in allows_sw and len(allows_sw[i]) == 0: allows_sw.pop(i)
        assert all(len(v) == self.max_mem for v in allows_sw.values()), \
            f"Not all sw actions have the same number of allowed actions. {[len(v) for v in allows_sw.values()]}"
        
        # also, all elements in allows_lw_mem and allows_sw_mem should be either num_mem or 0
        for i in self.actions: 
            if i in allows_lw_mem and len(allows_lw_mem[i]) == 0: allows_lw_mem.pop(i)
        assert all(len(v) == self.max_reg for v in allows_lw_mem.values()), \
            "Not all lw_mem actions have the same number of allowed actions."
        for i in self.actions: 
            if i in allows_sw_mem and len(allows_sw_mem[i]) == 0: allows_sw_mem.pop(i)
        assert all(len(v) == self.max_reg for v in allows_sw_mem.values()), \
            "Not all sw_mem actions have the same number of allowed actions."

        # finally, compute initial masks
        # there are no actions so nothing is disallowed.
        # we need to allow all actions that read from the initial registers
        # and memory locations.
        # also all actions that write to the next register
        # or next memory location.
        last_reg = max(init_active_registers)
        last_mem = max(init_active_memory)
        mask = np.zeros(len(self.actions), dtype=np.bool_)
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
        allowed_actions = list(allowed_actions)
        mask[allowed_actions] = True
        self._base_mask = (mask, (last_reg, last_mem))
        
    def get_mask(self, history: List[int]) -> np.ndarray:
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
            mask, (last_reg, last_mem) = self._base_mask[0].copy(), self._base_mask[1]
        else:
            self._stats['mask_history_hitmiss'].append({'hit': len(history), 'miss': len(steps_to_eval)} )
            mask, (last_reg, last_mem) = self._mask_cache[crnt_hash][0].copy(), self._mask_cache[crnt_hash][1]
        # now  that we have the mask, we need to update it
        # being overly optimistic about the register and memory accesses is still a problem.
        # we need to compute register and memory masks separately and take their union
        allows = set()
        disallows = set()
        for step in steps_to_eval:
            step_reg, step_mem = self._action_ops[step] # mem=-1 if step is not a lw or sw
            # gather the allows and disallows for the step
            new_last_reg = max(last_reg, step_reg) # lookahead.
            if step_mem > last_mem:
                # get LW acts that this step allows, filtered by the current memory
                last_mem = step_mem
                allows.update(self._allows_lw_mem[step][:new_last_reg+1] if step in self._allows_lw_mem else [])
                allows.update(self._allows_sw_mem[step][:new_last_reg+2] if step in self._allows_sw_mem else [])
            if step_reg > last_reg:
                assert step not in x86_source_source,\
                    f"Source-to-source step _reads_ from register>last_reg, this should not happen"
                allows.update(self._allows_map[step])
                # get SW acts that this step allows, filtered by the current register
                allows.update(self._allows_lw[step][:last_mem+1] if step in self._allows_lw else [])
                allows.update(self._allows_sw[step][:last_mem+2] if step in self._allows_sw else [])
                last_reg = step_reg
            # if step is an lw of sw, get the list of disallowed actions
            disallows = self._disallows_map.get(step, set())
        # update the mask with the allows and disallows
        update = allows - disallows
        update = list(update)
        self._stats['mask_updates'].append(len(update))
        if len(update) == 0:
            self._stats['mask_empty'] = self._stats['mask_empty'] + 1
            return mask
        self._stats['mask_nonempty'] = self._stats['mask_nonempty'] + 1
        # update the mask with the new allows
        mask[update] = True
        # cache the mask
        # make sure we don't exceed the bound
        if len(self._mask_cache) >= self._mask_max_size:
            # remove the oldest mask
            oldest_key = next(iter(self._mask_cache))
            del self._mask_cache[oldest_key]
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
        self._max_num_hits = np.sum(self._output_mask.astype(np.int32))
        # if this is True, we add a negative reward for latency.
        # the cumulative latency penalty for a program of length L is L*latency_reward_weight.
        # instead of adding it to the reward in each step, we add it at the end of the episode,
        # regardless of the correctess.
        self._penalize_latency = task_spec.penalize_latency
        self.latency_reward = 0.0 # latency reward is 0.0 unless the program is correct.
        
        self._emulator = multi_machine(
            mem_size=task_spec.num_mem,
            num_machines=task_spec.num_inputs,
            initial_state=self._inputs,
            # TODO: this is the hard-coded X1 register for now.
            special_x_regs=np.array([1], dtype=np.int32),
            mode=task_spec.emulator_mode, # use 32-bit mode
            track_usage=False,
            use_fp=False
        )
        # TODO: make this distributed
        self._action_space_storage = x86ActionSpaceStorage(
            max_reg=task_spec.num_regs,
            max_mem=task_spec.num_mem,
            mode=task_spec.emulator_mode, # use 32-bit mode
            init_active_registers=[0,1],
            init_active_memory=list(range(task_spec.inputs.inputs.shape[1]))
        )
        self.reset()

    def _reset_program(self):
        self._program = Program(
            npy_program=np.zeros((self._task_spec.max_program_size, 3), dtype=np.int32),
            asm_program=[],
            int_program=[]
        )
    
    def _eval_output(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        masked_output = np.multiply(output, self._output_mask) # mask out the parts we don't care about
        hits = masked_output == self._outputs # check which outputs match the expected outputs
        num_hits = hits.sum()
        all_hits = num_hits == self._max_num_hits
        return all_hits, num_hits
    
    def _eval_latency(self) -> np.ndarray:
        """Returns a scalar latency for the program."""
        # NOTE: not used.
        latencies = np.asarray([
                self._emulator.measure_latency(self._program.asm_program)
                for _ in range(self._task_spec.num_latency_simulations)
            ],
            dtype=np.float32
        )
        return latencies
    
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
        # update the previous number of correct items.
        self._prev_num_hits = self._num_hits
        
        if include_latency: # cannot be <0 btw
            if not self._task_spec.use_actual_latency:
                # this is more efficient and proportional to the original latency calculation
                # since we only consider branchless programs
                # TODO: for branching programs this will no longer do.
                self.latency_reward = self._task_spec.latency_reward_weight * len(self._program)
            else:
                latencies = self._eval_latency()
                self.latency_reward = np.quantile(
                    latencies, self._task_spec.latency_quantile
                ) * self._task_spec.latency_reward_weight
        
        return correctness_reward
    
    def _make_observation(self) -> Dict[str, np.ndarray]:
        # get the current state of the CPU
        return CPUState(
            registers= self._emulator.registers[:, :self._task_spec.num_regs].astype(np.int32),
            # TODO: active registers and memory are no longer used.
            memory= self._emulator.memory.astype(np.int32),
            program= self._program.npy_program.astype(np.int32),
            program_length= np.asarray(len(self._program), dtype=np.int32),
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
        
        # we can now compute the reward. only calculate latency if the program is correct.
        # according to the pseudocode and Mankowicz et al. 2023.
        reward = self._compute_reward(include_latency=self._is_correct or self._penalize_latency)
        
        step_type = StepType.FIRST if len(self._program) == 0 else (
                        StepType.MID if not is_terminal else
                            StepType.LAST)
        ts = TimeStep(
            step_type=step_type,
            reward=np.array(reward, dtype=np.float32),
            discount=np.asarray(1.0, dtype=np.float32), # NOTE: not sure what discount here means.
            observation=observation,
            # NOTE: we add latency reward in the 'extras' field.
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
                npy_program = state.observation['program']
            else: # then it is a CPUState._asdict()
                npy_program = state['program']
            # logger.debug("AssemblyGame.reset: npy_program shape %s", npy_program.shape)
            # either B x num_inputs x 3 or no batch dimension
            if len(npy_program.shape) > 2:
                # we need to remove the batch dimension
                assert npy_program.shape[0] == 1, "Batch dimension is not 1, resetting is ambigouous."
                npy_program = np.squeeze(npy_program, axis=0)

            # convert the numpy program to a list of assembly instructions
            asm_program, int_program = self._action_space_storage.npy_to_asm_int(npy_program)
            self._program = Program(
                npy_program=npy_program,
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
        if isinstance(actions, (int, np.int32, np.int64)):
            # single action
            actions = [int(actions)]
            # single action as numpy int32
        elif not isinstance(actions, (list)):
            raise TypeError("Actions must be a list of integers or a single integer, not %s" % type(actions))
        # check if the actions are valid
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
        legal_actions = self._action_space_storage.get_mask(self._program.int_program)
        return legal_actions
    
    def reward_spec(self):
        return Array(shape=(), dtype=np.float32)
    def discount_spec(self):
        return Array(shape=(), dtype=np.float32)
    def observation_spec(self):
        return CPUState(
            registers=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_regs), dtype=np.int32),
            # TODO: active registers and memory are no longer used.
            memory=Array(shape=(self._task_spec.num_inputs, self._task_spec.num_mem), dtype=np.int32),
            program=Array(shape=(self._task_spec.max_program_size, 3), dtype=np.int32),
            program_length=Array(shape=(), dtype=np.int32),
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
        new_game.latency_reward = self.latency_reward
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
        self._ckpt = None
    
    def load_checkpoint(self):
        """Loads a saved model state, if it exists."""
        self._needs_reset = False
        self._environment = self._ckpt.copy()

    def save_checkpoint(self):
        """Saves the model state so that we can reset it after a rollout."""
        self._ckpt = self._environment.copy()
    
    def make_checkpoint(self) -> AssemblyGame:
        """Returns the current checkpoint of the model."""
        self.save_checkpoint() # update the checkpoint.
        return self._ckpt
    
    def set_checkpoint(self, ckpt: AssemblyGame):
        self._ckpt = ckpt
    
    def get_checkpoint_size(self) -> int:
        import pickle
        env = self._environment
        space_storage = env._action_space_storage
        # the size of the checkpoint is the size of timesteps + maximum size of the mask cache
        # randomly sample a program
        prog = np.random.choice(np.asarray(list(space_storage.actions.keys())), size=env._task_spec.max_program_size, replace=True)
        # execute the program and get a timestep
        ts = env.step(prog.tolist())
        _ = env.legal_actions()
        mask_cache = space_storage._mask_cache
        
        ts_size = len(pickle.dumps(ts))
        mask_size = len(pickle.dumps(mask_cache))
        # reset the env
        env.reset()
        
        mask_size *= space_storage._mask_max_size
        print('Mask max size', space_storage._mask_max_size)
        print('TS size', ts_size, 'mask size', mask_size)
        print('Checkpoint size:', ts_size + mask_size)
        return ts_size + mask_size
    
    def update(
        self,
        timestep: TimeStep, # prior to executing the action
        action: np.ndarray, # opcode, operands
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
            return (
                timestep.observation['program'] == self._environment._last_ts.observation['program']
            ).all()
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
    
    def get_latency_reward(self) -> float:
        return np.array(self._environment.latency_reward, dtype=np.float32)

class EnvironmentFactory:
    def __init__(self, config: AlphaDevConfig): self._task_spec = config.task_spec
    def __call__(self): return AssemblyGame(task_spec=self._task_spec)

class ModelFactory:
    def __init__(self, config: AlphaDevConfig): self._task_spec = config.task_spec
    def __call__(self, env_spec: EnvironmentSpec): return AssemblyGameModel(task_spec=self._task_spec, name='AssemblyGameModel')

def environment_spec_from_config(config: AlphaDevConfig) -> EnvironmentSpec:
    return acme_make_environment_spec(EnvironmentFactory(config)())
