"""
In the sorting programs presented in the AplhaDev paper (Mankowitz et. al. 2023), we see the following x86 instructions:
- mov    - move register to register, register to memory, or memory to register
- cmp    - compare two registers and set register flags 
- cmovg  - conditional move if greater than
- cmovl  - conditional move if less than
- cmovge - conditional move if greater than or equal
- cmovle - conditional move if less than or equal
- jne    - jump if not equal
- je     - jump if equal
- j      - jump unconditionally

In addition, we know that memory and registers are accessed in incremental order.

To enumerate all possible actions, we use a recursion over three running indices:
- reg1
- reg2
- mem
For each (reg1, reg2, mem) tuple, we further enumerate all actions that are valid.

The recursive function is defined as follows:

J(r1, r2, m) = J(r-1, r2, m) . J(r1, r2-1, m) . J(r1, r2, m-1) . (r1, r2, m) | (1,1,1) [if r1 = r2 = m = 1]
Where . denotes concatenation.

Since we do not have access to a lightweight and fast x86 simulator, we use RISC-V instructions instead.
Since RISC-V is a load-store architecture, the set of opcodes is a bit different
but the number of actions should stay the same.

Relevant RISC-V instructions:
# MOV instructions
- lw  rd,  imm, rs   - load word from memory to register
- sw  rs2, imm, rs1  - store word from register to memory
- add rd,  x0,  rs   - copy register rs to register rd
# branch instructions. note that swapping rs1 and rs2 achieves the opposite effect
- blt rs1, rs2, label  - branch if less than (rs2 rs1 -- branch greater than)
- bge rs1, rs2, label  - branch if greater than or equal (rs2 rs1 -- branch less than or equal)


Note that RISC-V does not support conditional moves, which we overcome with the following macro
Accordingly, to efficiently implement 
    cmp A, B
    # conditional move if greater than
    cmovg  C, D
    # conditional move if less than or equal
    cmovlt E, F
x86 constructs (heavily used in AlphaDev sort algorithms), we use the following RISC-V macro:
    sub x1,  A, B       # cmp A, B (if A < B, x1 < 0)
    # conditional move if greater than
    ble x1,  0, pc+8    # skip next instruction if A < B
    add D,  x0, C       # copy C to D
    # conditional move if less than or equal
    bgt x1,  0, pc+8    # skip next instruction if A > B
    add F,  x0, E       # copy E to F

finally, to ensure no funny business, we require that the x1 register is not used for anything else.

This set of instructions, along with the enumerated memory and register locations should be 
sufficient to enumerate all possible actions to reproduce the AlphaDev fixed sort programs.

Since RISC-V uses relative addressing, but memory locations are absolute, we express memory locations as
imm = location; rsX = x0 (hard-coded zero register);

"""
from typing import List, Tuple, Dict, Callable, Any, NamedTuple
from itertools import product

import jax.numpy as jnp

X0 = 0
X1 = 1 # reserved for comparison
REG_T = 10
MEM_T = 11

riscv_valid_registers = [X0] + list(range(2, 32)) # x0 is hard-coded zero register

# define the x86 instruction set used in AlphaDev fixed sort programs
x86_signatures = {
    "mv" : (REG_T, REG_T),
    "lw" : (MEM_T, REG_T),
    "sw" : (REG_T, MEM_T),
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

# TODO: action space, such that the emulator and the space reference each other.
# the emulator should call ActionSpace.get(action) to get the action
# the action space can use the emulator to get the action
# the action space also needs to define a masking function that masks invalid actions based on the emulator state
class x86ActionSpace:
    def __init__(self,
            actions: List[Callable[[Any],Tuple[str, Tuple[int, int]]]],
            state: 'CPUState'):
        self.actions = actions
        self.state = state
        
    def get(self, action_id: int) -> List[RiscvAction]:
        return x86_to_riscv(*self.actions[action_id], self.state) # convert to RISC-V
    
class x86ActionSpaceStorage:
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