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
from typing import List, Tuple, Dict
from itertools import product

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

def x86_to_riscv(x86_opcode, x86_operands):
    """
    Convert x86 opcode and operands to RISC-V opcode and operands.
    Args:
        x86_opcode (str): The x86 opcode.
        x86_operands (list): The x86 operands.
    Returns:
        list: A list of RISC-V instructions. Using the conversion described above
    """
    if x86_opcode == "mv": # move between registers
        return lambda _: [("ADD", (x86_operands[0], X0, x86_operands[1]))]
    elif x86_opcode == "lw": # load word from memory to register
        return lambda _: [("SW", (x86_operands[1], x86_operands[0]))]
    elif x86_opcode == "sw": # store word from register to memory
        return lambda _: [("LW", (x86_operands[0], x86_operands[1]))]
    elif x86_opcode == "cmp": # compare two registers
        return lambda _: [("SUB", (X1, x86_operands[0], x86_operands[1]))]
    elif x86_opcode == "cmovg": # conditional move if greater than
        return lambda state: [
            ("BLE", (X1, 0, state.pc+8)),  # skip next instruction if A < B
            ("ADD", (x86_operands[1], X0, x86_operands[0]))  # copy C to D
        ]
    elif x86_opcode == "cmovle": # conditional move if less than or equal
        return lambda state: [
            ("BGT", (X1, 0, state.pc+8)),  # skip next instruction if A > B
            ("ADD", (x86_operands[1], X0, x86_operands[0]))  # copy E to F
        ]
    else:
        raise ValueError(f"Unknown opcode: {x86_opcode}")

def enumerate_actions(max_reg: int, max_mem: int) -> List[Tuple[str, Tuple[int, int]]]:
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
