
import numpy as np
from tinyfive.machine import machine
from tinyfive.multi_machine import multi_machine, pseudo_asm_machine
from alphadev.utils import x86_to_riscv, x86_enumerate_actions


sort3_x86_asm = [
    "mov 0x4(%0), %%eax ".strip(),
    "mov 0x8(%0), %%ecx ".strip(),
    "cmp %%eax, %%ecx   ".strip(),
    "mov %%eax, %%edx   ".strip(),
    "cmovl %%ecx, %%edx ".strip(),
    "mov (%0), %%r8d    ".strip(),
    "cmovg %%ecx, %%eax ".strip(),
    "cmp %%r8d, %%eax   ".strip(),
    "mov %%r8d, %%ecx   ".strip(),
    "cmovl %%eax, %%ecx ".strip(),
    "cmovle %%r8d, %%eax".strip(),
    "mov %%eax, 0x8(%0) ".strip(),
    "cmp %%ecx, %%edx   ".strip(),
    "cmovle %%edx, %%r8d".strip(),
    "mov %%r8d, (%0)    ".strip(),
    "cmovg %%edx, %%ecx ".strip(),
    "mov %%ecx, 0x4(%0) ".strip(),
]
sort3_riscv_asm = [
    "lw  x2  4 x0", # @0  mov 0x4(%0), %%eax  x2 = mem[0x4]
    "lw  x3  8 x0", # @4  mov 0x8(%0), %%ecx  x3 = mem[0x8]
    "sub x1 x2 x3", # @8  cmp %%eax, %%ecx    x1 = x2 - x3  -|- >0 if x2 > x3; <=0 if x2 <= x3
    "add x4 x0 x2", # @12 mov %%eax, %%edx    x4 = x2 + 0
    "bge x1 x0 24", # @16 cmovl %%ecx, %%edx  jump if ( x2-x3 ) >= 0 [x2 greater than x3]
    "add x4 x0 x3", # @20                     move if ( x2-x3 ) <  0 [x2 less    than x3]
    "lw  x5  0 x0", # @24 mov (%0), %%r8d     x5 = mem[0x0]
    "bge x0 x1 36", # @28 cmovg %%ecx, %%eax  jump if ( x2-x3 ) <= 0 [x2 less    than x3]
    "add x2 x0 x3", # @32                     move if ( x2-x3 ) >  0 [x2 greater than x3]
    "sub x1 x5 x2", # @36 cmp %%r8d, %%eax    x1 = x5 - x2
    "add x3 x0 x5", # @40 mov %%r8d, %%ecx    x3 = x5 + 0
    "bge x1 x0 52", # @44 cmovl %%eax, %%ecx  jump if ( x5-x2 ) >= 0 [x5 greater than x2]
    "add x3 x0 x2", # @48                     move if ( x5-x2 ) <  0 [x5 less    than x2] 
    "blt x0 x1 60", # @52 cmovle %%r8d, %%eax jump if ( x5-x2 ) >  0 [x5 greater than x2]
    "add x2 x0 x5", # @56                     move if ( x5-x2 ) <= 0 [x5 less    than x2]
    "sw  x2  8 x0", # @60 mov %%eax, 0x8(%0)  mem[0x8] = x2
    "sub x1 x3 x4", # @64 cmp %%ecx, %%edx    x1 = x3 - x4  -|- >0 if x3 > x4; <=0 if x3 <= x4
    "blt x0 x1 76", # @68 cmovle %%edx, %%r8d jump if ( x3-x4 ) >  0 [x3 greater than x4]
    "add x5 x0 x4", # @72                     move if ( x3-x4 ) <= 0 [x3 less    than x4]
    "sw  x5  0 x0", # @76 mov %%r8d, (%0)     mem[0x0] = x5
    "bge x0 x1 88", # @80 cmovg %%edx, %%ecx  jump if ( x3-x4 ) <= 0 [x3 less    than x4]
    "add x3 x0 x4", # @84                     move if ( x3-x4 ) >  0 [x3 greater than x4]
    "sw  x3  4 x0", # @88 mov %%ecx, 0x4(%0)
]
sort3_riscv_asm_split = []
for insn in sort3_riscv_asm:
    insn = insn.split()
    op, operands = insn[0], insn[1:]
    # convert the operands to integers
    operands = [int(op[1:]) if op[0] == 'x' else int(op) for op in operands]
    sort3_riscv_asm_split.append([op, *operands])

# there are 13 possible orderings of 3 numbers with replacement.
# the above algorithm sorts in descending order.
test_cases = np.array([
    [[1,1,1],[1,1,1]],
    [[1,1,2],[2,1,1]],
    [[1,2,1],[2,1,1]],
    [[2,1,1],[2,1,1]],
    [[1,2,2],[2,2,1]],
    [[2,1,2],[2,2,1]],
    [[2,2,1],[2,2,1]],
    [[1,2,3],[3,2,1]],
    [[1,3,2],[3,2,1]],
    [[2,1,3],[3,2,1]],
    [[2,3,1],[3,2,1]],
    [[3,1,2],[3,2,1]],
    [[3,2,1],[3,2,1]]
], dtype=np.int32)

# test case 1 -- testing the target program
def test_target_correct_1():
    # test ASM on a single machine
    m = machine(1024) # allocate 1024 bytes of memory
    m.pc = 0 # make sure program counter is at 0
    # write the program to memory from pc=0
    for insn in sort3_riscv_asm_split:
        op, operands = insn[0], insn[1:]
        # bit of a hack, but for this test case, we use x9 as the base instead of x0
        # this is needed because m.asm() writes the instruction to memory
        # so the memory region we are interested in is offset by the length of the program
        if op == 'lw' or op == 'sw':
            operands[-1] = 9
        # convert the instruction to a machine code instruction
        print('Writing insn', insn, 'as', op, operands, 'to pc', m.pc)
        m.asm(op, *operands)
    # input starts after the program
    m.x[9] = m.pc
    for i in range(test_cases.shape[0]):
        inp, out = test_cases[i,0], test_cases[i,1]
        # print(f"Test case {i}: input {inp}, expected output {out}")
        # write the input to memory
        m.write_i32_vec(inp, m.pc)
        # run the program
        m.exe(start=0, end=m.pc)
        # read the output from memory
        outp = m.read_i32_vec(m.pc, 3)
        # check if the output is correct
        assert np.array_equal(outp, out), f"Test case {i} failed: expected {out}, got {outp}"
    print("Test case 1 passed")

# convert the absolute offsets to relative offsets
# in branching instructions to make them covariant to the current pc
sort3_riscv_asm_rel = []
for insn in sort3_riscv_asm_split:
    op, operands = insn[0], insn[1:]
    if op.startswith('b'): #bge or blt, the last operand is the offset. always 8
        operands = [operands[0], operands[1], 8]
    sort3_riscv_asm_rel.append([op, *operands])

# manually convert the riscv code to callables that the machine expects
program = []
for insn in sort3_riscv_asm_rel:
    op, operands = insn[0], insn[1:]
    program.append((op.upper(), operands))

# test case 2 -- test a pseudo instruction machine 
def test_pseudo_asm_machine():
    m = pseudo_asm_machine(1024)
    # input starts at 0
    for i in range(test_cases.shape[0]):
        inp, out = test_cases[i,0], test_cases[i,1]
        # print(f"Test case {i}: input {inp}, expected output {out}")
        # write the input to memory
        m.write_i32_vec(inp, 0)
        # run the program
        m.exe(program=program)
        # read the output from memory
        outp = m.read_i32_vec(0, 3)
        # check if the output is correct
        assert np.array_equal(outp, out), f"Test case {i} failed: expected {out}, got {outp}"
    print("Test case 2 passed")

# test case 3 -- test a multi machine
def test_multi_machine():
    m = multi_machine(1024, 13, test_cases[:,0,:])
    # execute the program on all inputs at once
    m.exe(program=program)
    # read the output from memory
    outp = m.memory[:, :3]
    # check if the output is correct
    assert np.array_equal(outp, test_cases[:,1,:])
    print("Test case 3 passed")

# test case 4 -- test translation of x86 to riscv
x86_action_space = x86_enumerate_actions(6,5) # 5 regs and 5 mem locattions are more than enough
def test_x86_to_riscv():
    riscv_action_space = []
    for x86_action, x86_operands in x86_action_space.values():
        riscv_action_space.extend(x86_to_riscv(x86_action, x86_operands, 6))
    # check if all the actions in the test case are present in the riscv action space
    for insn in sort3_riscv_asm_rel:
        op, operands = insn[0], insn[1:]
        # check if the instruction is in the action space
        assert (op.upper(), tuple(operands)) in riscv_action_space, f"Instruction {insn} not found in action space"
    print("Test case 4 passed")

if __name__ == "__main__":
    test_target_correct_1()
    test_pseudo_asm_machine()
    test_multi_machine()
    test_x86_to_riscv()