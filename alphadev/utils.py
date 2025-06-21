from typing import Any, Callable, List, NamedTuple, Tuple, Dict, Generator, Literal
import itertools
import numpy as np
from .tf_util import tf

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
    num_latency_simulations: int # number of latency simulations to run
    inputs: IOExample # input examples for the task
    penalize_latency: bool # whether to penalize latency in the reward or only when the program is correct.
    use_actual_latency: bool # use actual latency for the task, or use the program length as a proxy
    emulator_mode: Literal['u8', 'i16', 'i32'] = 'u8' # emulator mode for the task, one of ['u8', 'i16', 'i32']

class CPUState(NamedTuple):
    registers: tf.Tensor # num_inputs x num_regs array of register values
    memory: tf.Tensor # num_inputs x num_mem array of memory locations
    program: tf.Tensor # max_program_size x 1 array of progrram instructions.
    program_length: tf.Tensor  # scalar length of the program 

class Program(NamedTuple):
    npy_program: np.ndarray # <max_program_size> x 3 (opcode, op1, op2)
    asm_program: List[Callable[[int], Any]] # list of pseudo-asm instructions
    int_program: List[int] 
    def __len__(self):
        return len(self.int_program)

# #################
# Input geneerators
# #################

def generate_sort_inputs(
    items_to_sort: int, max_len:int, num_samples: int=None) -> IOExample:
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
    
    return IOExample(
        inputs=i_list.astype(np.int32), # num_inputs x items_to_sort
        outputs=o_list.astype(np.int32), # num_inputs x max_len
        output_mask=o_mask.astype(np.bool_) # num_inputs x max_len boolean array masking irrelevant parts of the output,
    )

# #################
# x86 to RISC-V conversion
# #################

X0 = 0 # hard-wired zero register
X1 = 1 # reserved for comparison ops
REG_T = 10
MEM_T = 11
IMM_T = 12

def x86_to_riscv(opcode: str, operands: Tuple[int, int], mem_offset, mode:Literal['u8', 'i16', 'i32']) -> Tuple[str, Callable[[int], Tuple[int, int]]]:
    """
    Convert an x86 (pseudo-) instruction to a RISC-V (pseudo-) instruction.
    """
    blen = 4 if mode == 'u8' else 2 if mode == 'i16' else 1
    
    if opcode == "mv": # move between registers
        return [("ADD", (operands[1], X0, operands[0]))]
    elif opcode == "lw": # load word from memory to register
        return [("LW", (operands[1], (operands[0]-mem_offset)*blen, X0))]
    # rd,imm,rs -- rd, rs(imm)
    elif opcode == "sw": # store word from register to memory
        return [("SW", (operands[0], (operands[1]-mem_offset)*blen, X0))] 
    # rs1,imm,rs2 -- rs1, rs2(imm)
    elif opcode == "cmp": # compare two registers
        return [("SUB", (X1, operands[0], operands[1]))]
        # if A > B, then X1 > 0
        # if A < B, then X1 < 0
        # riscv has bge (>=) and blt (<) instructions
    elif opcode == "cmovg": # conditional move if greater than
        return [ # jump if A <= B; move otherwise
            ("BGE", (X0, X1, 8)), # X1 = A - B ; 0 >= A - B ; B >= A
            ("ADD", (operands[1], X0, operands[0])) # copy C to D
        ]
    elif opcode == "cmovge": # conditional move if greater than
        return [ # jump if A < B; move otherwise
            ("BLT", (X1, X0, 8)), # X1 = A - B ; A - B < 0 ; A < B
            ("ADD", (operands[1], X0, operands[0])) # copy C to D
        ]
    elif opcode == "cmovl": # conditional move if less than
        return [ # jump if A >= B; move otherwise
            ("BGE", (X1, X0, 8)), # X1 = A - B ; A - B >= 0 ; A >= B
            ("ADD", (operands[1], X0, operands[0])) # copy C to D
        ]
    elif opcode == "cmovle": # conditional move if less than or equal
        return [ # jump if A > B; move otherwise
            ("BLT", (X0, X1, 8)), # X1 = A - B ; 0 < A - B ; B < A
            ("ADD", (operands[1], X0, operands[0])) # copy C to D
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
    "cmovge" : (REG_T, REG_T),
    "cmovle" : (REG_T, REG_T),
    "cmovl" : (REG_T, REG_T),
    # skip jump instructions, they are not used in the published sort algorithms
    }

x86_opcode2int = { # opcodes should be 1-indexed, so that we can use 0 as a no-op
    k: i+1 for i, k in enumerate(x86_signatures.keys())
}

x86_source_source = {'cmp'}
x86_source_dest = {'mv', 'lw', 'sw', 'cmovg', 'cmovge', 'cmovle', 'cmovl'}
x86_dest_source = {}

def x86_enumerate_actions(max_reg: int, max_mem: int) -> Dict[int, Tuple[str, Tuple[int, int]]]:
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
