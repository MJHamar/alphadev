
# #################
# network definition
# #################
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Tuple

import sonnet as snn
from .tf_util import tf

import ml_collections
import functools
from acme.specs import Array, DiscreteArray
from dm_env import Environment
import tree


from .config import AlphaDevConfig
from .utils import TaskSpec, CPUState
from .distribution import DistributionSupport

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np

def positional_encoding(length, depth):
    """Adapted from https://www.tensorflow.org/text/tutorials/transformer"""
    logger.warning("[re]computing positional encoding for length %d and depth %d (only a concern if happens repeatedly)", length, depth)
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(snn.Module):
    """Adapted from https://www.tensorflow.org/text/tutorials/transformer"""
    def __init__(self, seq_size, feat_size):
        super().__init__()
        self.d_model = feat_size
        self.length = seq_size
        self.pos_encoding = positional_encoding(length=seq_size, depth=feat_size)

    def __call__(self, x):
        tf.debugging.assert_all_finite(x, "PositionalEmbedding input x")
        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        tf.debugging.assert_all_finite(x, "PositionalEmbedding after sqrt scaling")
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        tf.debugging.assert_all_finite(x, "PositionalEmbedding output")
        return x

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
        
        self.P_q = snn.Linear(self.num_heads * self.head_depth, name='P_q')
        self.P_k = snn.Linear(self.head_depth, name='P_k')
        self.P_v = snn.Linear(self.head_depth, name='P_v')
        self.P_o = snn.Linear(self.num_heads * self.head_depth, name='P_o')
        if self.attention_dropout:
            self.attention_dropout = snn.Dropout(self.attention_dropout, name='attn_dropout')
        else:
            self.attention_dropout = None
    
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
        *leading_dims, _ = inputs.shape # B x N x D
        # logger.debug("MQAB: inputs shape %s", inputs.shape)
        # P_q, P_k, P_v, P_o are parameters, which we declare here
        Q = self.P_q(inputs)
        # logger.debug("MQAB: Q shape %s, reshaping to %s", Q.shape, (*leading_dims, self.num_heads, self.head_depth))
        Q = tf.reshape(Q, (*leading_dims, self.num_heads, self.head_depth)) # B x N x H x K
        K = self.P_k(inputs)
        K = tf.reshape(K, (*leading_dims, self.head_depth)) # B x M x K
        V = self.P_v(inputs)
        V = tf.reshape(V, (*leading_dims, self.head_depth)) # B x M x V
        
        logits = tf.einsum("bnhk,bmk->bhnm", Q, K) # B x N x H x M
        weights = tf.nn.softmax(logits) # NOTE: no causal masking, this is an encoder block
        if self.attention_dropout: # boolean
            weights = snn.Dropout(self.attention_dropout, name='attn_dropout')(weights)
        O = tf.einsum("bhnm,bmv->bhnv", weights, V) # B x N x H x V
        # apply the output projection
        # logger.debug("MQAB: O shape %s, reshaping to %s", O.shape, (*leading_dims, self.num_heads * self.head_depth))
        O = tf.reshape(O, (*leading_dims, self.num_heads * self.head_depth)) # B x N x H*V
        Y = self.P_o(O) # B x N x V
        
        assert Y.shape == inputs.shape,\
            f"Output shape {Y.shape} does not match input shape {inputs.shape}."
        
        return Y # B x N x D
    
    def sinusoid_position_encoding(self, seq_size, feat_size):
        """Compute sinusoid absolute position encodings, 
        given a sequence size and feature dimensionality"""
        return self.encoder(seq_size, feat_size)

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
    # bin_mask = np.tile(2 ** np.arange(32), (flat_arr.shape[0], 1))
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
        
        seq_size, feat_size = task_spec.max_program_size, embedding_dim
        self.positional_embedding = PositionalEmbedding(seq_size, feat_size)
        
        self.program_mlp_embedder = snn.Sequential(
            [
                snn.Linear(self._embedding_dim), # (nF + 2*nL) x D -- input size is decided automatically
                snn.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                tf.nn.relu,
                snn.Linear(self._embedding_dim), # D x D
            ],
            name='per_instruction_program_embedder',
        )
        attention_params = self._hparams.representation.attention
        make_attention_block = functools.partial(
            MultiQueryAttentionBlock, attention_params
        )
        self.attention_encoders = snn.Sequential([
            make_attention_block(name=f'attention_program_sequencer_{i}')
            for i in range(self._hparams.representation.attention.num_layers)
        ], name='program_attention')
        
        self.locations_embedder = snn.Sequential(
            [
                # input is embedding_dim size, because we already encoded in either one-hot or binary
                snn.Linear(self._embedding_dim),
                snn.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                tf.nn.relu,
                snn.Linear(self._embedding_dim),
            ],
            name='per_locations_embedder',
        )
        self.all_locations_net = snn.Sequential(
            [
                snn.Linear(self._embedding_dim),
                snn.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                tf.nn.relu,
                snn.Linear(self._embedding_dim),
            ],
            name='per_element_embedder',
        )
        self.joint_locations_net = snn.Sequential(
            [
                snn.Linear(self._embedding_dim),
                snn.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                tf.nn.relu,
                snn.Linear(self._embedding_dim),
            ],
            name='joint_embedder',
        )
        self.joint_resnet = snn.Sequential([
            ResBlockV2(self._embedding_dim, name=f'joint_resblock_{i}')
            for i in range(self._hparams.representation.repr_net_res_blocks)
        ], name='joint_resnet')

    def __call__(self, inputs: CPUState):
        
        # logger.debug("representation_net program shape %s", inputs['program'].shape)
        # inputs is the observation dict
        batch_size = inputs['program'].shape[0]

        program_encoding = None
        if self._hparams.representation.use_program:
            program_encoding = self._encode_program(inputs, batch_size)
            tf.debugging.assert_all_finite(program_encoding, "RepresentationNet program_encoding contains NaN/Inf")

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
            tf.debugging.assert_all_finite(locations_encoding, "RepresentationNet locations_encoding (onehot) contains NaN/Inf")
        elif self._hparams.representation.use_locations_binary:
            locations_encoding = self._make_locations_encoding_binary(
                inputs, batch_size
            )
            tf.debugging.assert_all_finite(locations_encoding, "RepresentationNet locations_encoding (binary) contains NaN/Inf")

        # NOTE: this is not used.
        permutation_embedding = None
        if self._hparams.representation.use_permutation_embedding:
            raise NotImplementedError(
                'permutation embedding is not implemented and will not be. keeping for completeness.')
        # aggregate the locations and the program to produce a single output vector
        final_output = self.aggregate_locations_program(
            locations_encoding, permutation_embedding, program_encoding, batch_size
        )
        tf.debugging.assert_all_finite(final_output, "RepresentationNet final output contains NaN/Inf")
        return final_output

    def _encode_program(self, inputs: CPUState, batch_size):
        program = inputs['program']
        max_program_size = inputs['program'].shape[1] # TODO: this might not be a constant
        program_length = tf.cast(inputs['program_length'], tf.int32)
        program_onehot = self.make_program_onehot(
            program, batch_size, max_program_size
        )
        tf.debugging.assert_all_finite(program_onehot, "RepresentationNet program_onehot contains NaN/Inf")
        program_encoding = self.apply_program_mlp_embedder(program_onehot)
        tf.debugging.assert_all_finite(program_encoding, "RepresentationNet program_encoding after MLP contains NaN/Inf")
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
        # logger.debug("aggregate_locations_program: locations_encoding shape %s", locations_encoding.shape)
        tf.debugging.assert_all_finite(locations_encoding, "aggregate_locations_program locations_encoding contains NaN/Inf")
        
        locations_embedding = tf.vectorized_map(self.locations_embedder, locations_encoding)
        tf.debugging.assert_all_finite(locations_embedding, "aggregate_locations_program locations_embedding contains NaN/Inf")
        # logger.debug("aggregate_locations_program: locations_embedding shape %s", locations_embedding.shape)

        # broadcast the program encoding for each example.
        # this way, it matches the size of the observations.
        # logger.debug("aggregate_locations_program: program_encoding shape %s", program_encoding.shape)
        if program_encoding is not None:
            tf.debugging.assert_all_finite(program_encoding, "aggregate_locations_program program_encoding contains NaN/Inf")
            
        program_encoded_repeat = self.repeat_program_encoding(
            program_encoding[:, None, :], batch_size
        )
        tf.debugging.assert_all_finite(program_encoded_repeat, "aggregate_locations_program program_encoded_repeat contains NaN/Inf")
        # logger.debug("aggregate_locations_program: program_encoded_repeat shape %s", program_encoded_repeat.shape)

        grouped_representation = tf.concat( # concat the CPU state and the program.
            [locations_embedding, program_encoded_repeat], axis=-1
        )
        tf.debugging.assert_all_finite(grouped_representation, "aggregate_locations_program grouped_representation contains NaN/Inf")
        # logger.debug("aggregate_locations_program: grouped_representation shape %s", grouped_representation.shape)

        return self.apply_joint_embedder(grouped_representation, batch_size)

    def repeat_program_encoding(self, program_encoding, batch_size):
        program_encoding = tf.broadcast_to(
            program_encoding,
            [batch_size, self._task_spec.num_inputs, program_encoding.shape[-1]],
        )
        return program_encoding

    def apply_joint_embedder(self, grouped_representation, batch_size):
        tf.debugging.assert_all_finite(grouped_representation, "apply_joint_embedder input grouped_representation contains NaN/Inf")
        
        assert grouped_representation.shape[:2] == (batch_size, self._task_spec.num_inputs), \
            f"grouped_representation shape {grouped_representation.shape[:2]} does not match expected shape {(batch_size, self._task_spec.num_inputs)}"
        # logger.debug("apply_joint_embedder grouped_rep shape %s", grouped_representation.shape)
        # apply MLP to the combined program and locations embedding
        permutations_encoded = self.all_locations_net(grouped_representation)
        tf.debugging.assert_all_finite(permutations_encoded, "apply_joint_embedder permutations_encoded contains NaN/Inf")
        # logger.debug("apply_joint_embedder permutations_encoded shape %s", permutations_encoded.shape)
        
        # Combine all permutations into a single vector using a ResNetV2
        mean_permutations = tf.reduce_mean(permutations_encoded, axis=1, keepdims=True)
        tf.debugging.assert_all_finite(mean_permutations, "apply_joint_embedder mean_permutations contains NaN/Inf")
        
        joint_encoding = self.joint_locations_net(mean_permutations)
        tf.debugging.assert_all_finite(joint_encoding, "apply_joint_embedder joint_encoding after joint_locations_net contains NaN/Inf")
        # logger.debug("apply_joint_embedder joint_encoding shape %s", joint_encoding.shape)
        
        joint_encoding = self.joint_resnet(joint_encoding)
        tf.debugging.assert_all_finite(joint_encoding, "apply_joint_embedder joint_encoding after joint_resnet contains NaN/Inf")
        
        final_output = joint_encoding[:, 0, :] # remove the extra dimension
        tf.debugging.assert_all_finite(final_output, "apply_joint_embedder final output contains NaN/Inf")
        
        return final_output

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
        assert program_onehot.shape[:2] == (batch_size, max_program_size), \
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
        # assert program_length.shape[:2] == (batch_size, self._task_spec.num_inputs),\
        #     f"program_length shape {program_length.shape} does not match expected shape {(batch_size, self._task_spec.num_inputs)}"

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
        tf.get_logger().warning(
            "apply_program_mlp_embedder: program_encoding %s", program_encoding)
        program_encoding = self.program_mlp_embedder(program_encoding)
        tf.get_logger().warning(
            "apply_program_mlp_embedder: program_encoding after MLP %s", program_encoding)
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
        tf.debugging.assert_all_finite(program_encoding, "apply_program_attention_embedder input program_encoding contains NaN/Inf")
        program_encoding = self.positional_embedding(program_encoding)

        program_encoding = self.attention_encoders(program_encoding)

        return program_encoding

    def _make_locations_encoding_onehot(self, inputs: CPUState, batch_size):
        """Creates location encoding using onehot representation."""
        # logger.debug("make_locations_encoding_onehot shapes %s", str({k:v.shape for k,v in inputs['items']()}))
        memory = inputs['memory'] # B x E x M (batch, num_inputs, memory size)
        registers = inputs['registers'] # B x E x R (batch, num_inputs, register size)
        # logger.debug("registers shape %s, memory shape %s", registers.shape, memory.shape)
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
        locations = tf.cast(tf.concat([registers, memory], axis=-1), tf.int32) # B x E x (R + M)
        # logger.debug("locations shape %s", locations.shape)
        # to support inputs with sequences of states, we conditinally transpose
        # the locations tensor to have the shape [B, P, H, D]
        if len(locations.shape) == 4:
            # in this case, locations is [B, H, P, D]
            # and we need to transpose it to [B, P, H, D]
            locations = tf.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D]

        # One-hot encode the values in the memory and average everything across
        # permutations.
        # logger.debug("locations shape %s", locations.shape)
        locations_onehot = tf.one_hot( # shape is now B x E x num_locations x num_locations
            locations, self._task_spec.num_locations, dtype=tf.float32
        )
        # logger.debug("locations_onehot shape %s", locations_onehot.shape)
        locations_onehot = tf.reshape(locations_onehot, [batch_size, self._task_spec.num_inputs, -1])
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
        tf.debugging.assert_all_finite(x, "CategoricalHead input x contains NaN/Inf")
        
        # For training returns the logits, for inference the mean.
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
        logits = self._head(x) # project the embedding to the value support's numbeer of bins 
        tf.debugging.assert_all_finite(logits, "CategoricalHead logits after head network contains NaN/Inf")
        
        logits = tf.reshape(logits, (-1, self._value_support.num_bins)) # B x num_bins
        probs = tf.nn.softmax(logits) # take softmax -- probabilities over the bins
        tf.debugging.assert_all_finite(probs, "CategoricalHead probs after softmax contains NaN/Inf")
        
        # logger.debug("CategoricalHead: logits shape %s, probs shape %s", logits.shape, probs.shape)
        mean = self._value_support.mean(probs) # compute the mean, which is probs * [0, max_val/num_bins, 2max_val/num_bins, max_val]
        tf.debugging.assert_all_finite(mean, "CategoricalHead mean output contains NaN/Inf")
        
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
        
        self.policy_head = make_head_network(
            self.embedding_dim, self.task_spec.num_actions
        )
        self.value_head = CategoricalHead(self.embedding_dim, self.support)
        self.latency_value_head = CategoricalHead(self.embedding_dim, self.support)

    def __call__(self, embedding: tf.Tensor):
        tf.debugging.assert_all_finite(embedding, "PredictionNet input embedding contains NaN/Inf")
        
        # logger.debug("PredictionNet: latency_value_head %s", latency_value_head)
        correctness_value = self.value_head(embedding)
        tf.debugging.assert_all_finite(correctness_value['mean'], "PredictionNet correctness_value mean contains NaN/Inf")
        tf.debugging.assert_all_finite(correctness_value['logits'], "PredictionNet correctness_value logits contains NaN/Inf")
        
        # logger.debug("PredictionNet: correctness_value shape %s", str({k:v.shape for k, v in correctness_value.items()}))
        latency_value = self.latency_value_head(embedding)
        tf.debugging.assert_all_finite(latency_value['mean'], "PredictionNet latency_value mean contains NaN/Inf")
        tf.debugging.assert_all_finite(latency_value['logits'], "PredictionNet latency_value logits contains NaN/Inf")
        
        # logger.debug("PredictionNet: latency_value shape %s", str({k:v.shape for k, v in latency_value.items()}))

        # embedding is B x embedding_dim
        # with an uninitialised network, its distribution should be close to
        # a standard normal distribution with mean 0 
        
        # for debugging, we can check the distribution of the embedding
        # if logger.isEnabledFor(logging.DEBUG):
        #     embedding_mean = np.mean(embedding)
        #     embedding_std = jnp.std(embedding)
        #     embedding_min = jnp.min(embedding)
        #     embedding_max = jnp.max(embedding)
        #     logger.debug("PredictionNet.distr_check: embedding min %s, max %s mean %s std %s", embedding_min, embedding_max, embedding_mean, embedding_std)
        if len(embedding.shape) == 2:
            embedding = tf.expand_dims(embedding, axis=1)
        policy = self.policy_head(embedding) # B x num_actions
        tf.debugging.assert_all_finite(policy, "PredictionNet policy output contains NaN/Inf")
        
        policy = tf.reshape(policy, (-1, self.task_spec.num_actions)) # B x num_actions
        # similarly, the policy should be close to a uniform distribution
        # with a mean of 1/num_actions
        # if logger.isEnabledFor(logging.DEBUG):
        #     policy_mean = jnp.mean(policy)
        #     policy_std = jnp.std(policy)
        #     policy_min = jnp.min(policy)
        #     policy_max = jnp.max(policy)
        #     logger.debug("PredictionNet.distr_check: policy min %s, max %s mean %s std %s", policy_min, policy_max, policy_mean, policy_std)
        
        final_value = correctness_value['mean'] + latency_value['mean']
        tf.debugging.assert_all_finite(final_value, "PredictionNet final value contains NaN/Inf")
        
        output = NetworkOutput(
            value=final_value,
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
    
    def __init__(self, hparams, task_spec, input_signature,
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
        self.forward = tf.function(
            self.inference, input_signature=[input_signature], jit_compile=True
        )
    
    def inference(self, inputs: CPUState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes and returns the policy and value logits for the AZLearner."""
        logger.warning("AlphaDevNetwork [retracing]: inputs %s (only a concern if happens repeatedly)", str({k:v.shape for k,v in inputs.items()}))
        # inputs is the observation dict
        embedding: tf.Tensor = self._representation_net(inputs)
        tf.debugging.assert_all_finite(embedding, "AlphaDevNetwork embedding from representation net contains NaN/Inf")
        # logger.debug("AlphaDevNetwork: embedding shape %s", embedding.shape)
        
        prediction: NetworkOutput = self._prediction_net(embedding)
        tf.debugging.assert_all_finite(prediction.value, "AlphaDevNetwork prediction value contains NaN/Inf")
        tf.debugging.assert_all_finite(prediction.policy_logits, "AlphaDevNetwork prediction policy_logits contains NaN/Inf")
        # logger.debug("AlphaDevNetwork: prediction obtained")
        return self._return_fn(prediction)
    
    def __call__(self, inputs: CPUState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Alias for inference."""
        return self.inference(inputs)

def make_input_spec(observation_spec) -> DiscreteArray:
    # observation_spac is possibly a nested dict of dm_env.specs.Array type.
    # we want to convert them to TensorSpec
    def process_leaf(leaf: Array) -> tf.TensorSpec:
        return tf.TensorSpec(
            shape=leaf.shape, dtype=leaf.dtype, name=leaf.name
        )
    return tree.map_structure(
        process_leaf, observation_spec
    )

class NetworkFactory:
    def __init__(self, config: AlphaDevConfig): self._hparams = config.hparams; self._task_spec = config.task_spec
    def __call__(self, spec: DiscreteArray, name:str='AlphaDevNetwork'): return AlphaDevNetwork(hparams=self._hparams, task_spec=self._task_spec, input_signature=spec, name=name)
