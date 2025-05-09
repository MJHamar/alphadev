import tensorflow as tf

class DistributionSupport(object):

    def __init__(self, value_max: float, num_bins: int):
        self.value_max = value_max
        self.num_bins = num_bins
        self.value_support = tf.cast(
            tf.linspace(0, value_max, num_bins), tf.float32)

    def mean(self, logits: tf.Tensor) -> float:
        # logits has shape B x num_bins
        # logger.debug("DistSup.mean compute twohot logits for %s", logits.shape)
        assert logits.shape[1] == self.num_bins, \
            f"Logits shape {logits.shape} does not match num_bins {self.num_bins}"
        # compute the mean of the logits
        mean = tf.tensordot(logits, self.value_support, axes=1) # (B, num_bins) * (num_bins,) = (B,)
        assert mean.shape == (logits.shape[0],), \
            f"Mean shape {mean.shape} does not match logits shape {logits.shape}"
        return mean

    def scalar_to_two_hot(self, scalar: tf.Tensor) -> tf.Tensor:
        """
        Converts a scalar to a two-hot encoding.
        Finds the two closest bins to the scalar (lower and upper) and
        sets these indices to 1. All other indices are set to 0.
        """
        # Bins are -probably- a linear interpolation between 0 and value_max
        # and we need to assign non-zero values to the two closest bins
        # based on proximity to the scalar.
        # input is B x 1
        # output needs to be B x num_bins
        
        # first, calculate the steep size
        step = self.value_max / self.num_bins # scalar step size
        # find the two closest bins
        scalar = tf.constant(scalar)
        low_bin = tf.cast(tf.floor(scalar / step), tf.int32) # B x 1
        high_bin = low_bin + 1 # B x 1
        # find prroximity to the bins
        low_bin_proximity = tf.abs(scalar - low_bin * step)
        high_bin_proximity = tf.abs(scalar - high_bin * step)
        # weights are the inverse of the proximity
        low_bin_weight = high_bin * step - low_bin_proximity
        high_bin_weight = low_bin * step - high_bin_proximity
        # create the two-hot encoding
        # which is a B x num_bins tensor of 0s except for locations
        # low_bin and high_bin, which are both vectors of (num_bins,) dimensions
        # and index the second dimension of the output. 
        # the value of the output at these indices is defined by the low/high_bin_weight
        # vectors, which are also (num_bins,) in size.
        
        two_hot = tf.zeros((scalar.shape[0], self.num_bins), dtype=tf.float32)
        two_hot = tf.tensor_scatter_nd_update(
            two_hot,
            tf.stack([tf.range(scalar.shape[0]), low_bin], axis=1),
            tf.cast(low_bin_weight, tf.float32),
        )
        two_hot = tf.tensor_scatter_nd_update(
            two_hot,
            tf.stack([tf.range(scalar.shape[0]), high_bin], axis=1),
            tf.cast(high_bin_weight, tf.float32),
        )
        
        # set the two closest bins to 1
        return two_hot
