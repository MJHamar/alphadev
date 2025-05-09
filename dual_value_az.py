"""Definition of the AlphaZero learner with a dual value head."""
import tensorflow as tf
from acme.agents.tf.mcts.learning import AZLearner

from .distribution import DistributionSupport

# From AlphaDev pseudocode
def scalar_loss(prediction:tf.Tensor, target:tf.Tensor, value_max, num_bins) -> float:
    # Get the support from the network's configuration instead of trying to access it directly
    # from the transformed module
    support = DistributionSupport(value_max, num_bins)
    # sm_cross_entropy normalizes the prediction and compares it to the target, which
    # is a two-hot vector, summing up to 1.
    # The target is a scalar, so we need to convert it to a two-hot vector.
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=support.scalar_to_two_hot(target)
    )


class DualValueAZLearner(AZLearner):
    # we only override the _step method
    def __init__(self,
                 network,
                 optimizer,
                 dataset,
                 discount,
                 logger = None, counter = None):
        super().__init__(network, optimizer, dataset, discount, logger, counter)
    
    @tf.function
    def _step(self) -> tf.Tensor:
        """Do a step of SGD on the loss.
        
        Unlike AlphaZero, the AlphaDev pseudocode suggests that the loss is computed
        as the cross-entropy between the predicted and target values,
        where the value functions are interpreted as a categorical distribution
        of N outcomes, where each outcome n<N corresponds to the interpolation between
        0 and the maximum reward.
        Here, we implement this version.
        """

        inputs = next(self._iterator)
        # added c_t: correctness, l_t: latency reward terms. r_t = c_t*scale + l_t*scale
        o_t, _, r_t, d_t, o_tp1, c_t, l_t ,extras = inputs.data
        pi_t = extras['pi']

        with tf.GradientTape() as tape:
            # Forward the network on the two states in the transition.
            logits, value, correctness_logits, latency_logits = self._network(o_t)
            _, target_value, _, _ = self._network(o_tp1)
            target_value = tf.stop_gradient(target_value)

            correctness_loss = scalar_loss(correctness_logits, c_t,
                                           self._network.value_max, self._network.num_bins)
            latency_loss = scalar_loss(latency_logits, l_t,
                                       self._network.value_max, self._network.num_bins)

            # Value loss is simply on-policy TD learning.
            reward_target = r_t + self._discount * d_t * target_value
            value_loss = tf.square(reward_target - value)
            
            # Policy loss distills MCTS policy into the policy network.
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=pi_t)

            # Compute gradients.
            loss = tf.reduce_mean(correctness_loss + latency_loss + value_loss + policy_loss)
            
            gradients = tape.gradient(loss, self._network.trainable_variables)

        self._optimizer.apply(gradients, self._network.trainable_variables)

        return loss
