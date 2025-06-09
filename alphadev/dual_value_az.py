"""Definition of the AlphaZero learner with a dual value head."""
import tensorflow as tf
from acme.agents.tf.mcts import types
import tree
import numpy as np
from scipy import special

from .distribution import DistributionSupport
from .acting import MCTSActor
from .search import mcts, DvNode, visit_count_policy
from .learning import AZLearner
from .service.service import RPCClient
from .service.variable_service import VariableService

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        o_t, _, rcl_t, d_t, o_tp1, extras = inputs.data
        r_t, c_t, l_t = rcl_t[:,0], rcl_t[:,1], rcl_t[:,2]
        
        pi_t = extras['pi']

        with tf.GradientTape() as tape:
            # Forward the network on the two states in the transition.
            logits, value, correctness_logits, latency_logits = self._network(o_t)
            _, target_value, _, _ = self._network(o_tp1)
            target_value = tf.stop_gradient(target_value)

            correctness_loss = scalar_loss(correctness_logits, c_t,
                                           self._network._hparams.value_max, self._network._hparams.value_num_bins)
            latency_loss = scalar_loss(latency_logits, l_t,
                                       self._network._hparams.value_max, self._network._hparams.value_num_bins)

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

class DualValueMCTSActor(MCTSActor):
    _st_node = DvNode
    # override the _forward method to account for the correctness and latency logits from the network.
    def _forward(self, observation):
        """Performs a forward pass of the policy-value network."""
        if self._add_batch_dim:
            logits, value, _, _ = self._network(tree.map_structure(lambda o: tf.expand_dims(o, axis=0), observation))
        else:
            logits, value, _, _ = self._network(observation)

        # Convert to numpy & take softmax.
        if self._add_batch_dim:
            logits = logits.numpy().squeeze(axis=0)
        else:
            logits = logits.numpy()
        value = value.numpy().item()
        probs = special.softmax(logits)
        return probs, value

    def select_action(self, observation: types.Observation) -> types.Action:
        """Computes the agent's policy via MCTS."""
        if self._model.needs_reset:
            self._model.reset(observation)

        # Compute a fresh MCTS plan.
        # NOTE: we use a modified implementation here.
        root = self.mcts.search(observation)

        # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
        training_steps = self._counter.get_counts().get(self._counter.get_steps_key(), 0)
        temperature = self._temperature_fn(training_steps)
        # get the action mask from the model
        if self._model.needs_reset:
            self._model.reset(observation)
        action_mask = self._model.legal_actions()
        # perform masked visit count policy
        probs = visit_count_policy(root, temperature=temperature, mask=action_mask)
        assert probs.shape == (self._num_actions,), f"Expected probs shape {(self._num_actions,)}, got {probs.shape}."
        # sample an action from the masked visit count policy
        action = np.int32(np.random.choice(self._actions, p=probs))

        # Save the policy probs so that we can add them to replay in `observe()`.
        self._probs = probs.astype(np.float32)

        for observer in self._observers:
            observer.on_action_selection(
                node=root, probs=probs, action=action,
                training_steps=training_steps, temperature=temperature)

        return action
