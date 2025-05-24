"""Customized implementation of the AZLearner class from acme."""
from typing import Optional, List
import sonnet as snn
import tensorflow as tf
import numpy as np

import acme
from acme.utils import loggers
from acme.utils import counting
from acme.tf import utils as tf2_utils

from .variable_service import VariableService

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AZLearner(acme.Learner):
    """AlphaZero-style learning."""
    def __init__(
        self,
        network: snn.Module,
        optimizer: snn.Optimizer,
        dataset: tf.data.Dataset,
        discount: float,
        variable_service: VariableService,
        varibale_update_period: int = 10,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):

        # Logger and counter for tracking statistics / writing out to terminal.
        self._counter = counting.Counter(counter, 'learner')
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=30.)

        self._variable_service = variable_service
        self._variable_update_period = varibale_update_period

        # Internalize components.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
        self._optimizer = optimizer
        self._network = network
        self._variables = network.trainable_variables
        self._discount = np.float32(discount)

    @tf.function
    def _step(self) -> tf.Tensor:
        """Do a step of SGD on the loss."""

        inputs = next(self._iterator)
        o_t, _, r_t, d_t, o_tp1, extras = inputs.data
        pi_t = extras['pi']

        with tf.GradientTape() as tape:
            # Forward the network on the two states in the transition.
            logits, value = self._network(o_t)
            _, target_value = self._network(o_tp1)
            target_value = tf.stop_gradient(target_value)

            # Value loss is simply on-policy TD learning.
            value_loss = tf.square(r_t + self._discount * d_t * target_value - value)

            # Policy loss distills MCTS policy into the policy network.
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=pi_t)

            # Compute gradients.
            loss = tf.reduce_mean(value_loss + policy_loss)
            gradients = tape.gradient(loss, self._network.trainable_variables)

        self._optimizer.apply(gradients, self._network.trainable_variables)

        return loss

    def step(self):
        """Does a step of SGD and logs the results."""
        loss = self._step()
        self._logger.write({'loss': loss})
        counts = self._counter.increment(**{'step': 1})
        if counts['step'] % self._variable_update_period == 0:
            logger.debug(f"updating variables at step {counts['step']}")
            self._variable_service.update(self.get_variables())

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        """Exposes the variables for actors to update from."""
        return tf2_utils.to_numpy(self._variables)
