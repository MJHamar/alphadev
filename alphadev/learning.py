"""Customized implementation of the AZLearner class from acme."""
from typing import Optional, List
import sonnet as snn
from .tf_util import tf
import numpy as np
import tree

import acme
from acme.utils import loggers
from acme.utils import counting
from acme.tf import utils as tf2_utils

from .service.variable_service import VariableService

import logging
base_logger = logging.getLogger(__name__)
base_logger.setLevel(logging.DEBUG)


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
        target_network: Optional[snn.Module] = None,
        target_update_period: Optional[int] = None,
        training_steps: Optional[int] = None,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):

        # Logger and counter for tracking statistics / writing out to terminal.
        self._counter = counting.Counter(counter, 'learner', return_only_prefixed=True)
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=30.)

        self._variable_service = variable_service
        self._variable_update_period = varibale_update_period
        self._target_update_period = target_update_period

        # Internalize components.
        # TODO(b/155086959): Fix type stubs and remove.
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
        self._optimizer = optimizer
        self._network = network
        if target_network is None:
            # If no target network is provided, use the same network.
            base_logger.debug("AZLearner: using the same network as target network")
            self._should_update_target = False
            target_network = network
        else:
            base_logger.debug(f"AZLearner: using a different network as target network: {target_network}")
            self._should_update_target = True
            self._target_network = target_network
        
        self._variables = network.trainable_variables
        self._discount = np.float32(discount)
        
        # the superclass will take care of this parameter.
        self._training_steps = training_steps
        
        # publish current variables
        if self._variable_service is not None:
            base_logger.debug(f"AZLearner: initializing variable service")
            self._variable_service.update(self.get_variables([]))
    
    def _maybe_update_target_network(self):
        """Updates the target network if needed."""
        if self._should_update_target:
            counts = self._counter.get_counts()
            base_logger.debug("AZLearner: maybe updating target network with counts %s", counts)
            steps = counts.get('step', 0)
            if steps % self._target_update_period == 0:
                base_logger.debug("AZLearner: updating target network at step %s", steps)
                tree.map_structure(
                    lambda t, s: t.assign(s), 
                    self._target_network.trainable_variables, 
                    self._network.trainable_variables
                )
    
    # @tf.function
    def _step(self) -> tf.Tensor:
        """Do a step of SGD on the loss."""

        inputs = next(self._iterator)
        o_t, _, r_t, d_t, o_tp1, extras = inputs.data
        pi_t = extras['pi']
        # NOTE: this implementation doesn't consider latency predictions.
        # see `dual_value_az.py`
        latency_t = extras['latency_reward']
        
        with tf.GradientTape() as tape:
            # Forward the network on the two states in the transition.
            logits, value = self._network(o_t)
            _, target_value = self._target_network(o_tp1)
            target_value = tf.stop_gradient(target_value)
            
            # Value loss is simply on-policy TD learning.
            value_loss = tf.square(r_t + self._discount * d_t * target_value - value)

            # Policy loss distills MCTS policy into the policy network.
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=pi_t)

            # Compute gradients.
            loss = tf.reduce_mean(value_loss + policy_loss)
            gradients = tape.gradient(loss, self._network.trainable_variables)
            
            # clip the gradients
            gradients, _ = tf.clip_by_global_norm(gradients, self._network._hparams.grad_clip_norm)
            
            # base_logger.debug('losses: logits %s, value %s, target_value %s, v_loss %s, p_loss %s, loss %s, grads %s',
            #                  logits, value, target_value, value_loss, policy_loss, loss,
            #                  [g for g in gradients])
        
        self._optimizer.apply(gradients, self._network.trainable_variables)
        
        return loss

    def step(self):
        """Does a step of SGD and logs the results."""
        self._maybe_update_target_network()
        
        loss = self._step()
        self._logger.write({'loss': loss})
        counts = self._counter.increment(**{'step': 1})
        base_logger.debug('counts %s', counts)
        if self._variable_service is not None:
            base_logger.debug(f"updating variables at step {counts['step']}")
            self._variable_service.update(self.get_variables([]))

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        """Exposes the variables for actors to update from."""
        return tf2_utils.to_numpy(self._variables)
