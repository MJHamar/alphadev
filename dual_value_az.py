"""Definition of the AlphaZero learner with a dual value head."""
import tensorflow as tf
from acme.agents.tf.mcts.learning import AZLearner
from acme.agents.tf.mcts.acting import MCTSActor
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts import search
from acme.agents.tf.mcts import models
import dm_env
import tree
import numpy as np
from scipy import special
import dataclasses

from distribution import DistributionSupport

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
    # override the _forward method to account for the correctness and latency logits from the network.
    def _forward(self, observation):
        """Performs a forward pass of the policy-value network."""
        logits, value, _, _ = self._network(tree.map_structure(lambda o: tf.expand_dims(o, axis=0), observation))

        # Convert to numpy & take softmax.
        logits = logits.numpy().squeeze(axis=0)
        value = value.numpy().item()
        probs = special.softmax(logits)

        return probs, value

    def select_action(self, observation: types.Observation) -> types.Action:
        """Computes the agent's policy via MCTS."""
        if self._model.needs_reset:
            self._model.reset(observation)

        # Compute a fresh MCTS plan.
        # NOTE: we use a modified implementation here.
        root = dv_mcts(
            observation,
            model=self._model,
            search_policy=search.puct,
            evaluation=self._forward,
            num_simulations=self._num_simulations,
            num_actions=self._num_actions,
            discount=self._discount,
        )

        # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
        probs = search.visit_count_policy(root)
        action = np.int32(np.random.choice(self._actions, p=probs))

        # Save the policy probs so that we can add them to replay in `observe()`.
        self._probs = probs.astype(np.float32)

        return action

@dataclasses.dataclass
class DvNode(search.Node):
    reward: np.ndarray = np.zeros((3,), dtype=np.float32)

    def expand(self, prior: np.ndarray):
        """Expands this node, adding child nodes."""
        assert prior.ndim == 1  # Prior should be a flat vector.
        assert self.terminal is False, "Can't expand terminal nodes."
        for a, p in enumerate(prior):
            self.children[a] = DvNode(prior=p)

# Dual-valued implementation of the MCTS algorithm from 
# `acme/agents/tf/mcts/search.py`
def dv_mcts(
    observation: types.Observation,
    model: models.Model,
    search_policy: search.SearchPolicy,
    evaluation: types.EvaluationFn,
    num_simulations: int,
    num_actions: int,
    discount: float = 1.,
    dirichlet_alpha: float = 1,
    exploration_fraction: float = 0.,
) -> DvNode:
    """Does Monte Carlo tree search (MCTS), AlphaZero style."""

    # Evaluate the prior policy for this state.
    prior, value = evaluation(observation)
    assert prior.shape == (num_actions,)

    # Add exploration noise to the prior.
    noise = np.random.dirichlet(alpha=[dirichlet_alpha] * num_actions)
    prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

    # Create a fresh tree search.
    root = DvNode()
    root.expand(prior)

    # Save the model state so that we can reset it for each simulation.
    model.save_checkpoint()
    for _ in range(num_simulations):
        # Start a new simulation from the top.
        trajectory = [root]
        node = root

        # Generate a trajectory.
        timestep = None
        while node.children:
            # Select an action according to the search policy.
            action = search_policy(node)

            # Point the node at the corresponding child.
            node = node.children[action]

            # Step the simulator and add this timestep to the node.
            timestep = model.step(action)
            # NOTE: changed line nr. 1
            node.reward = timestep.reward if timestep.reward is not None else np.zeros((3,), dtype=np.float32)
            node.terminal = timestep.last()
            trajectory.append(node)

        if timestep is None:
            raise ValueError('Generated an empty rollout; this should not happen.')

        # Calculate the bootstrap for leaf nodes.
        if node.terminal:
            # If terminal, there is no bootstrap value.
            value = 0.
        else:
            # Otherwise, bootstrap from this node with our value function.
            prior, value = evaluation(timestep.observation)

            # We also want to expand this node for next time.
            node.expand(prior)

        # Load the saved model state.
        model.load_checkpoint()

        # Monte Carlo back-up with bootstrap from value function.
        ret = value
        while trajectory:
            # Pop off the latest node in the trajectory.
            node = trajectory.pop()

            # Accumulate the discounted return
            ret *= discount
            # NOTE: changed line nr. 2
            ret += node.reward[0] # only the combined (correctness + latency) reward is used

            # Update the node.
            node.total_value += ret
            node.visit_count += 1

    # count the number of expanded nodes using BFS
    # queue = [root]
    # expanded_nodes = 0
    # max_num_expanded_children = 0
    # while queue:
    #     node = queue.pop(0)
    #     if node.children:
    #         expanded_nodes += 1
    #         num_children = 0
    #         for c in node.children.values():
    #             if c.children:
    #                 num_children += 1
    #                 queue.append(c)
    #         max_num_expanded_children = max(max_num_expanded_children, num_children)
    # logger.debug("MCTS Expanded %d nodes. max expanded children: %d", expanded_nodes, max_num_expanded_children)

    return root
