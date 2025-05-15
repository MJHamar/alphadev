"""
Extension of `acme.agents.tf.mcts.search`
"""
from acme.agents.tf.mcts.search import SearchPolicy, puct, Node, check_numerics
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts import models

import dataclasses
import numpy as np


@dataclasses.dataclass
class DvNode(Node):
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
    search_policy: SearchPolicy,
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
    assert prior.shape == (num_actions,), f"Expected prior shape {(num_actions,)}, got {prior.shape}."

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
    # print("MCTS Expanded %d nodes. max expanded children: %d" % (expanded_nodes, max_num_expanded_children))

    return root

def dyn_puct(
    node: Node,
    c_puct_base: float = 19652,
    c_puct_init: float = 1.25,
) -> int:
    """
    Selects an action according to the PUCT algorithm proposed by 
    Rosin 2011 and adapted in Silver et. al 2016.
    """
    # Calculate the PUCT scaling factor based on the visit counts of the parent.
    c_puct = (
        np.log((node.visit_count + c_puct_base + 1) / c_puct_base)
        + c_puct_init
    )
    # Make a call to the PUCT function with constant scaling.
    return puct(node, c_puct)

class PUCTSearchPolicy(SearchPolicy):
    """
    A search policy that uses the PUCT algorithm to select actions.
    """
    def __init__(self, c_puct_base: float = 19652, c_puct_init: float = 1.25):
        self.c_puct_base = c_puct_base
        self.c_puct_init = c_puct_init

    def __call__(self, node: Node) -> int:
        return dyn_puct(node, self.c_puct_base, self.c_puct_init)

def visit_count_policy(root: Node, temperature: float = 1.0, mask: np.ndarray = None) -> int:
    visits = root.children_visits
    masked_visits = visits * mask # multiply by the mask to keep the shape, but make invalid actions impossible to choose    
    rescaled_visits = masked_visits**(1 / temperature)
    probs = rescaled_visits / np.sum(rescaled_visits)
    check_numerics(probs)
    
    return probs