"""
Extension of `acme.agents.tf.mcts.search`
"""
from typing import Dict, Optional, List, Generator

from acme.agents.tf.mcts.search import SearchPolicy
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts import models

import dataclasses
import numpy as np

from tqdm import tqdm

@dataclasses.dataclass
class Node:
    """A MCTS node."""

    prior: float = 1.
    terminal: bool = False
    _reward: float = 0.
    _visit_count: int = 0
    _total_value: float = 0.
    _value: types.Value = 0.  # Q(s, a)
    children: Dict[types.Action, 'Node'] = dataclasses.field(default_factory=dict)
    _children_priors: List[float] = None
    action_mask: Optional[np.ndarray] = None
    container_cls: Optional[type] = None
    
    def expand(self, prior: np.ndarray, action_mask: Optional[np.ndarray] = None):
        """Expands this node, adding child nodes."""
        assert prior.ndim == 1  # Prior should be a flat vector.
        self.action_mask = action_mask if action_mask is not None else np.ones_like(prior, dtype=bool)
        self._children_priors = prior
        self.children = self._make_children(prior.shape[0])
    
    def _make_children(self, num_kids:int) -> Dict[types.Action, 'Node']:
        return NodeContainer(num_kids)

    @property
    def children_visits(self) -> np.ndarray:
        """Return array of visit counts of visited children."""
        return np.array([c.visit_count for c in self.children.values()])

    @property
    def children_values(self) -> np.ndarray:
        """Return array of values of visited children."""
        return np.array([c.value for c in self.children.values()])
    
    @property
    def children_priors(self) -> np.ndarray:
        return self._children_priors

    def visit(self, value: float):
        """Visit this node andd update its value by computing the new average
        based on the previous value and the new value. also increment the visit count."""
        self._visit_count += 1
        self._total_value += value
        self._value = self._total_value / self._visit_count

    @property
    def visit_count(self) -> int:
        """Return the visit count of this node."""
        return self._visit_count

    @property
    def value(self) -> types.Value:  # Q(s, a)
        return self._value
    
    @property
    def reward(self) -> float:
        return self._reward
    @reward.setter
    def reward(self, value: float):
        self._reward = value

class NodeContainer():
    default_node = Node()

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.nodes = {a: self.__class__.default_node for a in range(num_nodes)}
    
    def __getitem__(self, key: int) -> Node:
        if id(self.nodes[key]) == id(self.__class__.default_node):
            # If the node is the default node, create a new one.
            self.nodes[key] = Node()
        return self.nodes[key]
    
    def values(self) -> Generator[Node, None, None]:
        return self.nodes.values()


class DvNode(Node):
    """Just like a Node, but rewards are 3D vectors."""
    def _make_children(self, num_kids):
        return DvNodeContainer(num_kids)
    
    @property
    def reward(self) -> np.ndarray:
        """Return the reward of this node."""
        return self._reward[0]
    
    @reward.setter
    def reward(self, value) -> np.ndarray:
        self._reward = value

# NOTE: unfortunately, python doesn't support generics in any meaningful way.
class DvNodeContainer(NodeContainer):
    default_node = DvNode()


def mcts(
    observation: types.Observation,
    model: models.Model,
    search_policy: SearchPolicy,
    evaluation: types.EvaluationFn,
    num_simulations: int,
    num_actions: int,
    discount: float = 1.,
    dirichlet_alpha: float = 1,
    exploration_fraction: float = 0.,
    node_class: Optional[Node] = Node,
) -> Node:
    """Does Monte Carlo tree search (MCTS), AlphaZero style."""

    # Evaluate the prior policy for this state.
    prior, value = evaluation(observation)
    assert prior.shape == (num_actions,), f"Expected prior shape {(num_actions,)}, got {prior.shape}."

    # Add exploration noise to the prior.
    noise = np.random.dirichlet(alpha=[dirichlet_alpha] * num_actions)
    prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

    # Create a fresh tree search.
    root = node_class()
    legal_actions = model.legal_actions()
    root.expand(prior, legal_actions)

    # Save the model state so that we can reset it for each simulation.
    model.save_checkpoint()
    for _ in range(num_simulations):
        # Start a new simulation from the top.
        trajectory = [root]
        node = root

        # Generate a trajectory.
        timestep = None
        actions = []
        while node.children:
            # Select an action according to the search policy.
            action = search_policy(node)

            # Point the node at the corresponding child.
            node = node.children[action]
            trajectory.append(node)
            actions.append(action)

        # Replay the simulator until the current node and expand it.
        timestep = model.step(actions)
        node.reward = timestep.reward if timestep.reward is not None else 0.
        node.terminal = timestep.last()

        # Calculate the bootstrap for leaf nodes.
        if node.terminal:
            # If terminal, there is no bootstrap value.
            value = 0.
        else:
            # Otherwise, bootstrap from this node with our value function.
            prior, value = evaluation(timestep.observation)
            legal_actions = model.legal_actions()

            # We also want to expand this node for next time.
            node.expand(prior, legal_actions)

        # Load the saved model state.
        model.load_checkpoint()

        # Monte Carlo back-up with bootstrap from value function.
        ret = value
        while trajectory:
            # Pop off the latest node in the trajectory.
            node = trajectory.pop()

            # Accumulate the discounted return
            ret *= discount
            ret += node.reward

            # Update the node.
            node.visit(ret)

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

def puct(node: Node, ucb_scaling: float = 1.) -> types.Action:
    """PUCT search policy, i.e. UCT with 'prior' policy."""
    # Action values Q(s,a).
    value_scores = node.children_values
    # check_numerics(value_scores)

    # Policy prior P(s,a).
    priors = node.children_priors
    # check_numerics(priors)

    # Visit ratios.
    nominator = np.sqrt(node.visit_count)
    visit_ratios = nominator / (node.children_visits + 1)
    # check_numerics(visit_ratios)

    # Combine.
    puct_scores = value_scores + ucb_scaling * priors * visit_ratios
    return argmax(puct_scores, node.action_mask)

def argmax(values: np.ndarray, mask: np.ndarray) -> types.Action:
    """Argmax with random tie-breaking."""
    # check_numerics(values)
    max_value = np.max(values*mask) # mask the values to only consider valid actions
    return np.int32(np.random.choice(np.flatnonzero(values == max_value)))

def check_numerics(values: np.ndarray):
    """Raises a ValueError if any of the inputs are NaN or Inf."""
    if not np.isfinite(values).all():
        raise ValueError('check_numerics failed. Inputs: {}. '.format(values))

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
