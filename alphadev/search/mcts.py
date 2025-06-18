"""
Extension of `acme.agents.tf.mcts.search`
"""
from typing import Dict, Optional, List, Tuple, Sequence
from collections import defaultdict

from acme.agents.tf.mcts.search import SearchPolicy
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts import models

import dataclasses
import numpy as np

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NodeBase:
    """Locally stored MCTS node. uses lazy evaluation."""
    __annotations__ = {
        'parent': Optional['NodeBase'],
        'action': Optional[types.Action],
        'expanded': bool,
        'terminal': bool,
        'visit_count': int,
        'prior': np.ndarray,
        'W': np.ndarray,  # total value of the children.
        'Nw': np.ndarray,  # visit count of the children.
        'R': np.ndarray,  # reward of the children.
        'Nr': np.ndarray,  # visit count of the children.
        'mask': np.ndarray,  # boolean mask of legal actions.
        'children': Sequence['NodeBase'], # a list of pointers by default.
    }
    @classmethod
    def define(cls, width, lambda_: float=0.5) -> 'NodeBase':
        assert lambda_ >= 0 and lambda_ <= 1, "lambda must be in [0, 1]"
        class Node(cls):
            _width = width
            _lam = lambda_
        return Node
    
    def __init__(self,
        parent: Optional['NodeBase'] = None,
        action: Optional[types.Action] = None,
    ):
        self._parent = parent
        self._action = action
        self._expanded = np.array(False, dtype=np.bool_)
        self._terminal = np.array(False, dtype=np.bool_)
        self._children = defaultdict(lambda: None)  # lazy evaluation of children.
        self._prior = None
        self._W = None
        self._Nw = None
        self._R = None
        self._Nr = None
        self._mask = None
    
    def expand(self, prior: np.ndarray):
        self._expanded[...] = True
        self._prior = prior
    
    def backup_value(self, action:types.Action, value: float, discount: float = 1.0, trajectory: Optional[List['NodeBase']] = []):
        """Update the visit count and total value of this node."""
        # logger.debug(f"Backing up value {value} for action {action} in node {self} (type {type(self)}).")
        if len(trajectory) == 0:
            parent = self.parent
        else:
            parent = trajectory.pop()
        # Update the visit count and total value.
        self.W[action] += value
        self.Nw[action] += 1
        if parent is not None:
            # Recursively backup the value to the parent node.
            parent.backup_value(self.action, value * discount, discount, trajectory)
    
    def backup_reward(self, action:types.Action, reward: bool, discount: float, trajectory: Optional[List['NodeBase']] = []):
        """
        Recursively set the reward for this node and its parents.
        trajectory can be provided to avoid having to re-create the path.
        """
        logger.debug(f"Backing up reward {reward} for action {action} in node {self}.")
        if len(trajectory) == 0:
            parent = self.parent
        else:
            parent = trajectory.pop()
        logger.debug(f"action: {action}, type: {type(action)}")
        self.R[action] += reward
        self.Nr[action] += 1
        logger.debug(f"Backup; trajectory: {trajectory}, parent: {parent}, node: {self}")
        # input("stop here to debug")
        if parent is not None:
            parent.backup_reward(self.action, reward*discount, discount, trajectory)
        
    
    @property
    def zeros(self):
        return np.zeros(self._width, dtype=np.float32)
    
    @property
    def children_values(self) -> np.ndarray:
        """
        Return array of values of visited children.
        Q(s, a) = (1-lam) * R(s, a) / Nr(s, a) + lam * W(s, a) / Nw(s, a)
        via Silver et al. 2016.
        where lam is a hyperparameter that controls the balance between
        the reward and the value.
        """
        values_r = np.divide(self.R, self.Nr, out=self.zeros, where=self.Nr != 0)
        values_w = np.divide(self.W, self.Nw, out=self.zeros, where=self.Nw != 0)
        return (1-self._lam)*values_r + self._lam*values_w
    @property
    def children_visits(self) -> np.ndarray:
        """Return array of visit counts of visited children."""
        return self.Nr
    
    @property
    def parent(self):      return self._parent
    @property
    def action(self):      return self._action
    @property
    def expanded(self):    return self._expanded
    @property
    def terminal(self):    return self._terminal
    @property
    def visit_count(self):
        if self.parent is None: return np.sum(self.Nr)
        return self.parent.get_visit_count(self._action)
    @property
    def reward(self) -> float:
        # parent's action is None.
        return self.parent.get_reward(self._action)
    @property
    def prior(self):
        if self._prior is None: self._prior = np.zeros(self._width, dtype=np.float32)
        return self._prior
    @property
    def W(self):
        if self._W is None: self._W = np.zeros(self._width, dtype=np.float32)
        return self._W
    @property
    def Nw(self):
        if self._Nw is None: self._Nw = np.zeros(self._width, dtype=np.int32)
        return self._Nw
    @property
    def R(self):
        if self._R is None: self._R = np.zeros(self._width, dtype=np.float32)
        return self._R
    @property
    def Nr(self):
        if self._Nr is None: self._Nr = np.zeros(self._width, dtype=np.int32)
        return self._Nr
    @property
    def mask(self):
        if self._mask is None: self._mask = np.zeros(self._width, dtype=np.bool_)
        return self._mask
    
    @property
    def legal_actions(self) -> np.ndarray:
        """Return a boolean mask of legal actions."""
        return self.mask
    
    def is_root(self) -> bool:
        """Check if this node is the root node."""
        return self._parent is None
    
    def get_child(self, action: types.Action) -> Optional['NodeBase']:
        return self._children[action]
    def set_child(self, action: types.Action, child: 'NodeBase'):
        """Set a child node for the given action."""
        self._children[action] = child
    
    def set_root(self):
        """Set this node as the root node."""
        self._parent = None
        self._action = None
    
    def set_parent(self, parent: 'NodeBase', action: Optional[types.Action] = None):
        """Set the parent node and action for this node."""
        self._parent = parent
        self._action = action
    
    def set_legal_actions(self, legal_actions: np.ndarray):
        self.mask[...] = legal_actions
    
    def set_terminal(self, terminal: bool):
        self.terminal[...] = terminal
    
    def get_visit_count(self, action: Optional[types.Action]=None) -> int:
        logger.debug(f"Getting visit count for action: {action} in node: {self}.")
        return self.Nr[action]
    def get_reward(self, action: Optional[types.Action]=None) -> float:
        if self.is_root(): return 0.0
        return self.R[action] / self.Nr[action] if self.Nr[action] > 0 else 0.0

    def __repr__(self):
        return f"{'Root' if self.is_root() else 'Node'}(action={self.action}, ex={self.expanded}, term={self.terminal})"
    
    def __eq__(self):
        """Check if two nodes are equal based on their action and parent."""
        # FIXME
        return False 

class MCTSBase:
    """Base class for MCTS algorithms."""
    def __init__(self,
        num_simulations: int,
        num_actions: int,
        model: models.Model,
        search_policy: SearchPolicy,
        evaluation: types.EvaluationFn,
        discount: float = 1.,
        dirichlet_alpha: float = 1,
        exploration_fraction: float = 0.,
    ):
        self.model = model
        self.search_policy = search_policy
        self.evaluation = evaluation
        self.num_simulations = num_simulations
        self.num_actions = num_actions
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        # Define the node class with the number of actions.
        self.node_cls = NodeBase.define(num_actions)
        self._root = self._make_node()
    
    def search(self, observation: types.Observation, last_action: Optional[types.Action] = None) -> NodeBase:
        """Single-threaded Monte-Carlo Tree Search."""
        # 0. make a copy of the model's state.
        self.model.save_checkpoint()
        # 1. initialize the tree with the current observation.
        root = self.init_tree(observation, last_action)
        # 2. run the search policy for a number of simulations.
        for _ in tqdm(range(self.num_simulations), desc="MCTS Search", leave=False):
            self.rollout(root)
        # 7. return the root node.
        return root
    
    def reset(self):
        # nothing to do here.
        pass
    
    def init_tree(self,
        observation: types.Observation,
        last_action: Optional[types.Action] = None,
    ):
        # 1. get root node.
        if last_action is not None:
            # 1.1 find the new root node, if last action is given.
            old_root = self.get_root()
            new_root = self.get_child(old_root, last_action)
            # make all other branches of the tree unreachable
            self.set_root(new_root)
        else:
            # 1.2 otherwise, create a new root node.
            # delete the tree.
            self.reset_tree()
            new_root = self.get_root()
        # 2. get prior policy for the root node.
        if not new_root.expanded:
            # 2.1 if the root node is not expanded, we need a prior policy.
            # Evaluate the prior policy for this state.
            prior, _ = self.evaluation(observation)
            assert prior.shape == (self.num_actions,), f"Expected prior shape {(self.num_actions,)}, got {prior.shape}."
        else:
            # 2.2 if the root node is already expanded, we can use its prior.
            prior = new_root.prior
        
        # 3. Add exploration noise to the prior.
        noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
        prior = prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
        new_root.expand(prior) # it's fine to re-expand.
        
        # 4. Set legal actions if not already set.
        if not new_root.legal_actions.any():
            # 4.1 if the legal actions are not set, we need to get them from the model.
            legal_actions = self.model.legal_actions()
            new_root.set_legal_actions(legal_actions)
        
        # 5. return the root node.
        logger.debug(f"Initialized MCTS tree with root node: {new_root}.")
        return new_root
    
    def rollout(self, root:NodeBase) -> None:
        trajectory, actions = self.in_tree(root)
        if trajectory is None or actions is None:
            logger.warning("No valid trajectory found during MCTS rollout. Skipping.")
            return
        # 3. simulate the trajectory.
        node, timestep = self.simulate(trajectory[-1], trajectory[:-2], actions)
        # 4. evaluate (and expand) the node.
        _, value = self.evaluate(node, timestep)
        # 5. backup the value to the trajectory.
        self.backup(node, value)
        # 6. reset the model to the saved state.
        self.model.load_checkpoint()
    
    def in_tree(self, node: NodeBase) -> Tuple[List[NodeBase], List[types.Action]]:
        """Run the in-tree phase of MCTS and return the trajectory and actions."""
        # Start a new simulation from the top.
        logger.debug(f"Starting in-tree search from node: {node}.")
        trajectory = [node]
        actions = []
        while node.expanded:
            action = self.search_policy(node)
            # use `select_child` instead of `get_child` in case we need to do extra processing.
            node = self.select_child(node, action)
            # assert node not in trajectory, "Cycle detected. Node: {}; write head: {}; index {}; Trajectory: {}".format(node, self._local_write_head, self.header.available[self._local_write_head-2:self._local_write_head+2], trajectory)
            trajectory.append(node)
            actions.append(action)
        logger.debug(f"Selected actions: {actions} with trajectory: {trajectory}.")
        # Return the trajectory and actions.
        return trajectory, actions

    def simulate(self, node: NodeBase, ancestors: List[NodeBase], actions: List[types.Action])-> Tuple[NodeBase, types.Observation]:
        # Replay the simulator until the current node and expand it.
        timestep = self.model.step(actions)
        # depending on the model the reward might contain additional dimensions
        reward = timestep.reward if timestep.reward.ndim == 0 else timestep.reward[0]
        terminal = timestep.last()
        if not terminal:
            legal_actions = self.model.legal_actions()
            node.set_legal_actions(legal_actions)
        node.set_terminal(terminal)
        # backup nr.1: backup the observed reward
        node.parent.backup_reward(node.action, reward, self.discount, trajectory=ancestors)
        return node, timestep.observation
    
    def evaluate(self, node: NodeBase, observation: types.Observation) -> Tuple[np.ndarray, float]:
        # Calculate the bootstrap for leaf nodes.
        if node.terminal:
            # If terminal, there is no bootstrap value.
            prior = None
            value = 0.
        else:
            # Otherwise, bootstrap from this node with our value function.
            prior, value = self.evaluation(observation)
            # We also want to expand this node for next time.
            node.expand(prior)
        return prior, value
    
    def backup(self, node: NodeBase, value: float, trajectory: Optional[List[NodeBase]] = []):
        # Monte Carlo back-up with bootstrap from value function.
        logger.debug(f"MCTSBase.backup(): Backing up value {value} for node: {node}. actual root: {self.get_root()}.")
        assert node.parent is not None, f"Node node cannot be root, is it? {node.is_root()}"
        node.parent.backup_value(node.action, value, self.discount, trajectory=trajectory)
        # done.
    
    def get_root(self) -> NodeBase:
        return self._root
    def set_root(self, node: NodeBase):
        """Set the root node of the MCTS tree."""
        node.set_root()  # Ensure the node has no parent.
        self._root = node
    
    def reset_tree(self):
        """Reset the MCTS tree."""
        self._root = self._make_node()
    
    def get_child(self, node: NodeBase, action: types.Action) -> NodeBase:
        assert node.expanded, "NodeBase must be expanded to get a child."
        maybe_child = node.get_child(action)
        if maybe_child is None:
            child = self._make_node(parent=node, action=action)
            node.set_child(action, child)
            return child
        return maybe_child
    
    def select_child(self, node: NodeBase, action: types.Action) -> NodeBase:
        """Select a child node based on the action."""
        # This method can be overridden to implement custom selection logic.
        return self.get_child(node, action)
    
    def _make_node(self, parent: Optional[NodeBase] = None, action: Optional[types.Action] = None) -> NodeBase:
        """Create a new node and add it to the tree."""
        return self.node_cls(parent=parent, action=action)

def dyn_puct(
    node: NodeBase,
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

def puct(node: NodeBase, ucb_scaling: float = 1.) -> types.Action:
    """PUCT search policy, i.e. UCT with 'prior' policy."""
    # Action values Q(s,a) = R(s,a) / Nr(s,a) + W(s,a) / Nw(s,a).
    value_scores = node.children_values
    # check_numerics(value_scores)

    # Policy prior P(s,a).
    priors = node.prior
    # check_numerics(priors)

    # Visit ratios.
    # sqrt(Nr(s))/1+Nr(s,a) according to Silver et al. 2016.
    nominator = np.sqrt(node.visit_count)
    visit_ratios = nominator / (node.children_visits + 1)
    # check_numerics(visit_ratios)

    # Combine.
    puct_scores = value_scores + ucb_scaling * priors * visit_ratios
    return argmax(puct_scores, node.legal_actions)

argmax_rng = np.random.default_rng(42)  # Ensure we have a random number generator.
def argmax(values: np.ndarray, mask: np.ndarray) -> types.Action:
    """Argmax with random tie-breaking."""
    # check_numerics(values)
    max_value = np.max(values*mask) # mask the values to only consider valid actions
    return np.int32(argmax_rng.choice(np.arange(values.shape[0])[values == max_value]))

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

    def __call__(self, node: NodeBase) -> int:
        return dyn_puct(node, self.c_puct_base, self.c_puct_init)

def visit_count_policy(root: NodeBase, temperature: float = 1.0, mask: np.ndarray = None) -> int:
    visits = root.children_visits
    if mask is not None:
        visits = visits * mask # multiply by the mask to keep the shape, but make invalid actions impossible to choose    
    rescaled_visits = visits**(1 / temperature)
    probs = rescaled_visits / np.sum(rescaled_visits)
    # check_numerics(probs)
    
    return probs
