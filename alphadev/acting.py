"""
Reimplementation of `acme.agents.tf.mcts.acting` that doesn't such so much.
"""
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A MCTS actor."""

from typing import Optional, Tuple, Sequence, Callable
from collections import namedtuple
import functools

from acme.utils import counting
from acme import adders
from acme import specs
from acme.agents.tf.mcts.acting import MCTSActor as acmeMCTSActor
from acme.agents.tf.mcts import models
from acme.agents.tf.mcts import types
from acme.tf import variable_utils as tf2_variable_utils


import dm_env
import numpy as np
from scipy import special
import sonnet as snt
import tensorflow as tf
import tree

from .observers import MCTSObserver
from .search import visit_count_policy, mcts, Node
from .search_v2 import APV_MCTS
from .network import NetworkFactory, make_input_spec
from .service.variable_service import VariableService

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MCTSActor(acmeMCTSActor):
    """Executes a policy- and value-network guided MCTS search."""
    _st_node = Node # node implementation for single-threaded MCTS.
    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        model: models.Model,
        discount: float,
        num_simulations: int,
        search_policy: callable,
        temperature_fn: callable,
        dirichlet_alpha: float = 1.0,
        exploration_fraction: float = 0.0,
        search_retain_subtree: bool = True,
        # implementation-specific parameters
        use_apv_mcts: bool = False, # whether to use APV MCTS or single-threaded MCTS
        # single-threaded mode
        network: Optional[snt.Module] = None,
        variable_client: Optional[Callable] = None,
        # APV MCTS mode
        inference_factory: Optional[NetworkFactory] = None,
        apv_processes_per_pool: Optional[int] = None,
        virtual_loss_const: Optional[float] = None,
        # other
        adder: Optional[adders.Adder] = None,
        counter: Optional[counting.Counter] = None,
        observers: Optional[Sequence[MCTSObserver]] = [],
        name: str = 'mcts_actor',
    ):
        # check parameters
        _required_params = (
            [inference_factory, apv_processes_per_pool, virtual_loss_const]
            if use_apv_mcts else
            [network]
        )
        assert all(param is not None for param in _required_params), \
            "If use_apv_mcts is True, inference_factory, apv_processes_per_pool and virtual_loss_const must be provided. " \
            "If use_apv_mcts is False, network_factory and variable_service must be provided."
        
        # make sure we don't pass the network to the parent class if apv is used.
        if use_apv_mcts: network = None; variable_client = None
        
        super().__init__(
            environment_spec=environment_spec,
            model=model,
            network=network, # NOTE: unused when use_apv_mcts is True
            discount=discount,
            num_simulations=num_simulations,
            adder=adder,
            variable_client=variable_client,
        )
        self.name = name
        
        self._dirichlet_alpha = dirichlet_alpha
        self._exploration_fraction = exploration_fraction
        self._retain_subtree = search_retain_subtree
        
        self._search_policy = search_policy
        self._temperature_fn = temperature_fn
        self._counter = counter
        self._observers = observers
        
        self._use_apv_mcts = use_apv_mcts
        
        if self._use_apv_mcts:
            self.mcts = APV_MCTS(
                model=self._model,
                search_policy=self._search_policy,
                num_simulations=self._num_simulations,
                num_actions=self._num_actions,
                num_actors=apv_processes_per_pool,
                inference_factory=inference_factory,
                discount=self._discount,
                dirichlet_alpha=self._dirichlet_alpha,
                exploration_fraction=self._exploration_fraction,
                const_vl=virtual_loss_const,
                retain_subtree=self._retain_subtree,
                do_profiling=False,
                observers=[], # TODO: implement
                name=f'{self.name}_search',
            )
        else:
            # unify the APIs of the two MCTS implementations.
            self.mcts = namedtuple('single_threaded_mcts', ['search'])(search=functools.partial(
                mcts,
                model=self._model,
                search_policy=self._search_policy,
                evaluation=self._forward,
                num_simulations=self._num_simulations,
                num_actions=self._environment_spec.actions._num_values,
                discount=self._discount,
                dirichlet_alpha=self._dirichlet_alpha,
                exploration_fraction=self._exploration_fraction,
                node_class=self.__class__._st_node,
            ))
        self.last_action = None  # Last action selected by the actor.

    def _forward(
        self, observation: types.Observation) -> Tuple[types.Probs, types.Value]:
        """Performs a forward pass of the policy-value network."""
        assert not self._use_apv_mcts, "Use APV_MCTS instead of _forward for MCTSActor."
        # fix over acme implementation: accepts structured observations
        logits, value = self._network(tree.map_structure(lambda o: tf.expand_dims(o, axis=0), observation))

        # Convert to numpy & take softmax.
        logits = logits.numpy().squeeze(axis=0)

        value = value.numpy().item()
        probs = special.softmax(logits)

        return probs, value

    def select_action(self, observation: types.Observation) -> types.Action:
        """Computes the agent's policy via MCTS."""
        if self._model.needs_reset:
            self._model.reset(observation)

        root = self.mcts.search(observation, self.last_action) # TODO: pass last root to keep subtree intact
        
        # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
        if self._counter is None:
            training_steps = 0
        else:
            training_steps = self._counter.get_counts()[self._counter.get_steps_key()]
        temperature = self._temperature_fn(training_steps)
        # get the action mask from the model
        if self._model.needs_reset:
            self._model.reset(observation)
        action_mask = self._model.legal_actions()
        # perform masked visit count policy
        probs = visit_count_policy(root, temperature=temperature, mask=action_mask)
        assert probs.shape == (self._num_actions,), f"Expected probs shape {(self._num_actions,)}, got {probs.shape}."
        # sample an action from the masked visit count policy
        self.last_action = np.int32(np.random.choice(self._actions, p=probs))

        # Save the policy probs so that we can add them to replay in `observe()`.
        self._probs = probs.astype(np.float32)

        for observer in self._observers:
            observer.on_action_selection(
                node=root, probs=probs, action=self.last_action,
                training_steps=training_steps, temperature=temperature)

        return self.last_action
    
    def update(self, wait: bool = False):
        """Fetches the latest variables from the variable source, if needed."""
        if self._variable_client and self._variable_client._client.has_variables():
            self._variable_client.update(wait)
