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

from typing import Optional, Tuple, Sequence, Dict
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
from .search import visit_count_policy, mcts
from .search_v2 import APV_MCTS
from .network import NetworkFactory, make_input_spec
from .service.variable_service import VariableService

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MCTSActor(acmeMCTSActor):
    """Executes a policy- and value-network guided MCTS search."""

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        model: models.Model,
        network_factory: NetworkFactory,
        discount: float,
        num_simulations: int,
        search_policy: callable,
        temperature_fn: callable,
        search_retain_subtree: bool = True,
        use_apv_mcts: bool = False,
        apv_processes_per_pool: int = 2,
        dirichlet_alpha: float = 1.0,
        exploration_fraction: float = 0.0,
        virtual_loss_const: float = -1.0,
        search_batch_size: int = 1,
        inference_device_config: Optional[Dict[str, str]] = None,
        variable_service: Optional[VariableService] = None,
        variable_update_period: int = 100,
        adder: Optional[adders.Adder] = None,
        counter: Optional[counting.Counter] = None,
        observers: Optional[Sequence[MCTSObserver]] = [],
    ):
        if use_apv_mcts:
            network = None
            variable_client = None
        else:
            network = NetworkFactory(make_input_spec(environment_spec.observations))
            variable_client = tf2_variable_utils.VariableClient(
                client=variable_service,
                variables={'network': network.trainable_variables},
                update_period=variable_update_period,
        )
        
        super().__init__(
            environment_spec=environment_spec,
            model=model,
            network=network, # NOTE: unused when use_apv_mcts is True
            discount=discount,
            num_simulations=num_simulations,
            adder=adder,
            variable_client=variable_client,
        )
        self._search_policy = search_policy
        self._temperature_fn = temperature_fn
        self._counter = counter
        self._observers = observers
        
        self._use_apv_mcts = use_apv_mcts
        if self._use_apv_mcts:
            self.mcts = APV_MCTS(
                model=model,
                search_policy=search_policy,
                network_factory=network_factory,
                num_simulations=num_simulations,
                num_actions=environment_spec.actions._num_values,
                num_actors=apv_processes_per_pool,
                discount=discount,
                dirichlet_alpha=dirichlet_alpha,
                exploration_fraction=exploration_fraction,
                const_vl=virtual_loss_const,
                batch_size=search_batch_size,
                variable_service=variable_service,
                variable_update_period=variable_update_period,
                network_factory_args=(environment_spec.observations),
                inference_device_config=inference_device_config,
                retain_subtree=search_retain_subtree,
                name='APV_MCTS'
            )
        else:
            self.mcts = namedtuple('single_threaded_mcts', ['search'])(search=functools.partial(
                mcts,
                model=self._model,
                search_policy=self._search_policy,
                evaluation=self._forward,
                num_simulations=self._num_simulations,
                num_actions=self._environment_spec.actions._num_values,
                discount=self._discount,
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
