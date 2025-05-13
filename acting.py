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

from typing import Optional, Tuple

import acme
from acme import adders
from acme import specs
from acme.agents.tf.mcts.acting import MCTSActor as acmeMCTSActor
from acme.agents.tf.mcts import models
from acme.agents.tf.mcts import search
from acme.agents.tf.mcts import types
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import numpy as np
from scipy import special
import sonnet as snt
import tensorflow as tf
import tree

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MCTSActor(acmeMCTSActor):
    """Executes a policy- and value-network guided MCTS search."""

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        model: models.Model,
        network: snt.Module,
        discount: float,
        num_simulations: int,
        search_policy: callable,
        temperature_fn: callable,
        adder: Optional[adders.Adder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        counter: Optional[acme.utils.counting.Counter] = None,
    ):
        super().__init__(
            environment_spec=environment_spec,
            model=model,
            network=network,
            discount=discount,
            num_simulations=num_simulations,
            adder=adder,
            variable_client=variable_client,
        )
        self._search_policy = search_policy
        self._temperature_fn = temperature_fn
        self._counter = counter

    def _forward(
        self, observation: types.Observation) -> Tuple[types.Probs, types.Value]:
        """Performs a forward pass of the policy-value network."""
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

        # Compute a fresh MCTS plan.
        root = search.mcts(
            observation,
            model=self._model,
            search_policy=self._search_policy, # FIX: use the given search policy.
            evaluation=self._forward,
            num_simulations=self._num_simulations,
            num_actions=self._num_actions,
            discount=self._discount,
        )

        # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
        training_steps = self._counter.get_counts(self._counter.get_steps_key())
        probs = search.visit_count_policy(root, temperature=self._temperature_fn(training_steps))
        action = np.int32(np.random.choice(self._actions, p=probs))

        # Save the policy probs so that we can add them to replay in `observe()`.
        self._probs = probs.astype(np.float32)

        return action