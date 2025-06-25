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


import dm_env
import numpy as np
from scipy import special
import sonnet as snt
from .tf_util import tf
import tree

from .observers import MCTSObserver
from .search.mcts import MCTSBase, visit_count_policy, NodeBase as Node
from .search.apv_mcts import APV_MCTS
from .network import NetworkFactory, make_input_spec
from .service.variable_service import VariableService
from .service.inference_service import AlphaDevInferenceClient, InferenceNetworkFactory
from .device_config import DeviceConfig, ACTOR, CONTROLLER

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MCTSActor(acmeMCTSActor):
    """Executes a policy- and value-network guided MCTS search."""
    def __init__(
        self,
        device_config: DeviceConfig,
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
        network_factory: Optional[NetworkFactory] = None,
        variable_service: Optional[VariableService] = None,
        variable_update_period: int = 100,
        # APV MCTS mode
        inference_service: Optional[AlphaDevInferenceClient] = None,
        apv_processes_per_pool: Optional[int] = None,
        virtual_loss_const: Optional[float] = None,
        # other
        adder: Optional[adders.Adder] = None,
        counter: Optional[counting.Counter] = None,
        observers: Optional[Sequence[MCTSObserver]] = [],
        name: str = 'mcts_actor',
    ):
        """
        Initializes the MCTS actor. The actor performs MCTS search using a policy-value network each time select_action is called.
        
        There are three supported modes of searching:
        1. Single-threaded MCTS: Uses a single-threaded MCTS implementation with a network that is passed as an argument.
            - inference_service is ignored in this case.
        2. Asynchronous Policy-Value MCTS (APV_MCTS): Performs MCTS rollouts using N=`apv_processes_per_pool` worker processes.
            there are two modes:
            - 2.1. 'streamlined' mode: each worker uses its own copy of the network and performs all 4 phases of a rollout.
            - 2.2. 'alphago' mode: (described in Silver et al. 2016) N-1 workers perform tree search and simulation; a separate inference service 
                performs the policy and value evaluation. Finally, the N-th worker performs expansion and backup.
        If use_apv_mcts is False, we use the single-threaded MCTS implementation (mode 1). In this case, `network_factory` and `variable_service` are required and `inference_service` is ignored.
        Otherwise, if inference_service is not provided, we assume 'streamlined' mode (mode 2.1).
        finally, if inference_service is provided, we assume 'alphago' mode (mode 2.2). In this case, `network_factory` and `variable_service` is ignored.
        """
        super().__init__(
            environment_spec=environment_spec,
            model=model,
            network=None, # do not pass the network, we have our own logic.
            discount=discount,
            num_simulations=num_simulations,
            adder=adder,
            variable_client=None,
        )
        self.name = name
        self._device_config = device_config
        
        self._dirichlet_alpha = dirichlet_alpha
        self._exploration_fraction = exploration_fraction
        self._retain_subtree = search_retain_subtree
        
        self._search_policy = search_policy
        self._temperature_fn = temperature_fn
        self._counter = counter
        self._observers = observers
        
        self._use_apv_mcts = use_apv_mcts
        
        self.last_action = None  # Last action selected by the actor.
        
        if not self._use_apv_mcts:
            self._init_mode1(
                environment_spec=environment_spec,
                network_factory=network_factory,
                variable_service=variable_service,
                variable_update_period=variable_update_period,
            )
        else:
            if inference_service is None:
                # Use APV_MCTS in 'streamlined' mode.
                self._init_mode2_1(
                    environment_spec=environment_spec,
                    network_factory=network_factory,
                    variable_service=variable_service,
                    variable_update_period=variable_update_period,
                    apv_processes_per_pool=apv_processes_per_pool,
                    virtual_loss_const=virtual_loss_const,
                )
            else:
                # Use APV_MCTS in 'alphago' mode.
                self._init_mode2_2(
                    inference_service=inference_service,
                    apv_processes_per_pool=apv_processes_per_pool,
                    virtual_loss_const=virtual_loss_const,
                )
    
    def _make_eval_factory(self, network_factory, environment_spec, 
                           variable_service, variable_update_period):
        return InferenceNetworkFactory(
            network_factory=network_factory,
            observation_spec=environment_spec.observations,
            variable_service=variable_service,
            variable_update_period=variable_update_period,
        )
    
    def _init_mode1(
        self,
        environment_spec: specs.EnvironmentSpec,
        network_factory: NetworkFactory,
        variable_service: Optional[VariableService] = None,
        variable_update_period: int = 100,
    ):
        logger.info("Initializing MCTSActor in single-threaded mode.")
        # make an evaluation factory, which will instantiate the network, 
        # initialize its parameters and connect to the variable service.
        eval_factory = self._make_eval_factory(
            network_factory=network_factory,
            environment_spec=environment_spec,
            variable_service=variable_service,
            variable_update_period=variable_update_period,
        )
        # set the model's evaluation function to the evaluation factory.
        evaluation = eval_factory()
        # create the MCTS search object.
        self.mcts = MCTSBase(
            num_simulations=self._num_simulations,
            num_actions=self._num_actions,
            model=self._model,
            search_policy=self._search_policy,
            evaluation=evaluation,
            discount=self._discount,
            dirichlet_alpha=self._dirichlet_alpha,
            exploration_fraction=self._exploration_fraction,
        )

    def _init_mode2_1(self,
            environment_spec: specs.EnvironmentSpec,
            network_factory: NetworkFactory,
            variable_service: Optional[VariableService] = None,
            variable_update_period: int = 100,
            apv_processes_per_pool: Optional[int] = None,
            virtual_loss_const: Optional[float] = None,
    ):
        """Initializes the actor in 'streamlined' APV_MCTS mode."""
        logger.info("Initializing MCTSActor in APV_MCTS 'streamlined' mode.")
        # make an evaluation factory, which will instantiate the network, 
        # initialize its parameters and connect to the variable service.
        eval_factory = self._make_eval_factory(
            network_factory=network_factory,
            environment_spec=environment_spec,
            variable_service=variable_service,
            variable_update_period=variable_update_period,
        )
        self.mcts = APV_MCTS(
            device_config=self._device_config,
            num_simulations=self._num_simulations,
            num_actions=self._num_actions,
            model=self._model,
            search_policy=self._search_policy,
            num_workers= apv_processes_per_pool,
            inference_server=None,  # no inference server in this mode
            evaluation_factory=eval_factory,
            discount=self._discount,
            dirichlet_alpha=self._dirichlet_alpha,
            exploration_fraction=self._exploration_fraction,
            vl_constant=virtual_loss_const,
            # TODO pass lambda_ to give different weights to reward and value averages
            name=f'{self.name}_mcts'
            )
    
    def _init_mode2_2(
        self,
        inference_service: AlphaDevInferenceClient,
        apv_processes_per_pool: Optional[int] = None,
        virtual_loss_const: Optional[float] = None,
    ):
        """Initializes the actor in 'alphago' APV_MCTS mode."""
        logger.info("Initializing MCTSActor in APV_MCTS 'alphago' mode.")
        # In this mode, we use the inference service to perform policy and value evaluation.
        # The inference service is expected to be an instance of AlphaDevInferenceClient.
        if not isinstance(inference_service, AlphaDevInferenceClient):
            raise ValueError(f"Inference service must be an instance of AlphaDevInferenceClient not {type(inference_service)}.")
        
        self.mcts = APV_MCTS(
            device_config=self._device_config,
            num_simulations=self._num_simulations,
            num_actions=self._num_actions,
            model=self._model,
            search_policy=self._search_policy,
            num_workers=apv_processes_per_pool,
            inference_server=inference_service,  # use the inference service for evaluation
            evaluation_factory=None,  # no evaluation factory in this mode
            discount=self._discount,
            dirichlet_alpha=self._dirichlet_alpha,
            exploration_fraction=self._exploration_fraction,
            vl_constant=virtual_loss_const,
            # TODO pass lambda_ to give different weights to reward and value averages
            name=f'{self.name}_mcts'
        )
    
    def __del__(self):
        """Destructor to clean up resources."""
        if hasattr(self, 'mcts'):
            del self.mcts
    
    def _forward(
        self, observation: types.Observation) -> Tuple[types.Probs, types.Value]:
        """Performs a forward pass of the policy-value network."""
        raise RuntimeError("MCTSActor._forward shuld not be called. Use self.evaluation instead.")

    def select_action(self, observation: types.Observation) -> types.Action:
        """Computes the agent's policy via MCTS."""
        if self._model.needs_reset:
            self._model.reset(observation)
        
        root = self.mcts.search(observation=observation, last_action=self.last_action)
        # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
        if self._counter is None:
            training_steps = 0
        else:
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
        # same with the latency reward.
        self._latency_reward = self._model.get_latency_reward()
        
        for observer in self._observers:
            observer.on_action_selection(
                node=root, probs=probs, action=action,
                training_steps=training_steps, temperature=temperature, mcts=self.mcts)
        
        if self._retain_subtree:
            self.last_action = action
        
        return action
    
    def update(self, wait: bool = False):
        """Fetches the latest variables from the variable source, if needed."""
        # this is a no-op. Either the inference service or the evaluation factory takes care of updating the variables.
        pass

    def observe_first(self, timestep):
        # clear the last action to avoid leaking information between episodes.
        self.mcts.reset()
        self.last_action = None
        return super().observe_first(timestep)
    
    def observe(self, action: types.Action, next_timestep: dm_env.TimeStep):
        """Updates the agent's internal model and adds the transition to replay."""
        self._model.update(self._prev_timestep, action, next_timestep)
        self._prev_timestep = next_timestep
        if self._adder:
            self._adder.add(action, next_timestep, extras={'pi': self._probs, 'latency_reward': self._model.get_latency_reward()})
