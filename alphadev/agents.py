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

"""Defines the distributed MCTS agent topology via Launchpad."""

from typing import Callable, Optional, Sequence, Dict
import pickle

import acme
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf.mcts import learning
from acme.agents.tf.mcts import models
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts import search
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
import acme.utils
import acme.utils.observers
import acme.utils.observers.base
import dm_env
import reverb
import sonnet as snt
from tempfile import TemporaryFile
import tree
import yaml
import sys
import os
from uuid import uuid4 as uuid

import numpy as np
from .dual_value_az import DualValueAZLearner, DualValueMCTSActor
from .network import NetworkFactory, make_input_spec
from .acting import MCTSActor
from .search.base import PUCTSearchPolicy
from .observers import MCTSObserver
from .service.service import Program, ReverbService, RPCService, RPCClient
from .service.inference_service import InferenceFactory, AlphaDevInferenceClient
from .config import AlphaDevConfig
from .device_config import DeviceAllocationConfig
from .service.variable_service import VariableService


class MCTS(agent.Agent):
    """A single-process MCTS agent."""

    def __init__(
        self,
        model: models.Model,
        network_factory: NetworkFactory,
        optimizer: snt.Optimizer,
        search_policy: Callable[[search.Node], types.Action],
        temperature_fn: Callable[[int], float],
        n_step: int,
        discount: float,
        replay_capacity: int,
        environment_spec: specs.EnvironmentSpec,
        batch_size: int,
        # Search parameters
        num_simulations: int,
        dirichlet_alpha: float = 1.0,
        exploration_fraction: float = 0.0,
        search_retain_subtree: bool = True,
        use_apv_mcts: bool = False,
        # APV MCTS parameters
        inference_factory: Optional[InferenceFactory] = None,
        apv_processes_per_pool: Optional[int] = None,
        virtual_loss_const: Optional[float] = None,
        # Other parameters
        use_dual_value_network: bool = False,
        logger: loggers.Logger = None,
        mcts_observers: Optional[Sequence[MCTSObserver]] = [],
    ):
        if use_apv_mcts:
            print('WARNING: APV_MCTS cannot be used in single-threaded mode. Support only for testing purposes.')

        extra_spec = {
            'pi':
                specs.Array(
                    shape=(environment_spec.actions.num_values,), dtype=np.float32)
        }
        # Create a replay server for storing transitions.
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=replay_capacity,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(
                environment_spec, extra_spec))
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f'localhost:{self._server.port}'
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount)

        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(server_address=address)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        network = network_factory(make_input_spec(environment_spec.observations))
        tf2_utils.create_variables(network, [environment_spec.observations])

        # Now create the agent components: actor & learner.
        self.counter = counting.Counter(None)
        self.logger = logger

        mcts_observers = mcts_observers(logger)

        if use_dual_value_network:
            actor = DualValueMCTSActor(
                environment_spec=environment_spec,
                model=model,
                discount=discount,
                num_simulations=num_simulations,
                search_policy=search_policy,
                temperature_fn=temperature_fn,
                dirichlet_alpha=dirichlet_alpha,
                exploration_fraction=exploration_fraction,
                search_retain_subtree=search_retain_subtree,
                # implementation-specific parameters
                # NOTE: we do not support APV MCTS in single-threaded agents.
                use_apv_mcts=use_apv_mcts,
                # single-threaded mode
                network=network_factory if not use_apv_mcts else None,
                variable_client=None, # no need for variable service in single-threaded mode
                # APV MCTS mode
                inference_factory=inference_factory,
                apv_processes_per_pool=apv_processes_per_pool,
                virtual_loss_const=virtual_loss_const,
                # other
                adder=adder,
                counter=self.counter,
                observers=mcts_observers,
                name='MCTSActor'
            )
            learner = DualValueAZLearner(
                network=network,
                optimizer=optimizer,
                dataset=dataset,
                discount=discount,
                logger=self.logger,
                counter=self.counter,
            )
        else:
            actor = MCTSActor(
                environment_spec=environment_spec,
                model=model,
                discount=discount,
                num_simulations=num_simulations,
                search_policy=search_policy,
                temperature_fn=temperature_fn,
                dirichlet_alpha=dirichlet_alpha,
                exploration_fraction=exploration_fraction,
                search_retain_subtree=search_retain_subtree,
                # implementation-specific parameters
                # NOTE: we do not support APV MCTS in single-threaded agents.
                use_apv_mcts=use_apv_mcts,
                # single-threaded mode
                network=network_factory if not use_apv_mcts else None,
                variable_client=None, # no need for variable service in single-threaded mode
                # APV MCTS mode
                inference_factory=inference_factory,
                apv_processes_per_pool=apv_processes_per_pool,
                virtual_loss_const=virtual_loss_const,
                # other
                adder=adder,
                counter=self.counter,
                observers=mcts_observers,
                name='MCTSActor'
            )
            learner = learning.AZLearner(
                network=network,
                optimizer=optimizer,
                dataset=dataset,
                discount=discount,
                logger=self.logger,
                counter=self.counter,
            )

        # The parent class combines these together into one 'agent'.
        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=100, # have at least some data in replay before starting the learner
            observations_per_step=1,# deterministic and fully observable
        )


class DistributedMCTS:
    """Distributed MCTS agent."""

    def __init__(
        self,
        # device configuration for the different processes.
        device_config: DeviceAllocationConfig,
        # basic params
        environment_factory: Callable[[], dm_env.Environment],
        network_factory: Callable[[specs.DiscreteArray], snt.Module],
        model_factory: Callable[[specs.EnvironmentSpec], models.Model],
        optimizer_factory: Callable[[], snt.Optimizer],
        environment_spec: specs.EnvironmentSpec,
        # search
        search_policy: Callable[[search.Node], types.Action],
        temperature_fn: Callable[[int], float],
        num_actors: int,
        num_simulations: int = 50,
        discount: float = 0.99,
        variable_update_period: int = 1000,
        dirichlet_alpha: float = 1.0,
        exploration_fraction: float = 0.0,
        search_retain_subtree: bool = True,
        use_apv_mcts: bool = False,
        # training
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        samples_per_insert: float = 32.0,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        importance_sampling_exponent: float = 0.2,
        priority_exponent: float = 0.6,
        n_step: int = 5,
        learning_rate: float = 1e-3,
        # APV MCTS parameters
        apv_processes_per_pool: Optional[int] = None,
        virtual_loss_const: Optional[float] = None,
        # inference server 
        use_inference_server: bool = False,
        inference_factory: Optional[InferenceFactory] = None,
        # Other parameters
        use_dual_value_network: bool = False,
        logger_factory: Callable[[], loggers.Logger] = None,
        observers: Optional[acme.utils.observers.base.EnvLoopObserver] = [],
        mcts_observers: Optional[Sequence[MCTSObserver]] = [],
    ):
        # check parameters

        # Internalize the device configuration.
        self._device_config = device_config

        # These factories create the relevant components on the workers.
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._model_factory = model_factory
        self._optimizer_factory = optimizer_factory
        self._env_spec = environment_spec

        # Search-related parameters
        self._search_policy = search_policy
        self._temperature_fn = temperature_fn
        self._num_actors = num_actors
        self._num_simulations = num_simulations
        self._discount = discount
        self._variable_update_period = variable_update_period
        self._dirichlet_alpha = dirichlet_alpha
        self._exploration_fraction = exploration_fraction
        self._search_retain_subtree = search_retain_subtree
        self._use_apv_mcts = use_apv_mcts
        self._use_inference_server = use_inference_server
        
        # Training parameters
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size
        self._target_update_period = target_update_period
        self._samples_per_insert = samples_per_insert
        self._min_replay_size = min_replay_size
        self._max_replay_size = max_replay_size
        self._importance_sampling_exponent = importance_sampling_exponent
        self._priority_exponent = priority_exponent
        self._n_step = n_step
        self._learning_rate = learning_rate
        # If using APV MCTS, we need to provide the inference factory and related parameters.
        if use_apv_mcts:
            assert apv_processes_per_pool is not None, 'Number of processes per pool must be provided for APV MCTS.'
            assert virtual_loss_const is not None, 'Virtual loss constant must be provided for APV MCTS.'
            self._search_num_actors = apv_processes_per_pool
            self._search_virual_loss_const = virtual_loss_const
        else:
            # make sure we don't pass these accidentally.
            self._inference_factory = inference_factory
            self._search_num_actors = apv_processes_per_pool
            self._search_virual_loss_const = virtual_loss_const
        # If using inference server, we need to provide the inference factory.
        if use_inference_server:
            assert inference_factory is not None, 'Inference factory must be provided when using inference server.'
            self._inference_factory = inference_factory
        
        self._use_dual_value_network = use_dual_value_network
        # set up logging
        if logger_factory is not None:
            self._logger_factory = logger_factory
        else:
            self._logger_factory = lambda: loggers.make_default_logger(
                'distributed_mcts', time_delta=30.0)
        # save observers
        self._observers = observers
        self._mcts_observers = mcts_observers

    def replay(self):
        """The replay storage worker."""
        limiter = reverb.rate_limiters.SampleToInsertRatio(
            min_size_to_sample=self._min_replay_size,
            samples_per_insert=self._samples_per_insert,
            error_buffer=self._batch_size)
        extra_spec = {
            'pi':
                specs.Array(
                    shape=(self._env_spec.actions.num_values,), dtype='float32')
        }
        signature = adders.NStepTransitionAdder.signature(self._env_spec,
                                                        extra_spec)
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._max_replay_size,
            rate_limiter=limiter,
            signature=signature)
        return [replay_table]

    def learner(self, replay: reverb.Client, counter: counting.Counter,
                variable_service: VariableService,
                logger: loggers.Logger):
        """The learning part of the agent."""
        # Create the networks.
        network = self._network_factory(make_input_spec(self._env_spec.observations))
        print('learner network created')
        tf2_utils.create_variables(network, [self._env_spec.observations])
        print('learner make dataset server_address:', replay.server_address)
        # The dataset object to learn from.
        dataset = datasets.make_reverb_dataset(
            server_address=replay.server_address,
            batch_size=self._batch_size,
            prefetch_size=self._prefetch_size)
        print('learner make dataset done')
        # Create the optimizer.
        optimizer = self._optimizer_factory()

        # Return the learning agent.
        if self._use_dual_value_network:
            return DualValueAZLearner(
                network=network,
                optimizer=optimizer,
                dataset=dataset,
                discount=self._discount,
                variable_service=variable_service,
                varibale_update_period=self._variable_update_period,
                logger=logger,
                counter=counter,
            )
        else:
            return learning.AZLearner(
                network=network,
                discount=self._discount,
                dataset=dataset,
                optimizer=optimizer,
                variable_service=variable_service,
                varibale_update_period=self._variable_update_period,
                logger=logger,
                counter=counter,
            )

    def actor(
        self,
        index: int,
        replay: reverb.Client,
        counter: counting.Counter,
        logger: loggers.Logger,
        variable_service: Optional[VariableService] = None,
        inference_service: Optional[AlphaDevInferenceClient] = None,
    ) -> acme.EnvironmentLoop:
        """The actor process."""

        # Build environment, model, network.
        environment = self._environment_factory()
        model = self._model_factory(self._env_spec)

        mcts_observers = self._mcts_observers(logger)
        
        if self._use_inference_server:
            assert inference_service is not None, 'Inference service must be provided when using inference server.'
            network = None
        else:
            network = self._network_factory(make_input_spec(self._env_spec.observations))
            variable_client = tf2_variable_utils.VariableClient(
                client=variable_service,
                variables={'network': network.trainable_variables},
                update_period=self._variable_update_period,
            )
        
        # Component to add things into replay.
        adder = adders.NStepTransitionAdder(
            client=replay,
            n_step=self._n_step,
            discount=self._discount,
        )

        if self._use_dual_value_network:
            actor = DualValueMCTSActor(
                environment_spec=self._env_spec,
                model=model,
                discount=self._discount,
                num_simulations=self._num_simulations,
                search_policy=self._search_policy,
                temperature_fn=self._temperature_fn,
                dirichlet_alpha=self.dirichlet_alpha,
                exploration_fraction=self._exploration_fraction,
                search_retain_subtree=self._search_retain_subtree,
                # implementation-specific parameters
                use_apv_mcts=self._use_apv_mcts,
                # single-threaded mode
                network=network,
                variable_client=variable_client,
                # APV MCTS mode
                inference_service=inference_service,
                apv_processes_per_pool=self._search_num_actors,
                virtual_loss_const=self._search_virual_loss_const,
                # other
                adder=adder,
                counter=counter,
                observers=mcts_observers,
                name=f'actor/{index}'
            )
        else:
            # Create the agent.
            actor = MCTSActor(
                environment_spec=self._env_spec,
                model=model,
                discount=self._discount,
                num_simulations=self._num_simulations,
                search_policy=self._search_policy,
                temperature_fn=self._temperature_fn,
                dirichlet_alpha=self.dirichlet_alpha,
                exploration_fraction=self._exploration_fraction,
                search_retain_subtree=self._search_retain_subtree,
                # implementation-specific parameters
                use_apv_mcts=self._use_apv_mcts,
                # single-threaded mode
                network=network,
                variable_client=variable_client,
                # APV MCTS mode
                inference_service=self._inference_service,
                apv_processes_per_pool=self._search_num_actors,
                virtual_loss_const=self._search_virual_loss_const,
                # other
                adder=adder,
                counter=counter,
                observers=mcts_observers,
                name=f'actor/{index}'
            )

        observers = self._observers(logger)

        # Create the loop to connect environment and agent.
        return acme.EnvironmentLoop(
            environment=environment,
            actor=actor,
            counter=counter,
            logger=logger,
            label='actor',
            observers=observers)

    def evaluator(
        self,
        counter: counting.Counter,
        logger: loggers.Logger,
        variable_service: VariableService = None,
    ):
        """The evaluation process."""
        # Build environment, model, network.
        environment = self._environment_factory()
        model = self._model_factory(self._env_spec)

        mcts_observers = self._mcts_observers(logger)
        
        if self._use_apv_mcts:
            network = variable_client = None
            self._inference_factory.set_variable_service(variable_service)
        else:
            network = self._network_factory(make_input_spec(self._env_spec.observations))
            variable_client = tf2_variable_utils.VariableClient(
                client=variable_service,
                variables={'network': network.trainable_variables},
                update_period=self._variable_update_period,
            )
        
        if self._use_dual_value_network:
            actor = DualValueMCTSActor(
                environment_spec=self._env_spec,
                model=model,
                discount=self._discount,
                num_simulations=self._num_simulations,
                search_policy=self._search_policy,
                temperature_fn=self._temperature_fn,
                dirichlet_alpha=self.dirichlet_alpha,
                exploration_fraction=self._exploration_fraction,
                search_retain_subtree=self._search_retain_subtree,
                # implementation-specific parameters
                use_apv_mcts=self._use_apv_mcts,
                # single-threaded mode
                network=network,
                variable_client=variable_client,
                # APV MCTS mode
                inference_factory=self._inference_factory,
                apv_processes_per_pool=self._search_num_actors,
                virtual_loss_const=self._search_virual_loss_const,
                # other
                counter=counter,
                observers=mcts_observers,
                name=f'evaluator'
            )
        else:
            # Create the agent.
            actor = MCTSActor(
                environment_spec=self._env_spec,
                model=model,
                discount=self._discount,
                num_simulations=self._num_simulations,
                search_policy=self._search_policy,
                temperature_fn=self._temperature_fn,
                dirichlet_alpha=self.dirichlet_alpha,
                exploration_fraction=self._exploration_fraction,
                search_retain_subtree=self._search_retain_subtree,
                # implementation-specific parameters
                use_apv_mcts=self._use_apv_mcts,
                # single-threaded mode
                network=network,
                variable_client=variable_client,
                # APV MCTS mode
                inference_factory=self._inference_factory,
                apv_processes_per_pool=self._search_num_actors,
                virtual_loss_const=self._search_virual_loss_const,
                # other
                counter=counter,
                observers=mcts_observers,
                name=f'evaluator'
            )

        observers = self._observers(logger)

        return acme.EnvironmentLoop(
            environment, actor, counter=counter, logger=logger, observers=observers, label='evaluator')

    def build(self, config: AlphaDevConfig):
        """Builds the distributed agent topology."""
        program = Program()

        with program.group('replay'):
            replay = program.add_service(ReverbService(
                priority_tables_fn=self.replay, port=config.replay_server_port))

        variable_service = VariableService(config)

        with program.group('counter'):
            counter: RPCClient = program.add_service(
                RPCService(
                    conn_config=config.distributed_backend_config,
                    instance_factory=counting.Counter,
                    instance_cls=counting.Counter,
                )
            )

        with program.group('logger'):
            logger = program.add_service(
                RPCService(
                    conn_config=config.distributed_backend_config,
                    instance_factory=self._logger_factory,
                    instance_cls=loggers.Logger,
                    )
                )

        with program.group('learner'):
            learner_device_config = self._device_config.get(
                DeviceAllocationConfig.make_process_key(
                    DeviceAllocationConfig.LEARNER_PROCESS
                )
            , None)
            program.add_service(
                RPCService(
                    conn_config=config.distributed_backend_config,
                    instance_factory=self.learner,
                    instance_cls=learning.AZLearner,
                    args=(replay, counter, variable_service, logger)),
                device_config=learner_device_config,
                )

        with program.group('evaluator'):
            eval_device_config = self._device_config.get(
                DeviceAllocationConfig.make_process_key(
                    DeviceAllocationConfig.ACTOR_PROCESS, 0
                ), None
            )
            program.add_service(
                RPCService(
                    conn_config=config.distributed_backend_config,
                    instance_factory=self.evaluator,
                    instance_cls=acme.EnvironmentLoop,
                    args=(counter, logger, variable_service),
                ),
                device_config=eval_device_config
                )

        with program.group('actor'):
            for idx in range(self._num_actors):
                if self._use_inference_server:
                    inference_device_config = self._device_config.get(
                        DeviceAllocationConfig.make_process_key(
                            DeviceAllocationConfig.INFERENCE_PROCESS, idx
                        ), None
                    )
                    # Create the inference service for this actor.
                    inference = self._inference_factory(variable_service=variable_service, label=f'inference/{idx}')
                    program.add_service(
                        inference,
                        device_config=inference_device_config,
                    )
                    variable_service = None # don't pass it to the actor, it will use the inference service instead
                else:
                    inference = None
                
                actor_device_config = self._device_config.get(
                    DeviceAllocationConfig.make_process_key(
                        DeviceAllocationConfig.ACTOR_PROCESS, idx
                    )
                , None)
                program.add_service(
                    RPCService(
                        conn_config=config.distributed_backend_config,
                        instance_factory=self.actor,
                        instance_cls=acme.EnvironmentLoop,
                        args=(
                            idx, replay, counter, logger, variable_service, inference
                        ),),
                    device_config=actor_device_config, # NOTE: only used when using single-threaded actors
                    )
        
        return program
