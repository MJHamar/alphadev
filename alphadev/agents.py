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
from .acting import MCTSActor
from .search import PUCTSearchPolicy
from .observers import MCTSObserver
from .loggers import LoggerService, LoggerServiceWrapper
from .service import Program, ReverbService, RPCService, RPCClient, SubprocessService
from .config import AlphaDevConfig
from .inference_service import InferenceClient
from .variable_service import VariableService


class MCTS(agent.Agent):
    """A single-process MCTS agent."""

    def __init__(
        self,
        network: snt.Module,
        model: models.Model,
        optimizer: snt.Optimizer,
        search_policy: Callable[[search.Node], types.Action],
        temperature_fn: Callable[[int], float],
        n_step: int,
        discount: float,
        replay_capacity: int,
        num_simulations: int,
        environment_spec: specs.EnvironmentSpec,
        batch_size: int,
        use_dual_value_network: bool = False,
        logger: loggers.Logger = None,
        mcts_observers: Optional[Sequence[MCTSObserver]] = [],
    ):

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

        tf2_utils.create_variables(network, [environment_spec.observations])

        # Now create the agent components: actor & learner.
        counter = counting.Counter(None)
        
        mcts_observers = mcts_observers(logger)

        if use_dual_value_network:
            actor = DualValueMCTSActor(
                environment_spec=environment_spec,
                model=model,
                network=network,
                discount=discount,
                adder=adder,
                num_simulations=num_simulations,
                search_policy=search_policy,
                temperature_fn=temperature_fn,
                counter=counter,
                observers=mcts_observers,
            )
            learner = DualValueAZLearner(
                network=network,
                optimizer=optimizer,
                dataset=dataset,
                discount=discount,
                logger=logger,
                counter=counter,
            )
        else:
            actor = MCTSActor(
                environment_spec=environment_spec,
                model=model,
                network=network,
                discount=discount,
                adder=adder,
                num_simulations=num_simulations,
                search_policy=search_policy,
                temperature_fn=temperature_fn,
                counter=counter,
                observers=mcts_observers,
            )
            learner = learning.AZLearner(
                network=network,
                optimizer=optimizer,
                dataset=dataset,
                discount=discount,
                logger=logger,
                counter=counter,
            )

        # The parent class combines these together into one 'agent'.
        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=10,
            observations_per_step=1,
        )


class DistributedMCTS:
    """Distributed MCTS agent."""

    def __init__(
        self,
        environment_factory: Callable[[], dm_env.Environment],
        network_factory: Callable[[specs.DiscreteArray], snt.Module],
        model_factory: Callable[[specs.EnvironmentSpec], models.Model],
        optimizer_factory: Callable[[], snt.Optimizer],
        search_policy: Callable[[search.Node], types.Action],
        temperature_fn: Callable[[int], float],
        num_actors: int,
        num_simulations: int = 50,
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
        discount: float = 0.99,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        variable_update_period: int = 1000,
        use_dual_value_network: bool = False,
        logger_factory: Callable[[], loggers.Logger] = None,
        observers: Optional[acme.utils.observers.base.EnvLoopObserver] = [],
        mcts_observers: Optional[Sequence[MCTSObserver]] = [],
    ):

        if environment_spec is None:
            environment_spec = specs.make_environment_spec(environment_factory())

        # These 'factories' create the relevant components on the workers.
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._model_factory = model_factory
        self._optimizer_factory = optimizer_factory
        self._search_policy = search_policy

        # Internalize hyperparameters.
        self._num_actors = num_actors
        self._num_simulations = num_simulations
        self._env_spec = environment_spec
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
        self._discount = discount
        self._variable_update_period = variable_update_period
        self._use_dual_value_network = use_dual_value_network
        self._temperature_fn = temperature_fn
        
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

    def inference(self, config: AlphaDevConfig):
        """
        The inference worker.
        Runs the inference service in a separate process.
        """
        # create a temporary file where we write the inference config.
        fname = os.path.abspath(os.path.curdir) + '/inference_config' + uuid().hex[:4] + '.yaml'
        
        pickle.dump(config, open(fname, 'wb'))
        # parameterize the subprocess call
        subp = [sys.executable,
            '-m',
            'alphadev.inference_service',
            fname]
        # create a handle for the inference service
        handle = InferenceClient(config)
        return subp, handle, fname

    def learner(self, replay: reverb.Client, counter: counting.Counter,
                logger: loggers.Logger):
        """The learning part of the agent."""
        # wrap the logger
        logger = LoggerServiceWrapper(logger)
        # Create the networks.
        network = self._network_factory(self._env_spec.actions)
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
                logger=logger,
                counter=counter,
            )
        else:
            return learning.AZLearner(
                network=network,
                discount=self._discount,
                dataset=dataset,
                optimizer=optimizer,
                logger=logger,
                counter=counter,
            )

    def actor(
        self,
        replay: reverb.Client,
        inference: InferenceClient,
        counter: counting.Counter,
        logger: loggers.Logger,
    ) -> acme.EnvironmentLoop:
        """The actor process."""
        # wrap the logger
        logger = LoggerServiceWrapper(logger)

        # Build environment, model, network.
        environment = self._environment_factory()
        model = self._model_factory(self._env_spec)
        
        mcts_observers = self._mcts_observers(logger)

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
                network=inference,
                discount=self._discount,
                adder=adder,
                num_simulations=self._num_simulations,
                search_policy=self._search_policy,
                temperature_fn=self._temperature_fn,
                counter=counter,
                observers=mcts_observers,
            )
        else:
            # Create the agent.
            actor = MCTSActor(
                environment_spec=self._env_spec,
                model=model,
                network=inference,
                discount=self._discount,
                adder=adder,
                num_simulations=self._num_simulations,
                search_policy=self._search_policy,
                temperature_fn=self._temperature_fn,
                counter=counter,
                observers=mcts_observers,
            )

        observers = self._observers(logger)

        # Create the loop to connect environment and agent.
        return acme.EnvironmentLoop(
            environment=environment,
            actor=actor,
            counter=counter,
            logger=logger,
            observers=observers)

    def evaluator(
        self,
        inference: InferenceClient,
        counter: counting.Counter,
        logger: loggers.Logger,
    ):
        """The evaluation process."""
        # wrap the logger
        logger = LoggerServiceWrapper(logger)

        # Build environment, model, network.
        environment = self._environment_factory()
        print('evaluator environment created', type(environment))
        model = self._model_factory(self._env_spec)
        print('evaluator model created: type: %s' % type(model))

        mcts_observers = self._mcts_observers(logger)

        if self._use_dual_value_network:
            actor = DualValueMCTSActor(
                environment_spec=self._env_spec,
                model=model,
                network=inference,
                discount=self._discount,
                num_simulations=self._num_simulations,
                search_policy=self._search_policy,
                temperature_fn=self._temperature_fn,
                counter=counter,
                observers=mcts_observers,
            )
        else:
            # Create the agent.
            actor = MCTSActor(
                environment_spec=self._env_spec,
                model=model,
                network=inference,
                discount=self._discount,
                num_simulations=self._num_simulations,
                search_policy=self._search_policy,
                temperature_fn=self._temperature_fn,
                counter=counter,
                observers=mcts_observers,
            )
        
        observers = self._observers(logger)

        return acme.EnvironmentLoop(
            environment, actor, counter=counter, logger=logger, observers=observers)

    def build(self, config: AlphaDevConfig):
        """Builds the distributed agent topology."""
        program = Program()

        with program.group('replay'):
            replay = program.add_service(ReverbService(
                priority_tables_fn=self.replay, port=config.replay_server_port))
            
        with program.group('inference'):
            inference = program.add_service(
                SubprocessService(
                    command_builder=self.inference, args=(config, ),)
            )

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

        # with program.group('learner'):
        #     learner = program.add_service(
        #         RPCService(
        #             conn_config=config.distributed_backend_config,
        #             instance_factory=self.learner,
        #             instance_cls=learning.AZLearner,
        #             args=(replay, counter, logger, variable_service))
        #         )

        with program.group('evaluator'):
            program.add_service(
                RPCService(
                    conn_config=config.distributed_backend_config,
                    instance_factory=self.evaluator,
                    instance_cls=acme.EnvironmentLoop,
                    args=(inference, counter, logger),
                    fork_worker=False)
                )

        # with program.group('actor'):
        #     for _ in range(self._num_actors):
        #         program.add_service(
        #             RPCService(
        #                 conn_config=config.distributed_backend_config,
        #                 instance_factory=self.actor,
        #                 instance_cls=acme.EnvironmentLoop,
        #                 args=(inference, replay, counter, logger))
        #             )

        return program
