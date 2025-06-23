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
from acme.utils import counting
from acme.utils import loggers
import acme.utils
import acme.utils.observers
import acme.utils.observers.base
import dm_env
import reverb
import sonnet as snt

from .dual_value_az import DualValueAZLearner
from .network import NetworkFactory, make_input_spec
from .acting import MCTSActor
from .observers import MCTSObserver
from .service.service import Program, ReverbService, RPCService, RPCClient, deploy_service, terminate_services
from .service.inference_service import AlphaDevInferenceClient, AlphaDevInferenceService
from .config import AlphaDevConfig
from .device_config import DeviceConfig, ACTOR, LEARNER, CONTROLLER
from .service.variable_service import VariableService
from .evaluation import EvaluationLoop
from .utils import reward_priority_fn

class MCTS(agent.Agent):
    """A single-process MCTS agent."""

    def __init__(
        self,
        device_config: DeviceConfig,
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
        # APV MCTS parameters
        use_apv_mcts: bool = False,
        apv_processes_per_pool: Optional[int] = None,
        virtual_loss_const: Optional[float] = None,
        # inference server parameters
        use_inference_server: bool = False,
        search_batch_size: int = 1,
        search_buffer_size: int = 3,
        # Other parameters
        training_steps: Optional[int] = None,
        use_dual_value_network: bool = False,
        use_target_network: bool = True,
        target_update_period: Optional[int] = None,
        logger: loggers.Logger = None,
        mcts_observers: Optional[Sequence[MCTSObserver]] = [],
    ):
        if use_apv_mcts:
            print('WARNING: APV_MCTS cannot be used in single-threaded mode. Support only for testing purposes.')
        assert not use_target_network or target_update_period is not None, \
            'Target update period must be provided if using target network.'
        
        extra_spec = {
            'pi': 
                specs.Array(shape=(environment_spec.actions.num_values,), dtype='float32'),
            'latency_reward':
                specs.Array(shape=(), dtype='float32'),
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
        
        if not use_apv_mcts or not use_inference_server:
            inference_client = None
            self.inference_handle = None
        elif use_inference_server:
            inference_service = AlphaDevInferenceService(
                num_blocks=search_buffer_size, # 2x the number of processes.
                network_factory=network_factory,
                input_spec=environment_spec.observations,
                output_spec=environment_spec.actions.num_values, # num actions
                batch_size=search_batch_size,
                variable_service=None,
                factory_args=([make_input_spec(environment_spec.observations)],), # for the network factory
            )
            # run the service
            inference_client = inference_service.create_handle()
            self.inference_handle = deploy_service(
                inference_service.run, device_config=device_config.get_config(ACTOR), label='inference_service')
        else:
            # If we are not using APV MCTS, we don't need an inference service.
            inference_client = None
            self.inference_handle = None
        
        # initialize the network we are training
        network = network_factory(make_input_spec(environment_spec.observations))
        tf2_utils.create_variables(network, [environment_spec.observations])
        if use_target_network:
            # If we are using a target network, we need to create it.
            # The target network is used to compute the value targets.
            target_network = network_factory(make_input_spec(environment_spec.observations))
            tf2_utils.create_variables(target_network, [environment_spec.observations])
        else:
            # If we are not using a target network, we can use the same network.
            target_network = None
        # we don't care about variable service here.
        
        # Now create the agent components: actor & learner.
        self.counter = counting.Counter(None)
        self.logger = logger
        
        mcts_observers = mcts_observers(logger)
        
        actor = MCTSActor(
            device_config=device_config,
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
            network_factory=network_factory,
            variable_service=None, # no need for variable service in single-threaded mode
            # APV MCTS mode
            inference_service=inference_client,
            apv_processes_per_pool=apv_processes_per_pool,
            virtual_loss_const=virtual_loss_const,
            # other
            adder=adder,
            counter=self.counter,
            observers=mcts_observers,
            name='MCTSActor'
        )
        if use_dual_value_network:
            learner = DualValueAZLearner(
                network=network,
                optimizer=optimizer,
                dataset=dataset,
                discount=discount,
                training_steps=training_steps,
                variable_service=None,
                target_network=target_network,
                target_update_period=target_update_period,
                logger=self.logger,
                counter=self.counter,
            )
        else:
            learner = learning.AZLearner(
                network=network,
                optimizer=optimizer,
                dataset=dataset,
                discount=discount,
                training_steps=training_steps,
                variable_service=None,
                target_network=target_network,
                target_update_period=target_update_period,
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

    def __del__(self):
        print('Shutting down MCTS agent...')
        if hasattr(self, '_actor'):
            del self._actor # delete the actor.
        if hasattr(self, '_learner'):
            del self._learner # delete the learner.
        if hasattr(self, '_counter') and hasattr(self, '_logger'):
            # delete the counter and logger.
            del self.counter; del self.logger # delete the counter and logger.
        # stop the inference service if it exists
        if self.inference_handle is not None:
            print('Shutting down inference service...')
            terminate_services(self.inference_handle, poll=False)

class DistributedMCTS:
    """Distributed MCTS agent."""

    def __init__(
        self,
        # device configuration for the different processes.
        device_config: DeviceConfig,
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
        # training
        do_train: bool = True,
        do_eval_based_updates: bool = True, # whether to use an evaluator process to see if the network should be updated.
        evaluation_update_threshold: float = 0.55, # if the evaluation cumulative reward is above this threshold, the network is updated.
        evaluation_episodes: int = 5,
        training_steps: int = 1000,
        batch_size: int = 256,
        prefetch_size: int = 4,
        use_target_network: bool = False,
        target_update_period: int = 100,
        samples_per_insert: float = 32.0,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        use_prioritized_replay: bool = True,
        priority_exponent: float = 0.6,
        n_step: int = 5,
        learning_rate: float = 1e-3,
        # APV MCTS parameters
        use_apv_mcts: bool = False,
        apv_processes_per_pool: Optional[int] = None,
        virtual_loss_const: Optional[float] = None,
        # inference server 
        use_inference_server: bool = False,
        search_batch_size: int = None,
        search_buffer_size: int = None,
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
        self._search_buffer_size = search_buffer_size
        self._search_batch_size = search_batch_size
        
        # Training parameters
        self._do_eval_based_updates = do_eval_based_updates
        self._evaluation_update_threshold = evaluation_update_threshold
        self._evaluation_episodes = evaluation_episodes
        
        self._do_train = do_train
        self._training_steps = training_steps
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size
        self._use_target_network = use_target_network
        self._target_update_period = target_update_period
        self._samples_per_insert = samples_per_insert
        self._min_replay_size = min_replay_size
        self._max_replay_size = max_replay_size
        self._use_prioritized_replay = use_prioritized_replay
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
            # self._inference_factory = 0
            self._search_num_actors = apv_processes_per_pool
            self._search_virual_loss_const = virtual_loss_const
        
        if use_prioritized_replay:
            # If we are using prioritized replay, we need to set up the priority functions.
            self.priortiy_fns: Dict[str, Callable[[types.Transition], float]] = {
                adders.DEFAULT_PRIORITY_TABLE: reward_priority_fn,
            }
        else:
            # If we are not using prioritized replay, we can use a uniform priority function.
            self.priortiy_fns = {
                adders.DEFAULT_PRIORITY_TABLE: None,
            }
        
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
                specs.Array(shape=(self._env_spec.actions.num_values,), dtype='float32'),
            'latency_reward':
                specs.Array(shape=(), dtype='float32'),
        }
        signature = adders.NStepTransitionAdder.signature(self._env_spec,
                                                        extra_spec)
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Prioritized(self._priority_exponent),
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
        tf2_utils.create_variables(network, [self._env_spec.observations])
        if self._use_target_network:
            # If we are using a target network, we need to create it.
            # The target network is used to compute the value targets.
            target_network = self._network_factory(make_input_spec(self._env_spec.observations))
            tf2_utils.create_variables(target_network, [self._env_spec.observations])
        else:
            # If we are not using a target network, we can use the same network.
            target_network = None
        
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
                training_steps=self._training_steps,
                target_network=target_network,
                target_update_period=self._target_update_period,
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
                target_network=target_network,
                target_update_period=self._target_update_period,
                training_steps=self._training_steps,
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
        inference_client: Optional[AlphaDevInferenceClient] = None,
        device_config: Optional[DeviceConfig] = None,
    ) -> acme.EnvironmentLoop:
        """The actor process."""

        # Build environment, model, network.
        # NOTE: both the model and the environment here are AssemblyGame instances.
        # we keep the two of them separate because that is what EnvironmentLoop expects.
        # also, this way it is easier to replace model with a learnt model (mu-zero style).
        environment = self._environment_factory()
        model = self._model_factory(self._env_spec)
        
        mcts_observers = self._mcts_observers(logger)
        
        # Component to add things into replay.
        adder = adders.NStepTransitionAdder(
            client=replay,
            n_step=self._n_step,
            discount=self._discount,
            priority_fns=self.priortiy_fns
        )
        
        actor = MCTSActor(
            device_config=device_config,
            environment_spec=self._env_spec,
            model=model,
            discount=self._discount,
            num_simulations=self._num_simulations,
            search_policy=self._search_policy,
            temperature_fn=self._temperature_fn,
            dirichlet_alpha=self._dirichlet_alpha,
            exploration_fraction=self._exploration_fraction,
            search_retain_subtree=self._search_retain_subtree,
            # implementation-specific parameters
            use_apv_mcts=self._use_apv_mcts,
            # single-threaded mode
            network_factory=self._network_factory,
            variable_service=variable_service,
            variable_update_period=self._variable_update_period,
            # APV MCTS mode
            inference_service=inference_client,
            apv_processes_per_pool=self._search_num_actors,
            virtual_loss_const=self._search_virual_loss_const,
            # other
            adder=adder,
            counter=counter,
            observers=mcts_observers,
            name=f'actor_{index}'
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
        variable_source: VariableService = None, # where the network uploads its variables
        variable_staging: VariableService = None, # where evaluator actors pull variables from.
        variable_service: VariableService = None, # from where the actors pull new variables.
        inference_client: Optional[AlphaDevInferenceClient] = None,
        device_config: Optional[DeviceConfig] = None,
    ):
        """The evaluation process."""
        # Build environment, model, network.
        environment = self._environment_factory()
        model = self._model_factory(self._env_spec)
        
        mcts_observers = self._mcts_observers(logger)
        
        actor = MCTSActor(
            device_config=device_config,
            environment_spec=self._env_spec,
            model=model,
            discount=self._discount,
            num_simulations=self._num_simulations,
            search_policy=self._search_policy,
            temperature_fn=self._temperature_fn,
            dirichlet_alpha=self._dirichlet_alpha,
            exploration_fraction=self._exploration_fraction,
            search_retain_subtree=self._search_retain_subtree,
            # implementation-specific parameters
            use_apv_mcts=self._use_apv_mcts,
            # single-threaded mode
            network_factory=self._network_factory,
            variable_service=variable_staging, # can be None
            variable_update_period=1, # update instantly
            # APV MCTS mode
            inference_service=inference_client, # can be None
            apv_processes_per_pool=self._search_num_actors,
            virtual_loss_const=self._search_virual_loss_const,
            # other
            counter=counter,
            observers=mcts_observers,
            name=f'evaluator'
        )
        
        observers = self._observers(logger)
        
        return EvaluationLoop(
            environment, actor,
            source_service=variable_source,
            staging_service=variable_staging,
            variable_service=variable_service,
            should_update_threshold=self._evaluation_update_threshold,
            evaluation_episodes=self._evaluation_episodes,
            counter=counter, logger=logger, observers=observers, label='evaluator')
        
    def build(self, config: AlphaDevConfig):
        """Builds the distributed agent topology."""
        program = Program()
        
        with program.group('replay'):
            replay = program.add_service(ReverbService(
                priority_tables_fn=self.replay, port=config.replay_server_port))
        
        if self._do_train:
            variable_service = VariableService(config)
        else:
            variable_service = None
        
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
        if self._do_train:
            with program.group('learner'):
                learner_device_config = self._device_config.get_config(LEARNER)
                program.add_service(
                    RPCService(
                        conn_config=config.distributed_backend_config,
                        instance_factory=self.learner,
                        instance_cls=learning.AZLearner,
                        args=(replay, counter, variable_service, logger)),
                    device_config=learner_device_config,
                    )

        if self._do_eval_based_updates:
            # the current variable_service becomes the staging service (the network is already parameterized with it)
            # and we create a new variable service that will be passed to the actors.
            variable_source = variable_service
            # create a staging service (where we load latest parameters for evaluation)
            variable_staging = VariableService(config)
            # and a new variable service where we push variables that performed better than the threshold.
            variable_service = VariableService(config)
            # disable checkpointing in the stagging and source services and set checkpointing to 1 in the new variable service.
            variable_source._checkpoint_dir = None
            variable_staging._checkpoint_dir = None
            variable_service._checkpoint_every = 1
            with program.group('evaluator'):
                eval_device_config = self._device_config.get_config(ACTOR)
                if self._use_apv_mcts:
                    actor_subconfig = self._device_config.make_subconfig(self._search_num_actors)
                elif not self._use_apv_mcts or self._use_inference_server:
                    actor_subconfig = self._device_config.make_subconfig(0) # no new GPU actors will be created in the subprocess
                if self._use_inference_server:
                    eval_inference_client = program.add_service(
                        AlphaDevInferenceService(
                            num_blocks=self._search_buffer_size, # 2x the number of processes.
                            network_factory=self._network_factory,
                            input_spec=self._env_spec.observations,
                            output_spec=self._env_spec.actions.num_values, # num actions
                            batch_size=self._search_batch_size,
                            variable_service=variable_service,
                            variable_update_period=self._variable_update_period,
                            factory_args=([make_input_spec(self._env_spec.observations)],), # for the network factory
                            name='eval_inference_service',
                        ), device_config=eval_device_config)
                    program.add_service(
                        RPCService(
                            conn_config=config.distributed_backend_config,
                            instance_factory=self.evaluator,
                            instance_cls=acme.EnvironmentLoop,
                            args=(counter, logger, 
                                  variable_source, variable_staging, variable_service,
                                  eval_inference_client, actor_subconfig),
                        ), device_config=self._device_config.get_config(CONTROLLER))
                else:
                    program.add_service(
                        RPCService(
                            conn_config=config.distributed_backend_config,
                            instance_factory=self.evaluator,
                            instance_cls=acme.EnvironmentLoop,
                            args=(counter, logger,
                                  variable_source, variable_staging, variable_service,
                                  None, actor_subconfig),
                        ), device_config=eval_device_config)
        
        with program.group('actor'):
            for idx in range(self._num_actors):
                actor_device_config = self._device_config.get_config(ACTOR)
                if self._use_apv_mcts:
                    actor_subconfig = self._device_config.make_subconfig(self._search_num_actors)
                elif not self._use_apv_mcts or self._use_inference_server:
                    actor_subconfig = self._device_config.make_subconfig(0) # no new GPU actors will be created in the subprocess
                if self._use_inference_server:
                    # when there is an inference server, only one GPU process is used per actor.
                    # no need to make a new device config for the actor.
                    actor_inference_client = program.add_service(
                        AlphaDevInferenceService(
                            num_blocks=self._search_buffer_size, # 2x the number of processes.
                            network_factory=self._network_factory,
                            input_spec=self._env_spec.observations,
                            output_spec=self._env_spec.actions.num_values, # num actions
                            batch_size=self._search_batch_size,
                            variable_service=variable_service,
                            variable_update_period=self._variable_update_period,
                            factory_args=([make_input_spec(self._env_spec.observations)],), # for the network factory
                            name=f'actor_{idx}_inference',
                        ), device_config=actor_device_config
                    )
                    program.add_service(
                        RPCService(
                            conn_config=config.distributed_backend_config,
                            instance_factory=self.actor,
                            instance_cls=acme.EnvironmentLoop,
                            args=(idx, replay, counter, logger, None, actor_inference_client,
                                  actor_subconfig)
                        ), device_config=self._device_config.get_config(CONTROLLER))
                else:
                    # on the other hand, if we are using 'streamlined' APV_MCTS, we need to preallocate search_num_actors
                    # GPU slots for each actor.
                    program.add_service(
                        RPCService(
                            conn_config=config.distributed_backend_config,
                            instance_factory=self.actor,
                            instance_cls=acme.EnvironmentLoop,
                            args=(idx, replay, counter, logger, variable_service, None,
                                  actor_subconfig),
                        ), device_config=actor_device_config)
        
        return program
