from alphadev.network import AlphaDevNetwork, NetworkFactory, make_input_spec
from alphadev.config import AlphaDevConfig
from alphadev.acting import MCTSActor
from alphadev.environment import AssemblyGameModel
from alphadev.device_config import DeviceConfig
from alphadev.search.mcts import PUCTSearchPolicy
from acme.specs import EnvironmentSpec, make_environment_spec
from acme.tf.variable_utils import VariableClient
from acme.tf.utils import create_variables
from acme.environment_loop import EnvironmentLoop

from alphadev.service.variable_service import VariableService

import os

checkpoint_key = 'variable_76d0d767'
config_path = os.path.dirname(__file__) + '/apv_mcts_config.yaml'
config = AlphaDevConfig.from_yaml(config_path)

game = AssemblyGameModel(config.task_spec)
env_spec = make_environment_spec(game._environment)

network_factory = NetworkFactory(config=config)

variable_service = VariableService(config)
variable_service._variable_key = checkpoint_key

device_config = DeviceConfig(config.device_config_path)

actor = MCTSActor(
    device_config=device_config,
    environment_spec=env_spec,
    model=game,
    discount=config.discount,
    num_simulations=config.num_simulations,
    search_policy=PUCTSearchPolicy(),
    temperature_fn=config.temperature_fn,
    use_apv_mcts=False,
    network_factory=network_factory,
    variable_service=variable_service,
    variable_update_period=1,
    inference_service=None,
)

env_loop = EnvironmentLoop(
    game._environment,
    actor,
)

# Run the environment loop to start the actor.
env_loop.run(num_episodes=100)
