from acme.utils import loggers
import wandb

class WandbLogger(loggers.Logger):
    """A logger that logs to Weights and Biases."""

    def __init__(self, config: dict):
        super().__init__()
        wandb.init(**config)

    def write(self, data: dict):
        wandb.log(data)

    def close(self):
        wandb.finish()
