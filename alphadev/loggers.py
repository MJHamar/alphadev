from typing import Callable
from acme.utils import loggers
import wandb
import pickle

class WandbLogger(loggers.Logger):
    """A logger that logs to Weights and Biases."""

    def __init__(self, config: dict):
        super().__init__()
        wandb.init(**config)

    def write(self, data: dict):
        wandb.log(data)

    def close(self):
        wandb.finish()

class LoggerService(object):
    """
    An asynchronous logger service that processes log messages in a separate thread.
    """
    def __init__(self, logger_factory: Callable[[], loggers.Logger]):
        """
        Initializes the logger, creating a queue for log messages.
        """
        self._logger = logger_factory()

    # NOTE: use log() as the writing method to avoid a limitation in launchpad.
    def log(self, message):
        """
        Public method to enqueue a log message. This method can be called from any thread.
        """
        # Unpickle the message
        message = pickle.loads(message)
        # Call the logger's write method
        self._logger.write(message)

    def close(self):
        """
        Stops the asynchronous logging service.
        """
        self._logger.close()

class LoggerServiceWrapper(object):
    def __init__(self, logger):
        self._logger_service = logger
    def write(self, message):
        """
        Public method to enqueue a log message. This method can be called from any thread.
        """
        msg_bin = pickle.dumps(message)
        self._logger_service.log(msg_bin)