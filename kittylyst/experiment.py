from typing import Any, Dict, List

from kittylyst.callback import ICallback
from kittylyst.logger import ILogger


class IExperiment:
    """An abstraction that knows **what** you would like to run.

    IExperiment contains information about the experiment –
    a model, a criterion, an optimizer, a scheduler, and their hyperparameters.
    It also contains information about the data and transformations used.
    """

    @property
    def seed(self) -> 42:
        return 42

    @property
    def hparams(self) -> Dict:
        return {}

    @property
    def engine_params(self) -> Dict:
        # @TODO what to do with these three?
        return {}

    @property
    def trial_params(self) -> Dict:
        # @TODO what to do with these three?
        return {}

    @property
    def stages(self) -> List[str]:
        return []

    def get_stage_params(self, stage: str) -> Dict[str, Any]:
        return {}

    def get_data(self, stage: str) -> Dict[str, Any]:
        pass

    def get_model(self, stage: str):
        pass

    def get_criterion(self, stage: str):
        pass

    def get_optimizer(self, stage: str, model):
        pass

    def get_scheduler(self, stage: str, optimizer):
        pass

    def get_callbacks(self, stage: str) -> Dict[str, ICallback]:
        return {}

    def get_loggers(self) -> Dict[str, ILogger]:
        # @TODO what to do with these three?
        return {}


class SingleStageExperiment(IExperiment):
    def __init__(
        self,
        model,
        loaders: Dict,
        callbacks: Dict = None,
        loggers: Dict = None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        stage: str = "train",
        num_epochs: int = 1,
    ):
        self._loaders = loaders
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._callbacks = callbacks or {}
        self._loggers = loggers or {}

        self._stage = stage
        self._num_epochs = num_epochs

    @property
    def stages(self) -> List[str]:
        return [self._stage]

    def get_stage_params(self, stage: str) -> Dict[str, Any]:
        return {"num_epochs": self._num_epochs}

    def get_data(self, stage: str) -> Dict[str, Any]:
        return self._loaders

    def get_model(self, stage: str):
        return self._model

    def get_criterion(self, stage: str):
        return self._criterion

    def get_optimizer(self, stage: str, model):
        return self._optimizer

    def get_scheduler(self, stage: str, optimizer):
        return self._scheduler

    def get_callbacks(self, stage: str) -> Dict[str, ICallback]:
        return self._callbacks

    def get_loggers(self) -> Dict[str, ILogger]:
        return self._loggers
