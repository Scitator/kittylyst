from typing import Any, Dict, List

from kittylyst.callback import ICallback
from kittylyst.engine import Engine, IEngine
from kittylyst.logger import ILogger
from kittylyst.trial import ITrial, Trial


class IExperiment:
    """An abstraction that knows **what** you would like to run.

    IExperiment contains information about the experiment –
    a model, a criterion, an optimizer, a scheduler, and their hyperparameters.
    It also contains information about the data and transformations used.
    """

    @property
    def seed(self) -> int:
        return 42

    @property
    def name(self) -> str:
        # @TODO: auto-generate name?
        return "experiment"

    @property
    def hparams(self) -> Dict:
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

    def get_engine(self) -> IEngine:
        # @TODO what to do with these three?
        return None

    def get_trial(self) -> ITrial:
        # @TODO what to do with these three?
        return None

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
        engine=None,
        trial=None,
        stage: str = "train",
        num_epochs: int = 1,
        hparams: Dict = None,
    ):
        self._loaders = loaders
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._callbacks = callbacks
        self._loggers = loggers
        self._engine = engine
        self._trial = trial

        self._stage = stage
        self._num_epochs = num_epochs
        self._hparams = hparams

    @property
    def name(self) -> str:
        return (
            "experiment"
            if self._trial is None
            else f"experiment_{self._trial.number}"
        )

    @property
    def hparams(self) -> Dict:
        if self._hparams is not None:
            return self._hparams
        if self._trial is not None:
            return self._trial.params
        return {}

    @property
    def stages(self) -> List[str]:
        return [self._stage]

    def get_stage_params(self, stage: str) -> Dict[str, Any]:
        return {
            "num_epochs": self._num_epochs,
            "migrate_from_previous_stage": False,
        }

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
        return self._callbacks or {}

    def get_loggers(self) -> Dict[str, ILogger]:
        return self._loggers or {}

    def get_engine(self) -> IEngine:
        return self._engine or Engine()

    def get_trial(self) -> ITrial:
        return self._trial
