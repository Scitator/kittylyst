from typing import Dict


class IExperiment:
    """An abstraction that knows **what** you would like to run.

    IExperiment contains information about the experiment –
    a model, a criterion, an optimizer, a scheduler, and their hyperparameters.
    It also contains information about the data and transformations used.
    """

    @property
    def stages(self):
        return []

    def get_stage_params(self, stage: str):
        return {}

    def get_loaders(self, stage: str):
        pass

    def get_model(self, stage: str):
        pass

    def get_criterion(self, stage: str):
        pass

    def get_optimizer(self, stage: str, model):
        pass

    def get_scheduler(self, stage: str, optimizer):
        pass

    def get_callbacks(self, stage: str):
        pass


class Experiment(IExperiment):
    def __init__(
        self,
        model,
        loaders: Dict,
        callbacks: Ellipsis = None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        stage: str = "train",
        num_epochs: int = 1,
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
    ):
        self._loaders = loaders
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._callbacks = callbacks or []

        self._stage = stage
        self.num_epochs = num_epochs
        self.main_metric = main_metric
        self.minimize_metric = minimize_metric
        self.verbose = verbose

    @property
    def stages(self):
        return [self._stage]

    def get_stage_params(self, stage: str):
        return {
            "num_epochs": self.num_epochs,
            "main_metric": self.main_metric,
            "minimize_metric": self.minimize_metric,
            "verbose": self.verbose,
        }

    def get_loaders(self, stage: str):
        return self._loaders

    def get_model(self, stage: str):
        return self._model

    def get_criterion(self, stage: str):
        return self._criterion

    def get_optimizer(self, stage: str, model):
        return self._optimizer

    def get_scheduler(self, stage: str, optimizer):
        return self._scheduler

    def get_callbacks(self, stage: str):
        return self._callbacks
