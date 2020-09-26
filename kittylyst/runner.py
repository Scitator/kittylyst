from typing import Any, Dict, List
from collections import defaultdict

from tqdm.auto import tqdm

from kittylyst.misc import set_random_seed


class IRunner:
    """An abstraction that knows **how** to run an experiment.

    IRunner contains all the logic of how to run the experiment,
    stages, epoch and batches.
    """

    def __init__(self):
        # experiment components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        # and callbacks
        self.callbacks: List = []

        # and the data
        self.loaders: Dict[str, Any] = None
        self.is_train_loader: bool = False
        # and the dataflow - model input, model output
        self.input = None
        self.output = None

        # metrics flow - batch, loader, epoch metrics
        self.batch_metrics: Dict = defaultdict(None)
        self.loader_metrics: Dict = defaultdict(lambda: [])

        # metrics & validation
        self.main_metric: str = None
        self.minimize_metric: bool = None

        # info
        self.num_epochs: int = 1
        self.epoch: int = 0
        self.loader_name: str = None
        self.verbose: bool = False

        self.experiment = None

    def _handle_batch(self, batch):
        raise NotImplementedError()

    def _prepare_for_stage(self, stage):
        set_random_seed()
        self.loaders = self.experiment.get_loaders(stage)
        self.model = self.experiment.get_model(stage)
        self.criterion = self.experiment.get_criterion(stage)
        self.optimizer = self.experiment.get_optimizer(stage, self.model)
        self.scheduler = self.experiment.get_scheduler(stage, self.optimizer)
        self.callbacks = self.experiment.get_callbacks(stage)
        for k, v in self.experiment.get_stage_params(stage).items():
            setattr(self, k, v)

    def _run_event(self, event: str) -> None:
        for callback in self.callbacks:
            getattr(callback, event)(self)

    def run_experiment(self, experiment):
        self.experiment = experiment

        for stage in self.experiment.stages:
            self._prepare_for_stage(stage)
            self._run_event("on_stage_start")

            for self.epoch in range(self.num_epochs):
                self._run_event("on_epoch_start")

                for self.loader_name, loader in self.loaders.items():
                    self.loader_metrics = defaultdict(lambda: [])
                    self.is_train_loader = self.loader_name.startswith("train")
                    self._run_event("on_loader_start")
                    loader = tqdm(loader) if self.verbose else loader

                    for batch in loader:
                        self._run_event("on_batch_start")
                        self._handle_batch(batch)
                        self._run_event("on_batch_end")

                    self._run_event("on_loader_end")
                self._run_event("on_epoch_end")
            self._run_event("on_stage_end")


class SupervisedRunner(IRunner):
    def _handle_batch(self, batch):
        features, targets = batch
        logits = list(map(self.model, features))
        self.input = {"features": features, "targets": targets}
        self.output = {"logits": logits}
