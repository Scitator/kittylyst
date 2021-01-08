from typing import Any, Dict, Tuple
from collections import defaultdict
from functools import lru_cache, partial

from kittylyst.callback import ICallback
from kittylyst.engine import Engine, IEngine
from kittylyst.experiment import IExperiment
from kittylyst.logger import ILogger
from kittylyst.misc import set_random_seed


@lru_cache(maxsize=42)
def _is_substring(origin_string: str, strings: Tuple):
    return any(x in origin_string for x in strings)


class IRunner(ICallback, ILogger):
    """An abstraction that knows **how** to run an experiment.

    IRunner contains all the logic of how to run the experiment,
    stages, epoch and batches.
    """

    def __init__(
        self,
        engine: IEngine = None,
        model=None,
        experiment: IExperiment = None,
    ):
        # main
        self.engine: IEngine = engine
        self.model = model
        self.experiment: IExperiment = experiment
        # the data
        self.loaders: Dict[str, Any] = None
        # components
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        # callbacks
        self.callbacks: Dict[str, ICallback] = {}
        # loggers
        self.loggers: Dict[str, ILogger] = {}

        # the dataflow - model input, model output
        # @TODO: could we make just self.batch_tensors ?
        # to store input and output?
        self.input = None
        self.output = None

        # @TODO: do we need to store metrics under Runner?
        # metrics flow - batch, loader and epoch metrics
        self.batch_metrics: Dict = defaultdict(None)
        self.loader_metrics: Dict = defaultdict(lambda: [])
        self.epoch_metrics: Dict = defaultdict(None)
        # self.stage_metrics: Dict = defaultdict(None)
        # self.experiment_metrics: Dict = defaultdict(None)

        # experiment info
        self.global_sample_step: int = 0
        self.global_batch_step: int = 0
        self.global_epoch: int = 0
        self.need_early_stop: bool = False
        # self.need_exception_reraise: bool = True

        # stage info
        self.stage: str = "infer"
        self.stage_len: int = 0
        self.is_infer_stage: bool = self.stage.startswith("infer")
        # epoch info
        self.stage_epoch: int = 0
        # loader info
        self.loader = None
        self.loader_sample_step: int = 0
        self.loader_batch_step: int = 0
        self.loader_key: str = None
        self.loader_len: int = 0
        self.loader_batch_size = 0
        self.is_train_loader: bool = False
        self.is_valid_loader: bool = False
        self.is_infer_loader: bool = True
        # batch info
        self.batch_size: int = 0

        # extra
        self.exception: Exception = None

    def log_metrics(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_metrics(*args, **kwargs)

    def log_image(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_image(*args, **kwargs)

    def log_hparams(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_hparams(*args, **kwargs)

    def on_experiment_start(self, runner: "IRunner"):
        assert self.experiment is not None
        self.loggers = self.experiment.get_loggers()
        self.log_hparams(hparams=self.experiment.hparams)

    def on_stage_start(self, runner: "IRunner"):
        assert self.stage is not None
        stage_params = self.experiment.get_stage_params(self.stage)
        # @TODO: think about naming here
        self.stage_len = stage_params["num_epochs"]
        # migrate_from_previous_stage = stage_params.get(...)
        # some custom logic is possible here
        set_random_seed(self.experiment.seed + self.global_epoch)
        self.loaders = self.experiment.get_data(self.stage)

        # @TODO: we need better approach here
        (
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
        ) = self.engine.init_components(
            model_fn=partial(self.experiment.get_model, stage=self.stage),
            criterion_fn=partial(
                self.experiment.get_criterion, stage=self.stage
            ),
            optimizer_fn=partial(
                self.experiment.get_optimizer, stage=self.stage
            ),
            scheduler_fn=partial(
                self.experiment.get_scheduler, stage=self.stage
            ),
        )

        self.callbacks = self.experiment.get_callbacks(self.stage)

    def on_epoch_start(self, runner: "IRunner"):
        assert self.loaders is not None
        for loader_key, loader in self.loaders.items():
            if len(loader) == 0:
                raise NotImplementedError(
                    f"DataLoader with name {loader_key} is empty."
                )
        self.global_epoch += 1
        self.stage_epoch += 1
        self.epoch_metrics: Dict = defaultdict(None)

    def on_loader_start(self, runner: "IRunner"):
        assert self.loader is not None
        self.loader_len = len(self.loader)
        if self.loader_len == 0:
            raise NotImplementedError(
                f"DataLoader with name {self.loader_key} is empty."
            )
        self.loader_sample_step = 0
        self.loader_batch_step = 0
        self.is_train_loader = self.loader_key.startswith("train")
        self.is_valid_loader = self.loader_key.startswith("valid")
        self.is_infer_loader = self.loader_key.startswith("infer")
        self.loader_metrics: Dict = defaultdict(lambda: [])

    def on_batch_start(self, runner: "IRunner"):
        self.batch_size = len(self.input[0])
        self.global_batch_step += 1
        self.loader_batch_step += 1
        self.global_sample_step += self.batch_size
        self.loader_sample_step += self.batch_size
        self.batch_metrics: Dict = defaultdict(None)

    def on_batch_end(self, runner: "IRunner"):
        # @TODO: do we need to log metrics here?
        self.log_metrics(
            metrics=self.batch_metrics,
            step=self.loader_batch_step,
            step_limit=self.loader_len,
            scope="batch",
        )
        self.log_metrics(
            metrics=self.batch_metrics,
            step=self.global_batch_step,
            scope="global_batch",
        )

    def on_loader_end(self, runner: "IRunner"):
        # @TODO: do we need to log metrics here?
        self.log_metrics(
            metrics=self.loader_metrics,
            step=self.stage_epoch,
            step_limit=self.stage_len,
            scope="loader",
            scope_name=self.loader_key,
        )
        self.epoch_metrics[self.loader_key] = self.loader_metrics.copy()

    def on_epoch_end(self, runner: "IRunner"):
        pass
        # @TODO: do we need to log metrics here?
        # self.log_metrics(
        #     metrics=self.epoch_metrics,
        #     step=self.stage_epoch,
        #     step_limit=self.stage_len,
        #     scope="epoch",
        # )
        # self.log_metrics(
        #     metrics=self.epoch_metrics,
        #     step=self.global_epoch,
        #     scope="global_epoch",
        # )

    def on_stage_end(self, runner: "IRunner"):
        pass

    def on_experiment_end(self, runner: "IRunner"):
        pass

    def on_exception(self, runner: "IRunner"):
        raise self.exception

    def _run_event(self, event: str) -> None:
        if _is_substring(event, ("start", "exception")):
            getattr(self, event)(self)
        for callback in self.callbacks.values():
            getattr(callback, event)(self)
        if _is_substring(event, ("end",)):
            getattr(self, event)(self)

    def _handle_batch(self, batch):
        raise NotImplementedError()

    def _run_batch(self) -> None:
        # self.input = self._handle_device(batch=self.input)
        self._run_event("on_batch_start")
        self._handle_batch(batch=self.input)
        self._run_event("on_batch_end")

    def _run_loader(self) -> None:
        self._run_event("on_loader_start")
        for self.loader_batch_step, self.input in enumerate(self.loader):
            self._run_batch()
            if self.need_early_stop:
                # @TODO: do we need extra event for early stop?
                self.need_early_stop = False
                break
        self._run_event("on_loader_end")

    def _run_epoch(self) -> None:
        self._run_event("on_epoch_start")
        for self.loader_key, self.loader in self.loaders.items():
            self._run_loader()
        self._run_event("on_epoch_end")

    def _run_stage(self) -> None:
        self._run_event("on_stage_start")
        while self.stage_epoch < self.stage_len:
            self._run_epoch()
            if self.need_early_stop:
                # @TODO: do we need extra event for early stop?
                self.need_early_stop = False
                break
        self._run_event("on_stage_end")

    def _run_experiment(self) -> None:
        self._run_event("on_experiment_start")
        for self.stage in self.experiment.stages:
            self._run_stage()
        self._run_event("on_experiment_end")

    def run_experiment(self, experiment: IExperiment = None) -> "IRunner":
        # @TODO: where should we init it?
        self.engine = Engine()
        self.experiment = experiment or self.experiment
        try:
            self._run_experiment()
        except (Exception, KeyboardInterrupt) as ex:
            self.exception = ex
            self._run_event("on_exception")
        return self


class SupervisedRunner(IRunner):
    def _handle_batch(self, batch):
        features, targets = batch
        logits = list(map(self.model, features))
        self.input = {"features": features, "targets": targets}
        self.output = {"logits": logits}
