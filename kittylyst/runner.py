from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache, partial

from kittylyst.callback import ICallback
from kittylyst.engine import IEngine
from kittylyst.experiment import IExperiment
from kittylyst.logger import ILogger
from kittylyst.misc import set_random_seed
from kittylyst.trial import ITrial


@lru_cache(maxsize=42)
def _has_str_intersections(origin_string: str, strings: Tuple):
    return any(x in origin_string for x in strings)


class IRunner(ICallback, ILogger, ABC):
    """An abstraction that knows **how** to run an experiment.

    IRunner contains all the logic of how to run the experiment,
    stages, epochs, loaders and batches.
    """

    def __init__(
        self, model=None, engine: IEngine = None,
    ):
        # the core
        self.model = model
        self.engine: IEngine = engine
        self.experiment: IExperiment = None
        self.trial: ITrial = None
        # the data
        self.loaders: Dict[str, Any] = None
        # the components
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        # the callbacks
        self.callbacks: Dict[str, ICallback] = {}
        # the loggers
        self.loggers: Dict[str, ILogger] = {}

        # the dataflow - model input, model output and other batch tensors
        self.batch = None

        # metrics flow - batch, loader and epoch metrics
        self.batch_metrics: Dict = defaultdict(None)
        self.loader_metrics: Dict = defaultdict(None)
        self.epoch_metrics: Dict = defaultdict(None)
        # self.stage_metrics: Dict = defaultdict(None)
        # self.experiment_metrics: Dict = defaultdict(None)

        # experiment info
        self.experiment_key: str = None
        self.global_epoch_step: int = 0
        self.global_batch_step: int = 0
        self.global_sample_step: int = 0

        # stage info
        self.stage_key: str = "infer"
        self.is_infer_stage: bool = self.stage_key.startswith("infer")
        self.stage_epoch_len: int = 0
        self.stage_epoch_step: int = 0
        self.stage_batch_step: int = 0
        self.stage_sample_step: int = 0

        # loader info
        self.loader = None
        self.loader_key: str = None
        self.is_train_loader: bool = False
        self.is_valid_loader: bool = False
        self.is_infer_loader: bool = True
        self.loader_batch_size: int = 0
        self.loader_batch_len: int = 0
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0

        # batch info
        self.batch_size: int = 0

        # extra
        self.exception: Exception = None
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True

    def log_metrics(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_metrics(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.experiment_key,
                global_sample_step=self.global_sample_step,
                global_batch_step=self.global_batch_step,
                global_epoch_step=self.global_epoch_step,
                # stage info
                stage_key=self.stage_key,
                stage_epoch_len=self.stage_epoch_len,
                stage_epoch_step=self.stage_epoch_step,
                stage_batch_step=self.stage_batch_step,
                stage_sample_step=self.stage_sample_step,
                # loader info
                loader_key=self.loader_key,
                loader_batch_len=self.loader_batch_len,
                loader_batch_step=self.loader_batch_step,
                loader_sample_step=self.loader_sample_step,
            )

    def log_image(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_image(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.experiment_key,
                global_sample_step=self.global_sample_step,
                global_batch_step=self.global_batch_step,
                global_epoch_step=self.global_epoch_step,
                # stage info
                stage_key=self.stage_key,
                stage_epoch_len=self.stage_epoch_len,
                stage_epoch_step=self.stage_epoch_step,
                stage_batch_step=self.stage_batch_step,
                stage_sample_step=self.stage_sample_step,
                # loader info
                loader_key=self.loader_key,
                loader_batch_len=self.loader_batch_len,
                loader_batch_step=self.loader_batch_step,
                loader_sample_step=self.loader_sample_step,
            )

    def log_hparams(self, *args, **kwargs) -> None:
        for logger in self.loggers.values():
            logger.log_hparams(
                *args,
                **kwargs,
                # experiment info
                experiment_key=self.experiment_key,
            )

    def flush_log(self) -> None:
        for logger in self.loggers.values():
            logger.flush_log()

    def close_log(self) -> None:
        for logger in self.loggers.values():
            logger.close_log()

    def on_experiment_start(self, runner: "IRunner"):
        assert self.experiment is not None
        self.experiment_key = self.experiment.name
        self.global_epoch_step: int = 0
        self.global_batch_step: int = 0
        self.global_sample_step: int = 0
        self.exception: Exception = None
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True

        self.trial = self.experiment.get_trial()
        self.engine = self.experiment.get_engine()
        self.loggers = self.experiment.get_loggers()
        self.log_hparams(hparams=self.experiment.hparams)
        # @TODO: should we report hparams to the trial?
        # self.experiment_metrics: Dict = defaultdict(None)

    def on_stage_start(self, runner: "IRunner"):
        assert self.stage_key is not None
        stage_params = self.experiment.get_stage_params(self.stage_key)
        self.is_infer_stage: bool = self.stage_key.startswith("infer")
        self.stage_epoch_len = stage_params["num_epochs"]
        self.stage_epoch_step: int = 0
        self.stage_batch_step: int = 0
        self.stage_sample_step: int = 0
        # self.stage_metrics: Dict = defaultdict(None)

    def on_epoch_start(self, runner: "IRunner"):
        self.global_epoch_step += 1
        self.stage_epoch_step += 1
        self.epoch_metrics: Dict = defaultdict(None)

    def on_loader_start(self, runner: "IRunner"):
        assert self.loader is not None
        self.is_train_loader: bool = self.loader_key.startswith("train")
        self.is_valid_loader: bool = self.loader_key.startswith("valid")
        self.is_infer_loader: bool = self.loader_key.startswith("infer")
        self.loader_batch_size: int = 0
        self.loader_batch_len: int = 0
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_metrics: Dict = defaultdict(None)

    def on_batch_start(self, runner: "IRunner"):
        self.batch = self.engine.sync_device(tensor_or_module=self.batch)
        self.batch_size = len(self.batch[0])
        self.global_batch_step += 1
        self.stage_batch_step += 1
        self.loader_batch_step += 1
        self.global_sample_step += self.batch_size
        self.stage_sample_step += self.batch_size
        self.loader_sample_step += self.batch_size
        self.batch_metrics: Dict = defaultdict(None)

    def on_batch_end(self, runner: "IRunner"):
        # @TODO: do we need to log metrics here?
        self.log_metrics(metrics=self.batch_metrics, scope="batch")

    def on_loader_end(self, runner: "IRunner"):
        # @TODO: do we need to log metrics here?
        self.log_metrics(metrics=self.loader_metrics, scope="loader")
        self.epoch_metrics[self.loader_key] = self.loader_metrics.copy()

    def on_epoch_end(self, runner: "IRunner"):
        # @TODO: do we need to log metrics here?
        self.log_metrics(metrics=self.epoch_metrics, scope="epoch")
        # self.stage_metrics[self.stage_epoch_step] = self.epoch_metrics.copy()
        self.flush_log()

    def on_stage_end(self, runner: "IRunner"):
        # self.log_metrics(metrics=self.stage_metrics, scope="stage")
        # self.experiment_metrics[self.stage_key] = self.stage_metrics.copy()
        self.engine.deinit_components()

    def on_experiment_end(self, runner: "IRunner"):
        # @TODO: should we report results to the trial?
        # self.log_metrics(metrics=self.experiment_metrics, scope="experiment")
        self.close_log()

    def on_exception(self, runner: "IRunner"):
        raise self.exception

    def _run_event(self, event: str) -> None:
        if _has_str_intersections(event, ("_start",)):
            getattr(self, event)(self)
        for callback in self.callbacks.values():
            getattr(callback, event)(self)
        if _has_str_intersections(event, ("_end", "_exception")):
            getattr(self, event)(self)

    @abstractmethod
    def _handle_batch(self, batch):
        pass

    def _run_batch(self) -> None:
        self._run_event("on_batch_start")
        self._handle_batch(batch=self.batch)
        self._run_event("on_batch_end")

    def _run_loader(self) -> None:
        self._run_event("on_loader_start")
        for self.loader_batch_step, self.batch in enumerate(self.loader):
            self._run_batch()
            if self.need_early_stop:
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
        while self.stage_epoch_step < self.stage_epoch_len:
            self._run_epoch()
            if self.need_early_stop:
                self.need_early_stop = False
                break
        self._run_event("on_stage_end")

    def _run_experiment(self) -> None:
        self._run_event("on_experiment_start")
        for self.stage_key in self.experiment.stages:
            if self.engine.rank < 0:
                # single-device branch
                self._run_stage()
            else:
                # ddp-device branch
                # mp.spawn(self._run_stage, num_process=self.engine.world_size)
                raise NotImplementedError()
        self._run_event("on_experiment_end")

    def run(self, experiment: IExperiment = None) -> "IRunner":
        self.experiment = experiment
        try:
            self._run_experiment()
        except (Exception, KeyboardInterrupt) as ex:
            self.exception = ex
            self._run_event("on_exception")
        return self


class IStageBasedRunner(IRunner):
    def on_stage_start(self, runner: "IRunner"):
        super().on_stage_start(runner)

        stage_params = self.experiment.get_stage_params(self.stage_key)
        migrate_model_from_previous_stage = stage_params.get(
            "migrate_model_from_previous_stage", True
        )
        # some custom logic is possible here
        if self.model is not None and migrate_model_from_previous_stage:
            model_fn = lambda: self.model
        else:
            model_fn = partial(self.experiment.get_model, stage=self.stage_key)
        set_random_seed(self.experiment.seed + self.global_epoch_step)
        self.loaders = self.experiment.get_data(self.stage_key)

        # @TODO: we need a better approach here
        (
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
        ) = self.engine.init_components(
            model_fn=model_fn,
            criterion_fn=partial(
                self.experiment.get_criterion, stage=self.stage_key
            ),
            optimizer_fn=partial(
                self.experiment.get_optimizer, stage=self.stage_key
            ),
            scheduler_fn=partial(
                self.experiment.get_scheduler, stage=self.stage_key
            ),
        )

        self.callbacks = self.experiment.get_callbacks(self.stage_key)

    def on_epoch_start(self, runner: "IRunner"):
        super().on_epoch_start(runner)
        assert self.loaders is not None
        for loader_key, loader in self.loaders.items():
            if len(loader) == 0:
                raise NotImplementedError(
                    f"DataLoader with name {loader_key} is empty."
                )

    def on_loader_start(self, runner: "IRunner"):
        super().on_loader_start(runner)
        self.loader_batch_len = len(self.loader)
        if self.loader_batch_len == 0:
            raise NotImplementedError(
                f"DataLoader with name {self.loader_key} is empty."
            )


class SupervisedRunner(IStageBasedRunner):
    def _handle_batch(self, batch):
        features, targets = batch
        logits = list(map(self.model, features))
        self.batch = {
            "features": features,
            "targets": targets,
            "logits": logits,
        }
