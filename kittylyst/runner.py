from typing import Any, Dict, Tuple
from collections import defaultdict
from functools import lru_cache, partial

from kittylyst.callback import ICallback
from kittylyst.engine import IEngine
from kittylyst.experiment import IExperiment
from kittylyst.logger import ILogger
from kittylyst.misc import set_random_seed
from kittylyst.trial import ITrial


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
        model=None,
        engine: IEngine = None,
        # experiment: IExperiment = None,
        # trial: ITrial = None,
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

        # the dataflow - model input, model output
        # @TODO: could we make just self.batch_tensors ?
        # to store input and output?
        # self.input = None
        # self.output = None
        self.batch = None

        # @TODO: do we need to store metrics under Runner?
        # metrics flow - batch, loader and epoch metrics
        self.batch_metrics: Dict = defaultdict(None)
        self.loader_metrics: Dict = defaultdict(lambda: [])
        self.epoch_metrics: Dict = defaultdict(None)
        # self.stage_metrics: Dict = defaultdict(None)
        # self.experiment_metrics: Dict = defaultdict(None)

        # experiment info
        self.experiment_key: str = None
        self.global_sample_step: int = 0
        self.global_batch_step: int = 0
        self.global_epoch_step: int = 0

        # stage info
        self.stage_key: str = "infer"
        self.stage_epoch_len: int = 0
        self.stage_epoch_step: int = 0
        # @TODO: stage batch, sample step? do we need them?
        self.is_infer_stage: bool = self.stage_key.startswith("infer")
        # loader info
        self.loader = None
        self.loader_key: str = None
        self.loader_batch_len: int = 0
        self.loader_batch_step: int = 0
        self.loader_sample_step: int = 0
        self.loader_batch_size = 0
        self.is_train_loader: bool = False
        self.is_valid_loader: bool = False
        self.is_infer_loader: bool = True
        # batch info
        self.batch_size: int = 0

        # extra
        self.exception: Exception = None
        self.need_early_stop: bool = False
        # self.need_exception_reraise: bool = True

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
                stage_len=self.stage_epoch_len,
                stage_epoch_step=self.stage_epoch_step,
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

    def flush(self) -> None:
        for logger in self.loggers.values():
            logger.flush()

    def close(self) -> None:
        for logger in self.loggers.values():
            logger.close()

    def on_experiment_start(self, runner: "IRunner"):
        assert self.experiment is not None
        self.experiment_key = self.experiment.name
        self.trial = self.experiment.get_trial()
        self.engine = self.experiment.get_engine()
        self.loggers = self.experiment.get_loggers()
        self.log_hparams(hparams=self.experiment.hparams)
        # @TODO: should we report hparams to the trial?

    def on_stage_start(self, runner: "IRunner"):
        assert self.stage_key is not None
        stage_params = self.experiment.get_stage_params(self.stage_key)
        # @TODO: think about naming here
        self.stage_epoch_len = stage_params["num_epochs"]
        self.stage_epoch_step: int = 0
        self.is_infer_stage: bool = self.stage_key.startswith("infer")
        # migrate_from_previous_stage = stage_params.get(...)
        # some custom logic is possible here

        set_random_seed(self.experiment.seed + self.global_epoch_step)
        self.loaders = self.experiment.get_data(self.stage_key)

        # @TODO: we need a better approach here
        (
            self.model,
            self.criterion,
            self.optimizer,
            self.scheduler,
        ) = self.engine.init_components(
            model_fn=partial(self.experiment.get_model, stage=self.stage_key),
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
        assert self.loaders is not None
        for loader_key, loader in self.loaders.items():
            if len(loader) == 0:
                raise NotImplementedError(
                    f"DataLoader with name {loader_key} is empty."
                )
        self.global_epoch_step += 1
        self.stage_epoch_step += 1
        self.epoch_metrics: Dict = defaultdict(None)

    def on_loader_start(self, runner: "IRunner"):
        assert self.loader is not None
        self.loader_batch_len = len(self.loader)
        if self.loader_batch_len == 0:
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
        self.batch_size = len(self.batch[0])
        self.global_batch_step += 1
        self.loader_batch_step += 1
        self.global_sample_step += self.batch_size
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
        self.flush()

    def on_stage_end(self, runner: "IRunner"):
        # self.log_metrics(metrics=self.stage_metrics, scope="stage")
        self.engine.deinit_components()

    def on_experiment_end(self, runner: "IRunner"):
        # @TODO: should we report results to the trial?
        # self.log_metrics(metrics=self.experiment_metrics, scope="experiment")
        self.close()

    def on_exception(self, runner: "IRunner"):
        raise self.exception

    def _run_event(self, event: str) -> None:
        if _is_substring(event, ("start",)):
            getattr(self, event)(self)
        for callback in self.callbacks.values():
            getattr(callback, event)(self)
        if _is_substring(event, ("end", "exception")):
            getattr(self, event)(self)

    def _handle_batch(self, batch):
        raise NotImplementedError()

    def _run_batch(self) -> None:
        self.batch = self.engine.sync_device(tensor_or_module=self.batch)
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


class SupervisedRunner(IRunner):
    def _handle_batch(self, batch):
        features, targets = batch
        logits = list(map(self.model, features))
        # self.input = {"features": features, "targets": targets}
        # self.output = {"logits": logits}
        self.batch = {"features": features, "targets": targets, "logits": logits}
