from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import sys

import optuna
from tqdm.auto import tqdm

from kittylyst.metric import AverageMetric, IMetric
from kittylyst.misc import unvalue

if TYPE_CHECKING:
    from kittylyst.runner import IRunner


class ICallback:
    """An abstraction that lets you customize your experiment run logic."""

    def on_experiment_start(self, runner: "IRunner") -> None:
        pass

    def on_stage_start(self, runner: "IRunner") -> None:
        pass

    def on_epoch_start(self, runner: "IRunner") -> None:
        pass

    def on_loader_start(self, runner: "IRunner") -> None:
        pass

    def on_batch_start(self, runner: "IRunner") -> None:
        pass

    def on_batch_end(self, runner: "IRunner") -> None:
        pass

    def on_loader_end(self, runner: "IRunner") -> None:
        pass

    def on_epoch_end(self, runner: "IRunner") -> None:
        pass

    def on_stage_end(self, runner: "IRunner") -> None:
        pass

    def on_experiment_end(self, runner: "IRunner") -> None:
        pass

    def on_exception(self, runner: "IRunner") -> None:
        pass


class MetricCallback(ICallback):
    def __init__(
        self,
        metric: IMetric,
        input_key: str,
        output_key: str,
        compute_on_batch: bool = True,
    ):
        self.metric = metric
        self.input_key = input_key
        self.output_key = output_key
        self.compute_on_batch = compute_on_batch

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        # @TODO: do we need to sync tensors here?
        inputs = runner.batch[self.output_key]
        targets = runner.batch[self.input_key]
        inputs = runner.engine.sync_tensor(inputs)
        targets = runner.engine.sync_tensor(targets)

        self.metric.update(inputs, targets)
        if self.compute_on_batch:
            runner.batch_metrics.update(self.metric.compute_key_value())

    def on_loader_end(self, runner: "IRunner") -> None:
        runner.loader_metrics.update(self.metric.compute_key_value())


class CriterionCallback(ICallback):
    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self.average_metric = AverageMetric()

    def on_loader_start(self, runner: "IRunner") -> None:
        self.average_metric.reset()

    def on_batch_end(self, runner: "IRunner"):
        # @TODO: do we need to sync tensors here?
        logits, targets = runner.batch["logits"], runner.batch["targets"]
        logits = runner.engine.sync_tensor(logits)
        targets = runner.engine.sync_tensor(targets)

        loss = runner.criterion(logits, targets)
        l2_loss = self.alpha * sum((p * p for p in runner.model.parameters()))
        loss = loss + l2_loss
        runner.batch_metrics.update({"loss": loss})
        self.average_metric.update(unvalue(loss), len(targets))

    def on_loader_end(self, runner: "IRunner") -> None:
        loss_mean, loss_std = self.average_metric.compute()
        runner.loader_metrics.update(
            {"loss_mean": loss_mean, "loss_std": loss_std}
        )


class OptimizerCallback(ICallback):
    def __init__(self, metric_key: str = "loss"):
        self.metric_key = metric_key

    def on_batch_end(self, runner: "IRunner"):
        if runner.is_train_loader:
            runner.engine.zero_grad(runner.model, runner.optimizer)
            runner.batch_metrics[self.metric_key].backward()
            runner.engine.optimizer_step(runner.model, runner.optimizer)
            runner.batch_metrics.update({"lr": runner.optimizer.lr})

    def on_loader_end(self, runner: "IRunner") -> None:
        runner.loader_metrics.update({"lr": runner.optimizer.lr})


class SchedulerCallback(ICallback):
    def on_epoch_end(self, runner: "IRunner"):
        runner.scheduler.step(runner.stage_epoch_step)


# Should it be ICallback or *ILogger*?
# class VerboseCallback(ICallback):
#     def on_loader_start(self, runner: "IRunner") -> None:
#         runner.loader = tqdm(runner.loader)


class VerboseCallback(ICallback):
    def __init__(self):
        super().__init__()
        self.tqdm: tqdm = None

    def on_loader_start(self, runner: "IRunner"):
        self.tqdm = tqdm(
            total=runner.loader_batch_len,
            desc=f"{runner.stage_epoch_step}/{runner.stage_epoch_len}"
            f" * Epoch ({runner.loader_key})",
            # leave=True,
            # ncols=0,
            # file=sys.stdout,
        )

    def on_batch_end(self, runner: "IRunner"):
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(unvalue(v))
                if unvalue(v) > 1e-3
                else "{:1.3e}".format(unvalue(v))
                for k, v in sorted(runner.batch_metrics.items())
            }
        )
        self.tqdm.update()

    def on_loader_end(self, runner: "IRunner"):
        self.tqdm.clear()
        self.tqdm.close()
        self.tqdm = None


class IMetricHandlerCallback(ABC, ICallback):
    def __init__(
        self,
        loader_key: str,
        metric_key: str,
        minimize: bool = True,
        min_delta: float = 1e-6,
    ):
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.minimize = minimize
        self.best_score = None

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)

    @abstractmethod
    def handle(self, runner: "IRunner"):
        pass

    def on_epoch_end(self, runner: "IRunner") -> None:
        score = runner.epoch_metrics[self.loader_key][self.metric_key]
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            self.handle(runner=runner)


class TopNMetricHandlerCallback(IMetricHandlerCallback):
    def __init__(
        self,
        loader_key: str,
        metric_key: str,
        minimize: bool = True,
        min_delta: float = 1e-6,
        save_n_best: int = 1,
    ):
        super().__init__(
            loader_key=loader_key,
            metric_key=metric_key,
            minimize=minimize,
            min_delta=min_delta,
        )
        self.save_n_best = save_n_best
        self.top_best_metrics = []

    def handle(self, runner: "IRunner"):
        self.top_best_metrics.append(
            (self.best_score, runner.stage_epoch_step,)
        )

        self.top_best_metrics = sorted(
            self.top_best_metrics,
            key=lambda x: x[0],
            reverse=not self.minimize,
        )
        if len(self.top_best_metrics) > self.save_n_best:
            self.top_best_metrics.pop(-1)

    def on_stage_end(self, runner: "IRunner") -> None:
        log_message = "Top-N best epochs:\n"
        log_message += "\n".join(
            [
                "{epoch}\t{metric:3.4f}".format(epoch=epoch, metric=metric)
                for metric, epoch in self.top_best_metrics
            ]
        )
        print(log_message)


class CheckpointCallback(TopNMetricHandlerCallback):
    def handle(self, runner: "IRunner"):
        # @TODO: here is very simplified logic
        super().handle(runner=runner)
        checkpoint = runner.engine.pack_checkpoint(
            model=runner.model,
            criterion=runner.criterion,
            optimizer=runner.optimizer,
            scheduler=runner.scheduler,
        )
        runner.engine.save_checkpoint(checkpoint, "./logpath.pth")

    def on_stage_end(self, runner: "IRunner") -> None:
        # @TODO: here is very simplified logic
        super().on_stage_end(runner=runner)
        checkpoint = runner.engine.load_checkpoint("./logpath.pth")
        runner.engine.unpack_checkpoint(
            checkpoint=checkpoint,
            model=runner.model,
            criterion=runner.criterion,
            optimizer=runner.optimizer,
            scheduler=runner.scheduler,
        )


# Should it be ICallback, *ILogger* or ITrial?
class OptunaPruningCallback(ICallback):
    def __init__(
        self, loader_key: str, metric_key: str, trial: optuna.Trial = None
    ):
        super().__init__()
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.trial = trial

    def on_stage_start(self, runner: "IRunner"):
        trial = runner.trial
        if (
            self.trial is None
            and trial is not None
            and isinstance(trial, optuna.Trial)
        ):
            self.trial = trial

        if self.trial is None:
            raise NotImplementedError("No Optuna trial found for logging")

    def on_epoch_end(self, runner: "IRunner"):
        metric_value = runner.epoch_metrics[self.loader_key][self.metric_key]
        self.trial.report(metric_value, step=runner.stage_epoch_step)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(
                runner.stage_epoch_step
            )
            raise optuna.TrialPruned(message)
