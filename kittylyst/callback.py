from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

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
        # @TODO: here should be some engine stuff with tensor sync?
        inputs = runner.output[self.output_key]
        targets = runner.input[self.input_key]
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
        logits, targets = runner.output["logits"], runner.input["targets"]
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
    def __init__(self, metric_key: str = "loss", alpha: float = 1e-4):
        self.metric_key = metric_key
        self.alpha = alpha

    def on_batch_end(self, runner: "IRunner"):
        if runner.is_train_loader:
            runner.model.zero_grad()
            runner.batch_metrics[self.metric_key].backward()
            runner.optimizer.step()


class SchedulerCallback(ICallback):
    def on_epoch_end(self, runner: "IRunner"):
        runner.scheduler.step(runner.stage_epoch)


class VerboseCallback(ICallback):
    def on_loader_start(self, runner: "IRunner") -> None:
        runner.loader = tqdm(runner.loader)


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
        self.top_best_metrics.append((self.best_score, runner.stage_epoch,))

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
