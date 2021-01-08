from typing import TYPE_CHECKING

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
    def __init__(self, metric: IMetric, input_key: str, output_key: str):
        self.metric = metric
        self.input_key = input_key
        self.output_key = output_key

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        # @TODO: here should be some engine stuff with tensor sync?
        inputs = runner.output[self.output_key]
        targets = runner.input[self.input_key]
        self.metric.update(inputs, targets)
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
