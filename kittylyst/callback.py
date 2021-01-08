from typing import Dict, TYPE_CHECKING

import numpy as np

from kittylyst.misc import format_metrics, unvalue

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


class CriterionCallback(ICallback):
    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha

    def on_batch_end(self, runner: "IRunner"):
        logits, targets = runner.output["logits"], runner.input["targets"]
        loss = runner.criterion(logits, targets)
        l2_loss = self.alpha * sum((p * p for p in runner.model.parameters()))
        loss = loss + l2_loss
        runner.batch_metrics.update({"loss": loss})


class AccuracyCallback(ICallback):
    def on_batch_end(self, runner: "IRunner"):
        logits, targets = runner.output["logits"], runner.input["targets"]
        accuracy = [
            (yi > 0) == (li.data > 0) for yi, li in zip(targets, logits)
        ]
        accuracy = sum(accuracy) / len(accuracy)
        runner.batch_metrics.update({"accuracy": accuracy})


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


class LoggerCallback(ICallback):
    def on_batch_end(self, runner: "IRunner"):
        for k, v in runner.batch_metrics.items():
            runner.loader_metrics[k].append(unvalue(v))

    def on_loader_end(self, runner: "IRunner"):
        metrics = {k: np.mean(v) for k, v in runner.loader_metrics.items()}
        msg = (
            f"{runner.stage_epoch + 1}/{runner.stage_len}"
            + f" Epoch ({runner.loader_key}) "
            + format_metrics(metrics)
        )
        print(msg)
