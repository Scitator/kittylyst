from typing import Any, Dict
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import roc_auc_score

from kittylyst.misc import unvalue


class IMetric(ABC):
    """Interface for all Metrics."""

    def __init__(self, compute_on_call: bool = True):
        self.compute_on_call = compute_on_call

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def compute(self) -> Any:
        pass

    @abstractmethod
    def compute_key_value(self) -> Dict[str, float]:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        self.update(*args, **kwargs)
        if self.compute_on_call:
            return self.compute()


class AverageMetric(IMetric):
    def __init__(self, compute_on_call: bool = True):
        super().__init__(compute_on_call=compute_on_call)
        self.n = 0
        self.value = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.num_samples = 0

    def reset(self) -> None:
        self.n = 0
        self.value = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.num_samples = 0

    def update(self, value: float, num_samples: int) -> None:
        self.value = value
        self.n += 1
        self.num_samples += num_samples

        if self.n == 1:
            # Force a copy in torch/numpy
            self.mean = 0.0 + value  # noqa: WPS345
            self.std = 0.0
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (
                value - self.mean_old
            ) * num_samples / float(self.num_samples)
            self.m_s += (
                (value - self.mean_old) * (value - self.mean) * num_samples
            )
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.num_samples - 1.0))

    def compute(self) -> Any:
        return self.mean, self.std

    def compute_key_value(self) -> Dict[str, float]:
        raise NotImplementedError()


class AccuracyMetric(AverageMetric):
    def update(self, logits, targets) -> None:
        logits = [unvalue(x) for x in logits]
        accuracy = [(yi > 0) == (li > 0) for yi, li in zip(targets, logits)]
        value = sum(accuracy) / len(accuracy)
        super().update(value, len(accuracy))

    def compute_key_value(self) -> Dict[str, float]:
        mean, std = super().compute()
        return {"accuracy_mean": mean, "accuracy_std": std}


class AUCMetric(IMetric):
    def __init__(self, compute_on_call: bool = True):
        super().__init__(compute_on_call=compute_on_call)
        self.inputs = []
        self.targets = []

    def reset(self) -> None:
        self.inputs = []
        self.targets = []

    def update(self, logits, targets) -> None:
        logits = [unvalue(x) for x in logits]
        self.inputs.extend(logits)
        self.targets.extend(targets)

    def compute(self) -> float:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        y_true = (np.array(self.targets) > 0).astype(np.int32)
        y_score = sigmoid(np.array(self.inputs))
        score = roc_auc_score(y_true=y_true, y_score=y_score)
        return score

    def compute_key_value(self) -> Dict[str, float]:
        return {"auc": self.compute()}
