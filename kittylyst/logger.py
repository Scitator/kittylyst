from typing import Dict, Optional
from abc import ABC, abstractmethod

import numpy as np

from kittylyst.misc import format_metrics, unvalue


class ILogger(ABC):
    """An abstraction that syncs experiment run with monitoring tools."""

    @property
    def name(self) -> str:
        return None

    @property
    def logdir(self) -> str:
        return None

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None,
    ) -> None:
        pass

    @abstractmethod
    def log_image(self, image: np.ndarray, step: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def log_hparams(self, hparams: Dict) -> None:
        pass


class ConsoleLogger(ILogger):
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None,
    ) -> None:
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        msg = f" Step ({step}) " + format_metrics(metrics)
        print(msg)

    def log_image(self, image: np.ndarray, step: Optional[int] = None) -> None:
        pass

    def log_hparams(self, hparams: Dict) -> None:
        print(f"Hparams: {hparams}")
