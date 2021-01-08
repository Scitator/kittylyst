from typing import Dict, List, Optional
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
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        scope: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def log_image(
        self,
        image: np.ndarray,
        step: Optional[int] = None,
        scope: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def log_hparams(self, hparams: Dict) -> None:
        pass


# @TODO: scope should be enum
# @TODO: logger could have extra init params for logging level choice
class ConsoleLogger(ILogger):
    def __init__(self, include: List[str] = None, exclude: List[str] = None):
        self.include = include
        self.exclude = exclude

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        step_limit: Optional[int] = None,
        scope: Optional[str] = None,
        scope_name: Optional[str] = None,
    ) -> None:
        if self.exclude is not None and scope in self.exclude:
            return
        elif (
            self.include is not None and scope in self.include
        ) or self.include is None:
            scope = scope_name or scope
            prefix = (
                f"{scope} ({step}) "
                if step_limit is None
                else f"{scope} ({step}/{step_limit}) "
            )
            msg = prefix + format_metrics(metrics)
            print(msg)

    def log_image(
        self,
        image: np.ndarray,
        step: Optional[int] = None,
        scope: Optional[str] = None,
    ) -> None:
        pass

    def log_hparams(self, hparams: Dict) -> None:
        print(f"Hparams: {hparams}")
