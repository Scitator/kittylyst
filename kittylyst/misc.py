from typing import Dict, List, Union
import json
from pathlib import Path
import random

import numpy as np
import yaml

from micrograd.engine import Value


def set_random_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


def unvalue(value):
    return value.data if isinstance(value, Value) else value


def format_metrics(dct: Dict):
    return " ".join([f"{k}: {unvalue(dct[k])}" for k in sorted(dct.keys())])


def save_config(
    config: Union[Dict, List],
    path: Union[str, Path],
    data_format: str = None,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    path = Path(path)

    if data_format is not None:
        suffix = data_format
    else:
        suffix = path.suffix

    assert suffix in [
        ".json",
        ".yml",
        ".yaml",
    ], f"Unknown file format '{suffix}'"

    with path.open(encoding=encoding, mode="w") as stream:
        if suffix == ".json":
            json.dump(config, stream, indent=indent, ensure_ascii=ensure_ascii)
        elif suffix in [".yml", ".yaml"]:
            yaml.dump(config, stream)


class MicroLoader:
    def __init__(self, X, y, batch_size=32, num_batches=10):
        self.X, self.y = X, y
        self.batch_size, self.num_batches = batch_size, num_batches
        self.iteration_index = 0

    def __iter__(self):
        self.iteration_index = 0
        return self

    def __next__(self):
        if self.iteration_index >= self.num_batches:
            raise StopIteration()
        self.iteration_index += 1
        ri = np.random.permutation(self.X.shape[0])[: self.batch_size]
        Xb, yb = self.X[ri], self.y[ri]
        Xb = [list(map(Value, xrow)) for xrow in Xb]
        return Xb, yb

    def __len__(self) -> int:
        return self.num_batches


class MicroCriterion:
    def __call__(self, logits, targets):
        losses = [(1 + -yi * li).relu() for yi, li in zip(targets, logits)]
        loss = sum(losses) * (1.0 / len(losses))
        return loss


class MicroOptimizer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr

    def step(self):
        for p in self.model.parameters():
            p.data -= self.lr * p.grad


class MicroScheduler:
    def __init__(self, optimizer, num_epochs):
        self.optimizer = optimizer
        self.start_lr = self.optimizer.lr
        self.num_epochs = num_epochs

    def step(self, epoch):
        learning_rate = (
            self.start_lr - self.start_lr * 0.9 * epoch / self.num_epochs
        )
        self.optimizer.lr = learning_rate
