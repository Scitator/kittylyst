import random

import numpy as np

from micrograd.engine import Value


def set_random_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


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
        self.num_epochs = num_epochs

    def step(self, epoch):
        learning_rate = 1.0 - 0.9 * epoch / self.num_epochs
        self.optimizer.lr = learning_rate
