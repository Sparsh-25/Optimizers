"""
Optimizers written from scratch in NumPy.
Formulas follow torch.optim so we can compare against PyTorch in test_verify.py.

p = position (x, y), g = gradient at p, lr = learning rate for this step.
All operations are element-wise.
"""
import numpy as np


class Optimizer:
    """Base class. Every optimizer must implement step(p, g, lr) -> new p."""
    name = "Optimizer"

    def step(self, p, g, lr):
        raise NotImplementedError


class SGD(Optimizer):
    name = "SGD"

    def step(self, p, g, lr):
        # p = p - lr * g
        return p - lr * g


class Momentum(Optimizer):
    name = "Momentum"

    def __init__(self, mu=0.9):
        self.mu = mu
        self.v = None

    def step(self, p, g, lr):
        if self.v is None:
            self.v = np.zeros_like(p)
        # velocity remembers past gradients
        self.v = self.mu * self.v + g
        return p - lr * self.v


class Nesterov(Momentum):
    name = "Nesterov"

    def step(self, p, g, lr):
        if self.v is None:
            self.v = np.zeros_like(p)
        self.v = self.mu * self.v + g
        # "look ahead": use g + mu*v instead of just v
        return p - lr * (g + self.mu * self.v)


class Adagrad(Optimizer):
    name = "Adagrad"

    def __init__(self, eps=1e-10):
        self.eps = eps
        self.s = None

    def step(self, p, g, lr):
        if self.s is None:
            self.s = np.zeros_like(p)
        # s only grows, so the effective lr keeps shrinking
        self.s = self.s + g ** 2
        return p - lr * g / (np.sqrt(self.s) + self.eps)


class RMSprop(Optimizer):
    name = "RMSprop"

    def __init__(self, alpha=0.99, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.s = None

    def step(self, p, g, lr):
        if self.s is None:
            self.s = np.zeros_like(p)
        # like Adagrad but old gradients are slowly forgotten
        self.s = self.alpha * self.s + (1 - self.alpha) * g ** 2
        return p - lr * g / (np.sqrt(self.s) + self.eps)


class Adam(Optimizer):
    name = "Adam"

    def __init__(self, b1=0.9, b2=0.999, eps=1e-8):
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, p, g, lr):
        if self.m is None:
            self.m = np.zeros_like(p)
            self.v = np.zeros_like(p)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g        # momentum part
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2   # RMSprop part
        # bias correction: m and v start at 0, so early values are too small
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return p - lr * m_hat / (np.sqrt(v_hat) + self.eps)


# used by the app to build a fresh optimizer from a dropdown choice
OPTIMIZERS = {
    "SGD": SGD,
    "Momentum": Momentum,
    "Nesterov": Nesterov,
    "Adagrad": Adagrad,
    "RMSprop": RMSprop,
    "Adam": Adam,
}


def make_optimizer(name, **params):
    """Always create a NEW object per run, so old m/v/s state is not reused."""
    return OPTIMIZERS[name](**params)


if __name__ == "__main__":
    # quick sanity check on the bowl f = x^2 + y^2 (gradient = 2p)
    for name in OPTIMIZERS:
        opt = make_optimizer(name)
        p = np.array([-4.0, 3.5])
        for _ in range(200):
            p = opt.step(p, 2 * p, lr=0.1)
        print(f"{name:9s} -> {p}")

    # SGD with lr = 0.5 should land exactly on (0, 0) in one step
    print("SGD lr=0.5, 1 step:", SGD().step(np.array([-4.0, 3.5]), np.array([-8.0, 7.0]), 0.5))
