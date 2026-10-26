"""
Proves our maths is right. Run:  python3 test_verify.py   (or: pytest test_verify.py)

1. Gradient check: hand-written gradient vs finite differences (f(p+h) - f(p-h)) / 2h
2. Gradient check: hand-written gradient vs PyTorch autograd
3. Optimizer check: our NumPy optimizers vs torch.optim, same start, same steps
"""
import numpy as np
import torch
from landscapes import LEVELS
from optimizers import make_optimizer

# torch.optim equivalents of our optimizers (PyTorch default hyperparameters)
TORCH = {
    "SGD": (torch.optim.SGD, {}),
    "Momentum": (torch.optim.SGD, {"momentum": 0.9}),
    "Nesterov": (torch.optim.SGD, {"momentum": 0.9, "nesterov": True}),
    "Adagrad": (torch.optim.Adagrad, {}),
    "RMSprop": (torch.optim.RMSprop, {}),
    "Adam": (torch.optim.Adam, {}),
}


def numerical_grad(f, p, h=1e-6):
    g = np.zeros(2)
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        g[i] = (f(p + e) - f(p - e)) / (2 * h)
    return g


def test_gradients_finite_difference():
    rng = np.random.default_rng(42)
    for lv in LEVELS:
        xmin, xmax, ymin, ymax = lv.bounds
        worst = 0
        for _ in range(20):
            p = rng.uniform([xmin, ymin], [xmax, ymax])
            # relative error, so big gradients (Rosenbrock) are judged fairly
            err = np.abs(lv.grad(p) - numerical_grad(lv.f, p)).max() / (1 + np.abs(lv.grad(p)).max())
            worst = max(worst, err)
        print(f"  {lv.name:32s} max relative error = {worst:.1e}")
        assert worst < 1e-6, lv.name


def test_gradients_autograd():
    rng = np.random.default_rng(7)
    for lv in LEVELS:
        xmin, xmax, ymin, ymax = lv.bounds
        for _ in range(20):
            p = rng.uniform([xmin, ymin], [xmax, ymax])
            tp = torch.tensor(p, dtype=torch.float64, requires_grad=True)
            lv.f(tp).backward()     # our f also works on torch tensors
            assert np.allclose(lv.grad(p), tp.grad.numpy(), atol=1e-9), lv.name
        print(f"  {lv.name:32s} matches autograd")


def run_numpy(lv, name, lr, gamma, steps):
    opt, p = make_optimizer(name), np.array(lv.start, dtype=float)
    for t in range(steps):
        p = opt.step(p, lv.grad(p), lr * gamma ** t)
    return p


def run_torch(lv, name, lr, gamma, steps):
    # float64, otherwise float32 rounding makes the paths drift apart
    tp = torch.tensor(lv.start, dtype=torch.float64, requires_grad=True)
    cls, kw = TORCH[name]
    opt = cls([tp], lr=lr, **kw)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    for _ in range(steps):
        opt.zero_grad()
        lv.f(tp).backward()
        opt.step()
        sched.step()
    return tp.detach().numpy()


def test_optimizers_match_torch():
    # noise-free levels only (Level 5 uses the bowl, which is already covered)
    lr_for = {"1. The Bowl": 0.05, "2. The Ravine": 0.01, "3. The Saddle Pass": 0.05,
              "4. Twin Valleys": 0.05, "6. Rosenbrock's Banana": 0.001,
              "7. Real Data: Wine (raw)": 0.01, "8. Real Data: Wine (normalised)": 0.05}
    for lv in LEVELS:
        if lv.noise > 0:
            continue
        lr = lr_for[lv.name]
        for name in TORCH:
            for gamma in (1.0, 0.97):    # with and without lr decay
                ours = run_numpy(lv, name, lr, gamma, steps=100)
                theirs = run_torch(lv, name, lr, gamma, steps=100)
                assert np.allclose(ours, theirs, atol=1e-6), (lv.name, name, gamma)
        print(f"  {lv.name:32s} all 6 optimizers match torch.optim (with and without decay)")


def test_real_data_gradient():
    # the expanded MSE formula must equal the plain sample-by-sample gradient
    rng = np.random.default_rng(3)
    for lv in LEVELS:
        if lv.batch_grad is None:
            continue
        x, y = lv.data
        everyone = np.arange(len(x))
        for _ in range(20):
            p = rng.uniform([-2, 5], [2, 15])
            loss = np.mean((p[0] * x + p[1] - y) ** 2)
            assert np.isclose(lv.f(p), loss), lv.name
            assert np.allclose(lv.batch_grad(p, everyone), lv.grad(p)), lv.name
        print(f"  {lv.name:32s} batch gradient on all samples = full gradient")


if __name__ == "__main__":
    print("1. Hand gradient vs finite differences")
    test_gradients_finite_difference()
    print("2. Hand gradient vs PyTorch autograd")
    test_gradients_autograd()
    print("3. NumPy optimizers vs torch.optim (100 steps)")
    test_optimizers_match_torch()
    print("4. Real-data levels: expanded formula vs sample-by-sample")
    test_real_data_gradient()
    print("\nALL CHECKS PASSED ✅")
