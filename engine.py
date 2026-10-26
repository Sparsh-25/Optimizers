"""
The game engine: runs one attempt of an optimizer on a level and decides the outcome.
Each loop iteration = one training step: gradient -> optimizer update -> new loss.
"""
from dataclasses import dataclass
import numpy as np
from optimizers import make_optimizer

REACHED, DIVERGED, TRAPPED, SADDLE, TOO_SLOW = "Reached", "Diverged", "Trapped", "Saddle", "Too slow"
EMOJI = {REACHED: "✅", DIVERGED: "💥", TRAPPED: "🪤", SADDLE: "🐴", TOO_SLOW: "🐢"}


@dataclass
class Result:
    path: np.ndarray     # (steps + 1, 2) positions, starting point included
    losses: np.ndarray   # f at each position
    outcome: str
    steps: int
    points: int
    stars: int
    message: str


def hessian(level, p, h=1e-5):
    """Second derivatives by finite differences of the (hand-written) gradient."""
    H = np.zeros((2, 2))
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        H[:, i] = (level.grad(p + e) - level.grad(p - e)) / (2 * h)
    return (H + H.T) / 2


def stable_lr(level):
    """SGD is stable only if lr < 2 / lambda_max (largest curvature at the start)."""
    lam_max = np.linalg.eigvalsh(hessian(level, np.array(level.start, float))).max()
    return 2 / lam_max


def score(steps, budget):
    points = int(200 + 1000 * (1 - steps / budget))
    if steps <= 0.3 * budget:
        stars = 3
    elif steps <= 0.6 * budget:
        stars = 2
    else:
        stars = 1
    return points, stars


def run(level, opt_name, lr, decay=1.0, batch_size=None, **opt_params):
    """batch_size: only for real-data levels, None = use all samples (full batch)."""
    opt = make_optimizer(opt_name, **opt_params)   # fresh optimizer every run
    rng = np.random.default_rng(level.seed)
    goals = np.array(level.goals)

    p = np.array(level.start, dtype=float)
    path, losses = [p], [level.f(p)]
    inside = 0          # consecutive steps inside the tolerance
    outcome = None

    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, level.budget + 1):
            g = level.grad(p)
            if level.noise > 0:
                g = g + level.noise * rng.standard_normal(2)   # simulated mini-batch noise
            if batch_size:
                idx = rng.choice(len(level.data[0]), batch_size, replace=False)
                g = level.batch_grad(p, idx)                   # real mini-batch gradient
            lr_t = lr * decay ** (t - 1)    # same as torch ExponentialLR
            p = opt.step(p, g, lr_t)
            path.append(p)
            losses.append(level.f(p))

            if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e3:
                outcome = DIVERGED
                break
            dist = np.linalg.norm(goals - p, axis=1).min()
            inside = inside + 1 if dist < level.tol else 0
            if inside >= level.hold:
                outcome = REACHED
                break

    steps = len(path) - 1
    if outcome is None:
        # budget ran out: look at where the ball stopped
        g_true = np.linalg.norm(level.grad(p))
        min_eig = np.linalg.eigvalsh(hessian(level, p)).min()
        if min_eig < 0 and g_true < 0.5:
            outcome = SADDLE
        elif g_true < 1e-3:
            outcome = TRAPPED
        else:
            outcome = TOO_SLOW

    points, stars = score(steps, level.budget) if outcome == REACHED else (0, 0)
    return Result(np.array(path), np.array(losses), outcome, steps, points, stars,
                  coach(level, opt_name, outcome, steps, noisy=level.noise > 0 or bool(batch_size)))


def coach(level, opt_name, outcome, steps, noisy=False):
    if outcome == REACHED:
        return f"Reached the goal in {steps} steps!"
    if outcome == DIVERGED:
        return (f"Your lr was too high and the ball flew off. On this level SGD is stable "
                f"only if lr < 2/lambda_max = {stable_lr(level):.3g}.")
    if outcome == SADDLE:
        return ("You're sitting on a saddle point: the gradient is almost zero but it is not "
                "a minimum. Try Adam, which takes full-size steps even when gradients are tiny.")
    if outcome == TRAPPED:
        return "You're stuck in a local minimum. Try Momentum to roll over the hill."
    if noisy:
        return "The noise keeps the ball jittering around the goal. Try lr decay (gamma < 1)."
    if opt_name == "Adagrad":
        return "Adagrad's effective lr kept shrinking until it nearly stopped. Try RMSprop or Adam."
    return "The loss is still falling, but slowly. Try a bigger lr, Momentum or Adam."


if __name__ == "__main__":
    from landscapes import LEVELS
    bowl, ravine, saddle, twin, storm, rosen = LEVELS
    tests = [
        (bowl, "SGD", 0.5, 1.0, {}),
        (bowl, "SGD", 1.2, 1.0, {}),
        (bowl, "SGD", 0.01, 1.0, {}),
        (ravine, "SGD", 0.019, 1.0, {}),
        (ravine, "Momentum", 0.03, 1.0, {}),
        (ravine, "Momentum", 0.033, 1.0, {"mu": 0.67}),
        (saddle, "SGD", 0.19, 1.0, {}),
        (saddle, "Adam", 0.1, 1.0, {}),
        (twin, "SGD", 0.1, 1.0, {}),
        (twin, "Momentum", 0.3, 1.0, {}),
        (storm, "SGD", 0.1, 1.0, {}),
        (storm, "SGD", 0.3, 0.97, {}),
        (rosen, "SGD", 0.001, 1.0, {}),
        (rosen, "Adagrad", 0.1, 1.0, {}),
    ]
    for lv, name, lr, decay, params in tests:
        r = run(lv, name, lr, decay, **params)
        print(f"{lv.name:22s} {name:8s} lr={lr:<6} {EMOJI[r.outcome]} {r.outcome:8s} "
              f"steps={r.steps:4d} pts={r.points:4d} {'★' * r.stars}")
        print(f"{'':24s}coach: {r.message}")
