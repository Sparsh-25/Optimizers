"""
Tuning script: tries every optimizer over a grid of learning rates on every level
and records the best result. Used to check that SGD fails where it should and to
build the demo cheat sheet. Run:  python3 tune.py   (writes tuning_log.md)
"""
import numpy as np
from landscapes import LEVELS
from optimizers import OPTIMIZERS
from engine import run, REACHED, EMOJI

DECAYS = [1.0, 0.99, 0.97, 0.95]


def sweep(level, opt_name, decay):
    """Try lrs from 1e-4 up to the level's slider limit, return the best result."""
    best_lr, best = None, None
    counts = {}
    for lr in np.geomspace(1e-4, level.lr_max, 60):
        r = run(level, opt_name, lr, decay)
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
        if r.outcome == REACHED and (best is None or r.steps < best.steps):
            best_lr, best = lr, r
    return best_lr, best, counts


lines = ["# Tuning log", "",
         "Each optimizer was tried with 60 learning rates from 1e-4 up to the level's lr limit, "
         "with default hyperparameters (mu = 0.9, Adam betas 0.9/0.999).", ""]

for lv in LEVELS:
    lines += [f"## {lv.name}  (budget {lv.budget}, tol {lv.tol})", "",
              "| Optimizer | Decay | Best lr | Steps | Stars | Outcomes over all lrs |",
              "|---|---|---|---|---|---|"]
    print(f"\n{lv.name}")
    for name in OPTIMIZERS:
        # decay is only interesting on the noisy level
        for decay in (DECAYS if lv.noise > 0 else [1.0]):
            lr, r, counts = sweep(lv, name, decay)
            summary = ", ".join(f"{EMOJI[k]} {v}" for k, v in counts.items())
            if r is None:
                row = f"| {name} | {decay} | - | failed | - | {summary} |"
                print(f"  {name:9s} decay={decay:<5} FAILED for every lr   ({summary})")
            else:
                row = f"| {name} | {decay} | {lr:.3g} | {r.steps} | {'★' * r.stars} | {summary} |"
                print(f"  {name:9s} decay={decay:<5} best lr={lr:<8.3g} steps={r.steps:4d} "
                      f"{'★' * r.stars}   ({summary})")
            lines.append(row)
    lines.append("")

with open("tuning_log.md", "w") as fh:
    fh.write("\n".join(lines))
print("\nSaved tuning_log.md")
