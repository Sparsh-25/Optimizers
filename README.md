# Optimizer Race

A small Streamlit game for our UCS668 assignment on Module 5 (training and optimization).

Training a network means walking downhill on a loss surface. In this game you are the optimizer. Each level is a 2D loss surface f(x, y) with a ball on it. You pick an optimizer (SGD, Momentum, Nesterov, Adagrad, RMSprop or Adam), set its learning rate and other settings, and press Launch. The ball moves using that optimizer's real update rule. You win if it reaches the minimum within the step budget.

Every level is built so that plain SGD fails in one specific way, and each level's lesson explains why. The game is our answer to "when does SGD not work?"

## Why 2D?

A real network has millions of weights, so you can't draw its loss surface. With two weights you can draw the whole surface and watch every step. The update rules are the same ones `torch.optim` uses. Only the number of weights is different.

## Levels

| # | Level | What goes wrong for SGD | What fixes it |
|---|---|---|---|
| 1 | The Bowl | lr too big, the ball explodes | lr < 2/λ_max |
| 2 | The Ravine | Steep one way, flat the other (condition number 100) | Momentum, Adam |
| 3 | The Saddle Pass | Gradient is almost zero on the ridge | Adam, RMSprop |
| 4 | Twin Valleys | Stops in the shallow local minimum | Momentum |
| 5 | Stochastic Storm | Noisy gradients keep it jittering | LR decay |
| 6 | Rosenbrock's Banana | Narrow curved valley, very slow | Tuned Momentum or Adam |
| 7 | Wine (raw) | Real data with an unnormalised feature (condition number 189) | Momentum, Adam, or normalise |
| 8 | Wine (normalised) | Small mini-batches give noisy gradients | Bigger batch or LR decay |

Levels 7 and 8 fit a line, alcohol = w × colour intensity + b, to the 178 wines in scikit-learn's Wine dataset. The ball's position is (w, b), so the contour plot is the real MSE loss over the dataset.

## Running it

You need Python 3.10 or newer.

```
pip install streamlit numpy plotly matplotlib scikit-learn torch
streamlit run app.py
```

Torch is only used by `test_verify.py`. The game itself runs on NumPy.

The app has three tabs:

- **Campaign**: pick a level, read the mission, choose an optimizer and press Launch. The ball starts rolling straight away with the loss curve drawing next to it. When it stops you get the result, stars, a coach message and the lesson.
- **Race**: run 2 to 4 optimizers on the same level. Predict the winner before the race for 300 bonus points.
- **Theory**: update rules, the "when does SGD fail" table, and why each level uses the numbers it does.

## Files

```
optimizers.py      the six optimizers, written from scratch in NumPy
landscapes.py      level definitions: f, hand-derived gradient, start, goal, rules, texts
engine.py          runs one attempt and decides the outcome and score
app.py             the Streamlit app
test_verify.py     checks the maths against finite differences and PyTorch
tune.py            sweeps every optimizer over 60 learning rates on every level
tuning_log.md      output of tune.py
demo_cheatsheet.md settings we use in the demo, all tested
```

Each optimizer is a class with a `step(p, g, lr)` method. A new object is created for every run, so state like Adam's m and v never carries over between runs.

Each level is a `Level` dataclass in `landscapes.py`. To add a level, add another `Level(...)` to the `LEVELS` list. No other code needs to change.

## Checking the maths

```
python3 test_verify.py
```

This runs four checks:

1. Every hand-written gradient matches a finite-difference gradient at 20 random points per level.
2. Every hand-written gradient matches PyTorch autograd.
3. Each of our six optimizers follows the same path as its `torch.optim` version for 100 steps on every noise-free level, with and without `ExponentialLR` decay. All in float64, because float32 rounding makes the paths drift.
4. On the Wine levels, the mini-batch gradient over all 178 samples equals the full gradient.

Each check prints its results and the script finishes with `ALL CHECKS PASSED`.

## How we picked the numbers

Some numbers come straight from theory. For example, on the Ravine SGD is only stable for lr < 2/100 = 0.02, and on Twin Valleys SGD always settles in the shallow valley when lr < 2/3.3 ≈ 0.6. That's why the lr slider on that level stops at 0.6.

Budgets, tolerances and lr limits were tuned with `tune.py`. For each level we checked that SGD fails at every allowed learning rate and that the intended optimizer can win. The full results are in `tuning_log.md`, and each level's reasoning is in its "Why these numbers?" box in the app.

## Things we noticed while tuning

- Adam is not always the best. With a constant lr it fails on the Bowl and on Twin Valleys at every learning rate we tried.
- RMSprop's first step is about 10 × lr because it has no bias correction. On the Saddle it reaches the goal in one step. Adam's bias correction exists to prevent this.
- Near the divergence limit some learning rates win by luck, flying straight through the goal. That's why Levels 5 to 8 make the ball stay inside the goal for several steps in a row.

## Limitations

- Only Levels 5 and 8 are stochastic. The other levels use exact gradients, so strictly they show gradient descent, not stochastic gradient descent.
- Noisy levels use a fixed random seed so runs can be repeated. A different seed gives different results.
- The saddle and local-minimum outcomes use simple thresholds on the gradient and the Hessian, so they are rules of thumb.
- A 2D linear model's loss is always a bowl shape. The Wine levels can show ill-conditioning and mini-batch noise, but not saddles or local minima. The synthetic levels cover those.
