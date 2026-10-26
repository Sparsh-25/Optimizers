"""
Level definitions. Each level is just data: a loss surface f(x, y), its
hand-derived gradient, where the ball starts, where the goal is, and the rules.
Adding a new level = adding a new Level(...) to the LEVELS list.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Level:
    name: str
    f: callable              # loss f(p), p = np.array([x, y])
    grad: callable           # gradient of f, worked out by hand
    start: tuple
    goals: list              # list of (x, y) minima that count as a win
    tol: float               # how close to a goal counts as "reached"
    budget: int              # max number of steps
    bounds: tuple            # (xmin, xmax, ymin, ymax) for the plot
    briefing: str            # shown before playing
    lesson: str              # shown after playing
    why_numbers: str         # viva backup: why these values
    hold: int = 1            # steps the ball must stay inside tol (Level 5)
    noise: float = 0.0       # std of gradient noise (Level 5)
    seed: int = 0            # fixed seed so noisy runs repeat
    lr_max: float = 1.5      # upper limit of the lr slider
    log_color: bool = False  # log colour scale for steep surfaces
    # real-data levels only (Levels 7-8)
    data: tuple = None       # (x, y) samples, for the scatter plot
    batch_grad: callable = None   # gradient on a mini-batch: batch_grad(p, idx)
    axis_names: tuple = ("x (weight 1)", "y (weight 2)")

    def f_min(self):
        return min(self.f(np.array(g)) for g in self.goals)


# ---------------- Level 1: The Bowl ----------------
bowl = Level(
    name="1. The Bowl",
    f=lambda p: p[0] ** 2 + p[1] ** 2,
    grad=lambda p: np.array([2 * p[0], 2 * p[1]]),
    start=(-4.0, 3.5), goals=[(0.0, 0.0)], tol=0.05, budget=60,
    bounds=(-5, 5, -5, 5),
    briefing="A perfect bowl. Any optimizer can win here, but can you pick the right learning rate?",
    lesson="Each SGD step multiplies the position by (1 - 2*lr). lr = 0.5 lands in 1 step, "
           "0.5 < lr < 1 overshoots but converges, lr > 1 explodes. Rule: SGD is stable only if "
           "lr < 2/lambda_max, and here lambda_max = 2.",
    why_numbers="Start is about 5.3 away and off-axis so both weights move. Tol 0.05 is about 1% "
                "of that distance. Budget 60: lr = 0.1 needs about 25 steps, lr = 0.01 needs "
                "about 250, so a too-small lr runs out of steps.",
)

# ---------------- Level 2: The Ravine ----------------
ravine = Level(
    name="2. The Ravine",
    f=lambda p: 0.5 * (p[0] ** 2 + 100 * p[1] ** 2),
    grad=lambda p: np.array([p[0], 100 * p[1]]),
    start=(-9.0, 1.0), goals=[(0.0, 0.0)], tol=0.05, budget=150,
    bounds=(-10, 10, -2, 2), lr_max=1.0, log_color=True,
    briefing="A long narrow valley: very steep across, almost flat along. Plain SGD struggles here.",
    lesson="Curvature is 1 along x and 100 along y (condition number 100). SGD needs lr < 0.02 "
           "for the steep y direction, but then x moves only about 2% per step. Momentum cancels "
           "the zig-zags and builds speed along the floor. Adam gives each weight its own step "
           "size. Real networks get ravines from unnormalised input features.",
    why_numbers="Factor 100 is large enough that SGD visibly fails but momentum can still win. "
                "Start far along the flat x direction so the crawl is the bottleneck. "
                "Budget 150: SGD needs about 270 steps, momentum (mu = 0.9, lr = 0.03) 66, "
                "the theory-optimal setting (mu = 0.67, lr = 0.033) 36, and Adam (lr = 0.6) 42. The lr slider goes up to 1 so Adam can take big enough steps; SGD still fails at every lr.",
)

# ---------------- Level 3: The Saddle Pass ----------------
saddle = Level(
    name="3. The Saddle Pass",
    f=lambda p: 5 * p[0] ** 2 + 0.25 * (p[1] ** 2 - 1) ** 2,
    grad=lambda p: np.array([10 * p[0], p[1] ** 3 - p[1]]),
    start=(1.0, 0.0001), goals=[(0.0, 1.0), (0.0, -1.0)], tol=0.1, budget=40,
    bounds=(-1.5, 1.5, -1.8, 1.8), lr_max=0.5,
    briefing="The ball starts on a mountain pass, almost exactly on the ridge. Roll off into either valley.",
    lesson="The origin is a saddle: a minimum along x, a maximum along y, and the gradient is "
           "almost 0 there. SGD multiplies y by only (1 + lr) per step and lr must stay below "
           "0.2 for x, so it looks stuck. Adam divides by the gradient's size, so even a tiny "
           "gradient gives a full step. In deep networks saddles are far more common than bad "
           "local minima.",
    why_numbers="5x^2 gives curvature 10, which caps SGD at lr < 0.2. y = 0.0001 means SGD "
                "needs about ln(10^4)/ln(1.2) = 50 steps to escape (exactly 0 would never "
                "escape). Budget 40 sits between SGD (about 50) and Adam (about 10).",
)

# ---------------- Level 4: Twin Valleys ----------------
global_x = min(np.roots([1, 0, -2, 0.3]).real)   # root of x^3 - 2x + 0.3 = 0, about -1.484
twin = Level(
    name="4. Twin Valleys",
    f=lambda p: 0.25 * p[0] ** 4 - p[0] ** 2 + 0.3 * p[0] + 0.5 * p[1] ** 2,
    grad=lambda p: np.array([p[0] ** 3 - 2 * p[0] + 0.3, p[1]]),
    start=(1.8, 1.0), goals=[(global_x, 0.0)], tol=0.1, budget=150,
    bounds=(-2.5, 2.5, -2, 2), lr_max=0.6,
    briefing="Two valleys: a shallow one near you and a deeper one over the hill. Only the deep one counts.",
    lesson="SGD stops in the first valley it reaches (local minimum near x = 1.33). Momentum's "
           "velocity can carry the ball over the hill near x = 0.15 into the global minimum "
           "near x = -1.48. Momentum does NOT guarantee the global minimum; it only helps "
           "escape shallow ones.",
    why_numbers="The 0.3x term tilts the double well so the minima differ (f = -0.59 vs -1.43). "
                "Start (1.8, 1) is on the shallow side. We tested every SGD lr from 0.01 to 0.6 "
                "and none reaches the global minimum, so the slider stops at 0.6 (larger lrs "
                "only win by lucky chaotic bounces or diverge).",
)

# ---------------- Level 5: Stochastic Storm ----------------
storm = Level(
    name="5. Stochastic Storm",
    f=bowl.f, grad=bowl.grad,
    start=(-4.0, 3.5), goals=[(0.0, 0.0)], tol=0.1, budget=120,
    bounds=(-5, 5, -5, 5), hold=10, noise=1.5, seed=0,
    briefing="The same bowl, but every gradient is noisy, like a mini-batch. Stay inside the goal for 10 steps in a row.",
    lesson="Noise simulates mini-batch gradients. With a constant lr the ball reaches the area "
           "but keeps jittering, because the noise never goes away. LR decay shrinks the steps "
           "so the jitter dies down. A tiny constant lr jitters less but is too slow; decay "
           "gets both: big steps early, small steps late. This is why we use lr schedulers.",
    why_numbers="Noise std 1.5 is small next to the gradient at the start (about 10) but "
                "dominates near the minimum, just like mini-batches. Tol 0.1 and budget 120: "
                "out of 60 lrs tried, constant-lr SGD wins for only 3 (a very small lr, lucky), "
                "while decay (gamma = 0.97) wins for 16, e.g. lr = 0.3 in 94 steps. Hold 10 "
                "stops lucky pass-throughs. Fixed seed makes runs repeatable.",
)

# ---------------- Level 6: Rosenbrock's Banana ----------------
rosen = Level(
    name="6. Rosenbrock's Banana",
    f=lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2,
    grad=lambda p: np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2),
                             200 * (p[1] - p[0] ** 2)]),
    start=(-1.5, 2.0), goals=[(1.0, 1.0)], tol=0.05, budget=1500,
    bounds=(-2, 2, -1, 3), lr_max=0.5, log_color=True, hold=5,
    briefing="The boss level: a narrow curved banana valley. Finding it is easy, following it to (1, 1) is hard.",
    lesson="Rosenbrock is the classic optimizer benchmark. The valley is steep across and very "
           "flat along its curve, so SGD needs a tiny lr and then crawls. Tuned Adam or "
           "Momentum follows the curve much faster.",
    why_numbers="Start (-1.5, 2) is the far side of the valley, so the whole curve must be "
                "followed. At (1, 1) the largest curvature is about 1000, so SGD needs "
                "lr < 0.002 and would take about 10,000 steps. Budget 1500 leaves room for "
                "tuned Momentum (lr = 0.0015, about 440 steps) or Adam (lr = 0.15, about 610). "
                "Hold 5 because near-divergent runs can fly through (1, 1) by luck.",
)

# ---------------- Levels 7-8: real data (Wine dataset) ----------------
# Model: alcohol = w * colour_intensity + b, loss = mean squared error.
# The two weights (w, b) are the ball's position, so the contour is the real loss surface.

def wine_level(normalise, **settings):
    from sklearn.datasets import load_wine   # bundled with sklearn, no download
    wine = load_wine()
    x = wine.data[:, wine.feature_names.index("color_intensity")]
    y = wine.data[:, wine.feature_names.index("alcohol")]
    if normalise:
        x = (x - x.mean()) / x.std()

    # MSE = mean((w*x + b - y)^2). Expanding the square, the data only enters
    # through these averages (this also lets f work on plot grids and torch tensors).
    Sxx, Sx, Sxy, Sy, Syy = np.mean(x * x), np.mean(x), np.mean(x * y), np.mean(y), np.mean(y * y)

    def f(p):
        w, b = p[0], p[1]
        return w * w * Sxx + 2 * w * b * Sx + b * b - 2 * w * Sxy - 2 * b * Sy + Syy

    def grad(p):
        w, b = p
        return np.array([2 * (w * Sxx + b * Sx - Sxy), 2 * (w * Sx + b - Sy)])

    def batch_grad(p, idx):
        # same gradient, but averaged over a random mini-batch of samples
        w, b = p
        r = w * x[idx] + b - y[idx]          # residuals (prediction errors)
        return np.array([2 * np.mean(r * x[idx]), 2 * np.mean(r)])

    best = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0]   # least squares answer
    return Level(f=f, grad=grad, batch_grad=batch_grad, data=(x, y), goals=[tuple(best)],
                 axis_names=("w (slope)", "b (intercept)"), **settings)


wine_raw = wine_level(
    normalise=False,
    name="7. Real Data: Wine (raw)",
    start=(-1.0, 8.0), tol=0.05, budget=200, bounds=(-1.5, 1.5, 6, 16),
    lr_max=0.5, log_color=True, hold=5,
    briefing="Real data! Fit a line predicting a wine's alcohol % from its colour intensity "
             "(178 real wines). The ball is the line's (slope, intercept).",
    lesson="This is the Ravine level, but on real data. Colour intensity is about 5 on average, "
           "so changing the slope w changes the loss far more than changing the intercept b. "
           "Curvatures are 63.5 and 0.34 (condition number 189), so SGD needs lr < 0.0315 and "
           "then crawls along b. Momentum and Adam fix it, but the real fix is in the next level: "
           "normalise the input.",
    why_numbers="Colour intensity was picked because its condition number (189) is close to "
                "the Ravine's 100; alcohol or proline as inputs give 44,000 and 4.4 million, "
                "which are unplayable. Start (-1, 8) is a bad line far from the answer "
                "(w = 0.19, b = 12.03, from least squares). Budget 200: SGD fails at every lr, "
                "Momentum (lr 0.012) takes 73 steps, Nesterov (lr 0.016) 49, Adam (lr 0.5) 94. The lr limit of 0.5 lets "
                "Adam take big enough steps. Hold 5 stops lucky fly-throughs.",
)

wine_norm = wine_level(
    normalise=True,
    name="8. Real Data: Wine (normalised)",
    start=(-1.0, 8.0), tol=0.05, budget=200, bounds=(-1.5, 1.5, 6, 16),
    lr_max=1.0, hold=10,
    briefing="Same data, but colour intensity is normalised (mean 0, std 1). Watch what happens "
             "to the landscape. Then try small mini-batches.",
    lesson="After normalising, both curvatures are exactly 2 (condition number 1): the ravine "
           "became the Bowl, and plain SGD with lr = 0.5 lands on the answer in 1 step (10 with the hold). This is why we "
           "normalise inputs. With mini-batches the gradient is computed on a random sample of "
           "wines, so it is noisy: smaller batches = more noise = more jitter (Level 5, but real). "
           "LR decay fixes it.",
    why_numbers="Normalising makes mean(x) = 0 and mean(x^2) = 1, so the Hessian is exactly "
                "2*I. Same start, tol and budget as Level 7 so the two are directly comparable. "
                "Hold 10 as in Level 5, because mini-batch runs can pass through by luck. Tested "
                "on 40 lrs: full batch wins for 24; batch 64 for 18; batch 4 for only 4 with "
                "constant lr but 16 with decay 0.97.",
)

LEVELS = [bowl, ravine, saddle, twin, storm, rosen, wine_raw, wine_norm]


if __name__ == "__main__":
    # print each level and check the goals really are minima (gradient = 0)
    for lv in LEVELS:
        g_at_goal = max(np.abs(lv.grad(np.array(g))).max() for g in lv.goals)
        print(f"{lv.name:24s} f(start) = {lv.f(np.array(lv.start)):9.3f}   "
              f"f_min = {lv.f_min():7.3f}   |grad| at goal = {g_at_goal:.1e}")
