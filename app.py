"""
Optimizer Race - Streamlit app.   Run:  streamlit run app.py
"""
import base64
import io
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from landscapes import LEVELS
from engine import run, stable_lr, EMOJI, REACHED, DIVERGED

st.set_page_config(page_title="Optimizer Race", page_icon="⛳", layout="wide")

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
MAX_FRAMES = 80

# Streamlit reruns the whole script on every click, so the score lives in session_state
if "score" not in st.session_state:
    st.session_state.score = 0
    st.session_state.best = {}        # level name -> best points
    st.session_state.last = None      # last campaign result
    st.session_state.race = None      # last race result
    st.session_state.run_id = 0       # counts launches, so each new run auto-plays

BET_BONUS = 300


def race_rank(r):
    """Sort key: finishers by steps, then survivors by final loss, diverged runs last."""
    if r.outcome == REACHED:
        return (0, r.steps)
    if r.outcome == DIVERGED:
        return (2, -r.steps)          # the one that survived longer ranks higher
    return (1, r.losses[-1])


# ---------------- plotting helpers (also used by the Race tab) ----------------

@st.cache_data
def surface_image(level_name):
    """The loss surface as a PNG picture (made once per level, then cached).
    We use a picture instead of a Plotly contour trace because Plotly hides
    contour traces while an animation is playing."""
    level = next(lv for lv in LEVELS if lv.name == level_name)
    xmin, xmax, ymin, ymax = level.bounds
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
    Z = level.f(np.array([X, Y]))
    if level.log_color:
        # steep levels: contours bunch up without a log scale
        Z = np.log(Z - level.f_min() + 1)
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax.contour(X, Y, Z, levels=30, colors="black", linewidths=0.3, alpha=0.4)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_figure(level, runs):
    """runs = list of (label, path, losses).
    Left: the ball rolling on the loss surface. Right: the loss curve, growing at the same time."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.1,
                        subplot_titles=("Loss surface", "Loss vs step (log scale)"))
    xmin, xmax, ymin, ymax = level.bounds
    fig.add_layout_image(source=surface_image(level.name), xref="x", yref="y", x=xmin, y=ymax,
                         sizex=xmax - xmin, sizey=ymax - ymin, sizing="stretch", layer="below")
    fig.add_scatter(x=[level.start[0]], y=[level.start[1]], mode="markers", name="start",
                    marker=dict(size=12, color="white", line=dict(width=2, color="black")), row=1, col=1)
    gx, gy = zip(*level.goals)
    fig.add_scatter(x=gx, y=gy, mode="markers", name="goal", row=1, col=1,
                    marker=dict(size=16, symbol="star", color="gold", line=dict(width=1, color="black")))

    # per run: a path line, a "ball" and a loss curve, all starting at step 0
    gaps = []
    for i, (label, path, losses) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        # subtract f_min so the minimum is 0, then log scale (tiny offset avoids log(0))
        gap = np.clip(losses - level.f_min(), 1e-12, None)
        gaps.append(gap)
        fig.add_scatter(x=path[:1, 0], y=path[:1, 1], mode="lines+markers", name=label, row=1, col=1,
                        line=dict(color=c, width=2), marker=dict(size=4, color=c))
        fig.add_scatter(x=path[:1, 0], y=path[:1, 1], mode="markers", showlegend=False, row=1, col=1,
                        marker=dict(size=14, color=c, line=dict(width=2, color="white")))
        fig.add_scatter(x=[0], y=gap[:1], mode="lines", showlegend=False, row=1, col=2,
                        line=dict(color=c, width=2))

    longest = max(len(p) for _, p, _ in runs)
    frame_ids = np.unique(np.linspace(1, longest, min(MAX_FRAMES, longest)).astype(int))
    frames = []
    for k in frame_ids:
        data = []
        for (_, path, _), gap in zip(runs, gaps):
            part = path[:k]                      # finished runs just stop growing
            data.append(go.Scatter(x=part[:, 0], y=part[:, 1]))
            data.append(go.Scatter(x=part[-1:, 0], y=part[-1:, 1]))
            data.append(go.Scatter(x=np.arange(len(gap[:k])), y=gap[:k]))
        frames.append(go.Frame(data=data, name=str(k), traces=list(range(2, 2 + 3 * len(runs)))))
    fig.frames = frames

    play = dict(label="▶ Replay", method="animate",
                args=[None, {"frame": {"duration": 60, "redraw": False}, "fromcurrent": False}])
    pause = dict(label="⏸", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
    slider_steps = [dict(method="animate", label=str(k - 1),
                         args=[[str(k)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                    for k in frame_ids]

    # fixed axes, otherwise the view jumps around while animating
    finite = np.concatenate([g[np.isfinite(g)] for g in gaps])
    fig.update_xaxes(range=[xmin, xmax], title=level.axis_names[0], showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[ymin, ymax], title=level.axis_names[1], showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(range=[0, max(longest - 1, 1)], title="step", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2,   # plotted: loss minus the lowest possible loss
                     range=[np.log10(finite.min()) - 0.3, np.log10(finite.max()) + 0.3])
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=40, b=90), template="plotly_white",
        legend=dict(orientation="h", y=1.12),
        updatemenus=[dict(type="buttons", buttons=[play, pause], x=0, y=-0.2, xanchor="left", direction="left")],
        sliders=[dict(steps=slider_steps, x=0.22, y=-0.14, len=0.78, currentvalue=dict(prefix="step "))],
    )
    return fig


def show_animation(fig, run_id, wait=False):
    """Shows the figure and starts the animation by itself (Plotly's auto_play).
    run_id changes on every launch, so a new run plays again; other clicks leave it alone.
    wait=True pauses the script until the ball stops, so the result isn't shown too early."""
    html = fig.to_html(full_html=False, include_plotlyjs=True, auto_play=True,
                       animation_opts={"frame": {"duration": 60, "redraw": False},
                                       "transition": {"duration": 0}})
    components.html(f"<!-- run {run_id} -->" + html, height=fig.layout.height + 20)
    if wait:
        # while we wait, Streamlit greys out the previous run's chart and result instead of
        # removing them, so hide those until the new result is ready
        st.markdown("<style>[data-stale='true'] {display: none;}</style>", unsafe_allow_html=True)
        time.sleep(len(fig.frames) * 0.06 + 1.0)   # 60 ms per frame + about 1 s to load the chart


def fit_figure(level, runs):
    """Real-data levels: the wines, and the line each optimizer ended with."""
    x, y = level.data
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", name="wines", marker=dict(color="lightgray", size=6))
    xs = np.array([x.min(), x.max()])
    w, b = level.goals[0]
    fig.add_scatter(x=xs, y=w * xs + b, mode="lines", name="best line (least squares)",
                    line=dict(color="black", dash="dash"))
    for i, (label, path) in enumerate(runs):
        w, b = path[-1]
        if np.all(np.isfinite(path[-1])):
            fig.add_scatter(x=xs, y=w * xs + b, mode="lines", name=f"{label}: y = {w:.2f}x + {b:.2f}",
                            line=dict(color=COLORS[i % len(COLORS)], width=3))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="colour intensity" + (" (normalised)" if x.mean() ** 2 < 1e-9 else ""),
                      yaxis_title="alcohol %", yaxis_range=[y.min() - 1, y.max() + 1])
    return fig


def batch_control(level, key):
    """Real-data levels only: how many wines each gradient step looks at."""
    if level.data is None:
        return None
    choice = st.select_slider("Mini-batch size", ["all", 64, 32, 16, 8, 4], value="all", key=key + "batch",
                              help="'all' = full batch (exact gradient). Smaller = noisier gradient.")
    return None if choice == "all" else choice


def optimizer_controls(name, level, key):
    """Widgets for one optimizer. Returns (lr, decay, params)."""
    lr = st.number_input("Learning rate (lr)", min_value=1e-5, max_value=float(level.lr_max),
                         value=min(0.1, level.lr_max), step=0.01, format="%.4f", key=key + "lr")
    params = {}
    if name in ("Momentum", "Nesterov"):
        params["mu"] = st.slider("Momentum μ", 0.0, 0.99, 0.9, 0.01, key=key + "mu")
    elif name == "RMSprop":
        params["alpha"] = st.slider("α (memory of old gradients)", 0.5, 0.999, 0.99, 0.001, key=key + "a")
    elif name == "Adam":
        params["b1"] = st.slider("β₁", 0.0, 0.99, 0.9, 0.01, key=key + "b1")
        params["b2"] = st.slider("β₂", 0.9, 0.9999, 0.999, 0.0001, format="%.4f", key=key + "b2")
    decay = st.slider("LR decay γ (1 = off)", 0.90, 1.0, 1.0, 0.005, format="%.3f", key=key + "decay")
    return lr, decay, params


# ---------------- the app ----------------

st.title("⛳ Optimizer Race")
st.caption("Training a network = rolling down a loss landscape. **You are the optimizer.** Can you beat plain SGD?")

tab_campaign, tab_race, tab_theory = st.tabs(["🎮 Campaign", "🏁 Race", "📖 Theory"])

with tab_campaign:
    level = st.selectbox("Level", LEVELS, format_func=lambda lv: lv.name)
    st.info(f"**Mission:** {level.briefing}")
    rules = f"Goal: get within **{level.tol}** of ⭐ in at most **{level.budget}** steps."
    if level.hold > 1:
        rules += f" You must stay inside for **{level.hold}** steps in a row."
    if level.noise > 0:
        rules += f" Gradients are noisy (σ = {level.noise})."
    st.write(rules)

    left, right = st.columns([1, 3])
    with left:
        opt_name = st.selectbox("Optimizer", ["SGD", "Momentum", "Nesterov", "Adagrad", "RMSprop", "Adam"])
        lr, decay, params = optimizer_controls(opt_name, level, key="camp_")
        batch = batch_control(level, key="camp_")
        st.caption(f"Hint: plain SGD is stable at the start only if lr < 2/λ_max = **{stable_lr(level):.3g}**")
        launched = st.button("🚀 Launch", type="primary", use_container_width=True)
        if launched:
            result = run(level, opt_name, lr, decay, batch_size=batch, **params)
            st.session_state.last = (level.name, opt_name, result)
            st.session_state.run_id += 1
            if result.outcome == REACHED:
                st.session_state.score += result.points
                old = st.session_state.best.get(level.name, 0)
                st.session_state.best[level.name] = max(old, result.points)

    with right:
        last = st.session_state.last
        if last is None or last[0] != level.name:
            start = np.array([level.start])
            show_animation(run_figure(level, [("", start, np.array([level.f(start[0])]))]), 0)
        else:
            _, name, r = last
            show_animation(run_figure(level, [(name, r.path, r.losses)]), st.session_state.run_id, wait=launched)

    if last is not None and last[0] == level.name:
        _, name, r = last
        text = f"### {EMOJI[r.outcome]} {r.outcome} after {r.steps} steps"
        if r.outcome == REACHED:
            text += f" — {'★' * r.stars}{'☆' * (3 - r.stars)}  +{r.points} points"
        st.markdown(text)
        (st.success if r.outcome == REACHED else st.warning)(f"**Coach:** {r.message}")
        if level.data is not None:
            st.plotly_chart(fit_figure(level, [(name, r.path)]))
        with st.expander("📚 Lesson for this level", expanded=True):
            st.write(level.lesson)
        with st.expander("🔢 Why these numbers?"):
            st.write(level.why_numbers)

with tab_race:
    st.write("Race 2-4 optimizers on the same level. **Predict the winner first** for a bonus!")
    race_level = st.selectbox("Level", LEVELS, format_func=lambda lv: lv.name, key="race_level")
    racers = st.multiselect("Optimizers", ["SGD", "Momentum", "Nesterov", "Adagrad", "RMSprop", "Adam"],
                            default=["SGD", "Momentum", "Adam"], max_selections=4)

    race_batch = batch_control(race_level, key="race_")   # same batch size for everyone, to be fair
    settings = {}
    cols = st.columns(max(len(racers), 1))
    for col, name in zip(cols, racers):
        with col:
            st.markdown(f"**{name}**")
            settings[name] = optimizer_controls(name, race_level, key="race_" + name)

    started = False
    if len(racers) >= 2:
        guess = st.radio("🔮 Who will win?", racers, horizontal=True)
        started = st.button("🏁 Start race", type="primary")
        if started:
            results = {name: run(race_level, name, lr, decay, batch_size=race_batch, **params)
                       for name, (lr, decay, params) in settings.items()}
            order = sorted(results, key=lambda n: race_rank(results[n]))
            bonus = BET_BONUS if order[0] == guess else 0
            st.session_state.score += bonus
            st.session_state.race = (race_level.name, results, order, guess, bonus, settings)
            st.session_state.run_id += 1
    else:
        st.write("Pick at least 2 optimizers.")

    race = st.session_state.race
    if race is not None and race[0] == race_level.name:
        _, results, order, guess, bonus, used = race
        # draw in the order the player picked them, so the legend doesn't give away the winner
        show_animation(run_figure(race_level, [(n, r.path, r.losses) for n, r in results.items()]),
                       st.session_state.run_id, wait=started)
        if bonus:
            st.success(f"🔮 You predicted **{guess}** correctly! +{bonus} points")
        else:
            st.error(f"🔮 You predicted **{guess}**, but **{order[0]}** won.")
        rows = []
        for place, n in enumerate(order, 1):
            r, (lr, decay, params) = results[n], used[n]
            extra = ", ".join(f"{k}={v}" for k, v in params.items())
            rows.append({"Place": place, "Optimizer": n,
                         "Settings": f"lr={lr:g}" + (f", decay={decay}" if decay < 1 else "") + (f", {extra}" if extra else ""),
                         "Outcome": f"{EMOJI[r.outcome]} {r.outcome}", "Steps": r.steps,
                         "Final loss": f"{r.losses[-1]:.3g}"})
        st.table(rows)
        if race_level.data is not None:
            st.plotly_chart(fit_figure(race_level, [(n, results[n].path) for n in order]))

with tab_theory:
    st.subheader("Update rules (same as torch.optim)")
    st.markdown("""
`g` = gradient, `p` = position (the weights), all operations element-wise.

| Optimizer | Update | Key idea |
|---|---|---|
| SGD | p = p − lr·g | Step straight downhill |
| Momentum | v = μ·v + g ; p = p − lr·v | Heavy ball: zig-zags cancel, steady directions speed up |
| Nesterov | v = μ·v + g ; p = p − lr·(g + μ·v) | Looks ahead, overshoots less |
| Adagrad | s = s + g² ; p = p − lr·g/(√s + ε) | Own lr per weight, but it keeps shrinking |
| RMSprop | s = α·s + (1−α)·g² ; p = p − lr·g/(√s + ε) | Adagrad that forgets old gradients |
| Adam | m = β₁m + (1−β₁)g ; v = β₂v + (1−β₂)g² ; m̂ = m/(1−β₁ᵗ) ; v̂ = v/(1−β₂ᵗ) ; p = p − lr·m̂/(√v̂ + ε) | Momentum + RMSprop + bias correction |

Defaults: μ = 0.9, α = 0.99, β₁ = 0.9, β₂ = 0.999, ε = 1e-8 (1e-10 for Adagrad).
LR decay: lr_t = lr·γᵗ (same as `torch.optim.lr_scheduler.ExponentialLR`).
""")

    st.subheader("When does SGD fail?")
    st.markdown("""
| Level | Failure | Why SGD fails | Fix |
|---|---|---|---|
| 1. Bowl | lr too big | Each step multiplies p by (1 − 2·lr); if that is bigger than 1 in size it explodes | lr < 2/λ_max |
| 2. Ravine | Ill-conditioning | Steep direction forces a tiny lr, flat direction then crawls | Momentum, Adam, normalise inputs |
| 3. Saddle | Saddle point | Gradient ≈ 0 on the ridge, so steps are tiny | Adam / RMSprop (divide by gradient size) |
| 4. Twin Valleys | Local minimum | Stops in the first valley | Momentum (rolls over the hill, not guaranteed) |
| 5. Storm | Mini-batch noise | Constant lr keeps jittering around the minimum | LR decay |
| 6. Rosenbrock | Curved narrow valley | Tiny lr needed, then very slow along the curve | Tuned Momentum or Adam |
| 7. Wine (raw) | Real-data ravine (κ = 189) | Unnormalised feature: slope is far more sensitive than intercept | Momentum, Adam — or normalise |
| 8. Wine (normalised) | Real mini-batch noise | Small batches give noisy gradients | Bigger batch or LR decay |
""")

    st.subheader("Key facts for the viva")
    st.markdown("""
- **Why only 2D?** With 2 weights we can draw the whole loss surface. The optimizer maths is identical to a network with millions of weights.
- **Stability rule:** for a quadratic, SGD is stable only if lr < 2/λ_max (λ_max = largest Hessian eigenvalue).
- **Bias correction:** m and v start at 0, so early on they are too small. Dividing by (1 − βᵗ) fixes this; the factor goes to 1 as t grows.
  Without it (RMSprop) the first step is about 10× lr — on the Saddle level RMSprop jumps to the goal in 1 step.
- **Adam is not always best:** with a constant lr it fails on the Bowl and Twin Valleys at every lr we tried.
- **Verified:** `python3 test_verify.py` checks our gradients against autograd and our optimizers against `torch.optim`.
""")

    st.subheader("Why these numbers?")
    for lv in LEVELS:
        with st.expander(lv.name):
            st.write(lv.why_numbers)

# drawn last so it already includes a score from this click
st.sidebar.metric("Total score", st.session_state.score)
for name, pts in st.session_state.best.items():
    st.sidebar.write(f"{name}: {pts} pts")
