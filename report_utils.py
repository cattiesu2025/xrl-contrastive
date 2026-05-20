"""Plotting/analysis helpers for report.ipynb.

Self-contained matplotlib (no Streamlit). Loads the trained checkpoints in
checkpoints/ and renders the figures the report narrates. Keeping these here
keeps the notebook lean.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from agents.configs import PROFILES, REWARD_CHANNELS
from agents.train import load_agent, evaluate_agent, greedy_trajectory
from envs.multi_obj_grid import MultiObjGridEnv

CHECKPOINT_DIR = "checkpoints"
_CH_COLORS = {"goal": "#27ae60", "safety": "#e74c3c", "coin": "#f1c40f", "step": "#95a5a6"}


def load_agents():
    """Load all trained agents from checkpoints/. Returns dict keyed by profile."""
    agents = {}
    for key, profile in PROFILES.items():
        path = os.path.join(CHECKPOINT_DIR, f"{key}.pt")
        model, ckpt = load_agent(path)
        agents[key] = {
            "model": model,
            "weights": profile["weights"],
            "name": profile["name"],
            "n_step": profile.get("n_step"),
            "returns": ckpt.get("episode_returns", []),
            "best_episode": ckpt.get("best_episode"),
        }
    return agents


def draw_grid(env, path=None, title="", ax=None):
    """Draw the 10x10 grid (hazards, resources, start, goal) and an optional path."""
    G = env.GRID_SIZE
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, G); ax.set_ylim(0, G)
    ax.set_xticks(range(G + 1)); ax.set_yticks(range(G + 1))
    ax.grid(True, color="grey", linewidth=0.3); ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for r, c in env.HAZARDS:
        ax.add_patch(plt.Rectangle((c, G - 1 - r), 1, 1, fc="#e74c3c", ec="grey", lw=0.3))
        ax.text(c + 0.5, G - 0.5 - r, "H", ha="center", va="center", fontsize=8, color="white")
    for (r, c) in env.RESOURCES:
        ax.add_patch(plt.Rectangle((c, G - 1 - r), 1, 1, fc="#f1c40f", ec="grey", lw=0.3))
        ax.text(c + 0.5, G - 0.5 - r, "R", ha="center", va="center", fontsize=8)
    sr, sc = env.START
    ax.add_patch(plt.Rectangle((sc, G - 1 - sr), 1, 1, fc="#3498db", ec="grey", lw=0.3))
    ax.text(sc + 0.5, G - 0.5 - sr, "S", ha="center", va="center", fontsize=9, color="white")
    gr, gc = env.GOAL
    ax.add_patch(plt.Rectangle((gc, G - 1 - gr), 1, 1, fc="#27ae60", ec="grey", lw=0.3))
    ax.text(gc + 0.5, G - 0.5 - gr, "G", ha="center", va="center", fontsize=9, color="white")
    if path:
        for i, (r, c) in enumerate(path):
            ax.plot(c + 0.5, G - 0.5 - r, "o", color="#2c3e50", markersize=4, alpha=0.6)
            if i > 0:
                pr, pc = path[i - 1]
                ax.annotate("", xy=(c + 0.5, G - 0.5 - r), xytext=(pc + 0.5, G - 0.5 - pr),
                            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2))
    return ax


def plot_trajectories(agents):
    """Greedy trajectory of every agent, side by side."""
    fig, axes = plt.subplots(1, len(agents), figsize=(4 * len(agents), 4))
    if len(agents) == 1:
        axes = [axes]
    for ax, (key, ag) in zip(axes, agents.items()):
        traj = greedy_trajectory(ag["model"], ag["weights"])
        draw_grid(MultiObjGridEnv(), path=traj["positions"], title=ag["name"], ax=ax)
    fig.tight_layout()
    return fig


def plot_training_curves(agents, window=50):
    """Rolling-mean scalarised return per agent, with the best-model snapshot marked."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, ag in agents.items():
        r = np.asarray(ag["returns"], dtype=float)
        if len(r) >= window:
            rm = np.convolve(r, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(r)), rm, label=f"{ag['name']} (n={ag['n_step']})")
        if ag["best_episode"]:
            ax.axvline(ag["best_episode"], ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Episode"); ax.set_ylabel(f"Scalarised return (rolling {window})")
    ax.set_title("Training curves — dashed lines mark the saved best-model snapshot")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def metrics_table(agents, n_eval=100):
    """Return a list of metric dicts for a quick results table."""
    rows = []
    for key, ag in agents.items():
        m = evaluate_agent(ag["model"], ag["weights"], n_eval=n_eval)
        rows.append({
            "agent": ag["name"], "n_step": ag["n_step"],
            "success": m["success_rate"], "R1": m["avg_R1"],
            "R2": m["avg_R2"], "length": m["avg_length"],
            "safety_violations": m["safety_violations"],
        })
    return rows


def plot_reward_decomposition(decomp, title=""):
    """Horizontal bar of weighted channel contributions at one decomposed state."""
    fig, ax = plt.subplots(figsize=(7, 2.6))
    vals = [decomp["greedy_channel_contributions"][ch] for ch in REWARD_CHANNELS]
    ax.barh(REWARD_CHANNELS, vals, color=[_CH_COLORS[c] for c in REWARD_CHANNELS], height=0.6)
    ax.axvline(0, color="black", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.2f}", va="center", fontsize=9)
    ax.set_xlabel("Weighted Q contribution"); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_total_tcav(all_tcav, agents):
    """Heatmap of the baseline total-Q TCAV scores (sign-fraction, 0..1)."""
    import seaborn as sns
    concepts = sorted(set().union(*[r.keys() for r in all_tcav.values()]))
    keys = list(agents.keys())
    M = np.array([[all_tcav[k][c]["tcav_score"] for c in concepts] for k in keys])
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.heatmap(M, annot=True, fmt=".2f", xticklabels=concepts,
                yticklabels=[agents[k]["name"] for k in keys],
                cmap="RdYlGn", center=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_title("Baseline: total-Q TCAV (mixed channels)  — >0.5 = concept raises Q")
    fig.tight_layout()
    return fig


def plot_per_channel_tcav(per_channel, agents):
    """Diverging heatmap of per-channel signed sensitivity, annotated with CAV acc.

    per_channel: {agent_key: {pair: {sensitivity, cav_accuracy, channel, concept}}}
    """
    import seaborn as sns
    keys = list(agents.keys())
    pairs = list(next(iter(per_channel.values())).keys())
    sens = np.array([[per_channel[k][p]["sensitivity"] for p in pairs] for k in keys])
    annot = np.array([[f"{per_channel[k][p]['sensitivity']:+.2f}\n(acc {per_channel[k][p]['cav_accuracy']:.2f})"
                       for p in pairs] for k in keys])
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(sens, annot=annot, fmt="", xticklabels=pairs,
                yticklabels=[agents[k]["name"] for k in keys],
                cmap="RdBu", center=0.0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "signed sensitivity"})
    ax.set_title("Per-channel signed sensitivity (interpret only where CAV acc is high)")
    fig.tight_layout()
    return fig
