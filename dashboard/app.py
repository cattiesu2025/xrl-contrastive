"""
Streamlit dashboard for interactive contrastive explanations.

Launch:  streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go

from agents.configs import PROFILES, REWARD_CHANNELS
from agents.train import load_agent, evaluate_agent, greedy_trajectory
from agents.decomposed_dqn import DecomposedDQN
from xai.reward_decomp import decompose_state, trajectory_decomposition, differentiating_channel
from xai.contrastive import explain_agent_decision
from envs.multi_obj_grid import MultiObjGridEnv

CHECKPOINT_DIR = "checkpoints"


# ---- Helpers ----

@st.cache_resource
def load_all_agents():
    agents = {}
    for key, profile in PROFILES.items():
        path = os.path.join(CHECKPOINT_DIR, f"{key}.pt")
        if os.path.exists(path):
            model, ckpt = load_agent(path)
            agents[key] = {
                "model": model,
                "weights": profile["weights"],
                "name": profile["name"],
                "color": profile["color"],
                "returns": ckpt.get("episode_returns", []),
            }
    return agents


def draw_grid(env, path=None, title="", ax=None):
    """Draw grid world on matplotlib axis."""
    G = env.GRID_SIZE
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(0, G)
    ax.set_ylim(0, G)
    ax.set_xticks(range(G + 1))
    ax.set_yticks(range(G + 1))
    ax.grid(True, color="grey", linewidth=0.3)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")

    for r, c in env.HAZARDS:
        ax.add_patch(plt.Rectangle((c, G - 1 - r), 1, 1, fc="#e74c3c", ec="grey", lw=0.3))
        ax.text(c + 0.5, G - 0.5 - r, "H", ha="center", va="center", fontsize=8, color="white")

    for idx, (r, c) in enumerate(env.RESOURCES):
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
                ax.annotate("", xy=(c + 0.5, G - 0.5 - r),
                            xytext=(pc + 0.5, G - 0.5 - pr),
                            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2))

    return ax


def agent_path_steps(model, weights, max_steps=200):
    """Walk the greedy trajectory and record, for each cell visited, the agent's
    REAL decision there: {(row,col): {collected, action, decomp}} (first visit).

    Using the agent's actual collected-coins state at each cell (not a fixed
    collected=0) keeps every shown decision in-distribution — the agent really
    experienced that state on its path."""
    env = MultiObjGridEnv(seed=0)
    obs = env.reset()
    steps = {}
    for _ in range(max_steps):
        pos = (env.row, env.col)
        d = decompose_state(model, obs, weights)
        if pos not in steps:
            steps[pos] = {"collected": env.collected, "action": d["greedy_action"], "decomp": d}
        obs, _, done, _ = env.step(d["greedy_action"])
        if done:
            break
    return steps


def build_clickable_grid(env, on_path):
    """Plotly grid: heatmap for cell colours + a transparent full-grid scatter that
    is clickable (on-path cells get a dark ring). Click returns (x=col, y=row)."""
    G = env.GRID_SIZE
    z = [[0] * G for _ in range(G)]
    for r, c in env.HAZARDS:
        z[r][c] = 1
    for r, c in env.RESOURCES:
        z[r][c] = 2
    z[env.START[0]][env.START[1]] = 3
    z[env.GOAL[0]][env.GOAL[1]] = 4
    colorscale = [
        [0.0, "#ecf0f1"], [0.2, "#ecf0f1"],
        [0.2, "#e74c3c"], [0.4, "#e74c3c"],
        [0.4, "#f1c40f"], [0.6, "#f1c40f"],
        [0.6, "#3498db"], [0.8, "#3498db"],
        [0.8, "#27ae60"], [1.0, "#27ae60"],
    ]
    fig = go.Figure(go.Heatmap(z=z, colorscale=colorscale, zmin=0, zmax=4,
                               showscale=False, xgap=1, ygap=1, hoverinfo="skip"))
    cells = [(r, c) for r in range(G) for c in range(G)]
    fig.add_trace(go.Scatter(
        x=[c for r, c in cells], y=[r for r, c in cells], mode="markers",
        marker=dict(size=22, color="rgba(0,0,0,0)",
                    line=dict(color="#2c3e50",
                              width=[2 if (r, c) in on_path else 0 for r, c in cells])),
        hovertemplate="row=%{y}, col=%{x}<extra></extra>", showlegend=False))
    fig.update_yaxes(autorange="reversed", title="row")
    fig.update_xaxes(title="col")
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ---- App ----

def main():
    st.set_page_config(page_title="XRL Contrastive Dashboard", layout="wide")
    st.title("XRL-Contrastive: Explainable Multi-Objective RL")

    agents = load_all_agents()
    if not agents:
        st.error("No trained agents found. Run `python -m experiments.run_all --phase train` first.")
        return

    tabs = st.tabs(["Trajectories", "Q-Decomposition", "Explore (click a cell)"])

    # ---- Tab 1: Trajectory comparison ----
    with tabs[0]:
        st.subheader("Greedy trajectory comparison")

        cols = st.columns(len(agents))
        for col, (key, agent) in zip(cols, agents.items()):
            with col:
                traj = greedy_trajectory(agent["model"], agent["weights"])
                env = MultiObjGridEnv()
                fig, ax = plt.subplots(figsize=(4, 4))
                draw_grid(env, path=traj["positions"], title=agent["name"], ax=ax)
                st.pyplot(fig)
                plt.close()

                metrics = evaluate_agent(agent["model"], agent["weights"], n_eval=50)
                st.metric("Success rate", f"{metrics['success_rate']:.0%}")
                st.metric("Avg R1", f"{metrics['avg_R1']:.2f}")
                st.metric("Avg R2", f"{metrics['avg_R2']:.2f}")

    # ---- Tab 2: Q-value decomposition ----
    with tabs[1]:
        st.subheader("Step-by-step Q-value decomposition")

        agent_key = st.selectbox("Select agent", list(agents.keys()),
                                 format_func=lambda k: agents[k]["name"])
        agent = agents[agent_key]

        steps = trajectory_decomposition(agent["model"], agent["weights"])
        step_idx = st.slider("Step", 0, max(len(steps) - 1, 1), 0) if len(steps) > 1 else 0

        step = steps[step_idx]
        st.write(f"**Position:** {step['position']}  →  "
                 f"**Action:** {step['action_names'][step['greedy_action']]}")

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        channel_names = REWARD_CHANNELS
        greedy = step["greedy_action"]
        vals = [step["greedy_channel_contributions"][ch] for ch in channel_names]
        colors = ["#27ae60", "#e74c3c", "#f1c40f", "#95a5a6"]
        bars = ax.barh(channel_names, vals, color=colors, height=0.6)
        ax.set_xlabel("Weighted Q contribution")
        ax.axvline(0, color="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=9)
        st.pyplot(fig)
        plt.close()

    # ---- Tab 3: Interactive explore ----
    with tabs[2]:
        st.subheader("Click a cell on the agents' paths to compare their decisions")
        st.caption("Each agent's decision is shown at the state it ACTUALLY reaches that cell in "
                   "— with the coins it has collected so far on its own greedy path — so every "
                   "row is a real, in-distribution decision. Ringed cells lie on ≥1 agent's path.")
        env = MultiObjGridEnv()
        paths = {k: agent_path_steps(ag["model"], ag["weights"]) for k, ag in agents.items()}
        on_path = set().union(*[set(p) for p in paths.values()])
        event = st.plotly_chart(build_clickable_grid(env, on_path),
                                on_select="rerun", selection_mode="points", key="explore_grid")

        pts = []
        if event and getattr(event, "selection", None):
            pts = event.selection.get("points", []) if isinstance(event.selection, dict) \
                else getattr(event.selection, "points", [])
        if not pts:
            st.info("Click any ringed cell (a cell on at least one agent's greedy path).")
        else:
            col, row = int(round(pts[0]["x"])), int(round(pts[0]["y"]))
            pos = (row, col)
            st.markdown(f"**Cell ({row}, {col})**")
            rows, sentences, visiting = [], [], []
            for k, ag in agents.items():
                step = paths[k].get(pos)
                if step is None:
                    rows.append({"agent": ag["name"], "visits?": "no", "coins so far": "—",
                                 "action": "—", "driver": "—",
                                 **{f"Q_{ch}": "—" for ch in REWARD_CHANNELS}})
                else:
                    decomp = step["decomp"]
                    contribs = decomp["greedy_channel_contributions"]
                    driver, _ = differentiating_channel(decomp)
                    rows.append({
                        "agent": ag["name"], "visits?": "yes",
                        "coins so far": bin(step["collected"]).count("1"),
                        "action": MultiObjGridEnv.ACTION_NAMES[step["action"]],
                        "driver": driver,
                        **{f"Q_{ch}": round(contribs[ch], 2) for ch in REWARD_CHANNELS},
                    })
                    sentences.append(explain_agent_decision(ag["name"], pos, decomp))
                    visiting.append((ag["name"], MultiObjGridEnv.ACTION_NAMES[step["action"]], driver))
            st.dataframe(pd.DataFrame(rows), hide_index=True)

            # Natural-language explanation (decisive channel = what tips each choice)
            st.markdown("**Why each agent chose its action** (driver = the channel that, "
                        "if removed, would flip the decision):")
            for s in sentences:
                st.markdown(f"- {s}")

            acts = {a for _, a, _ in visiting}
            if len(visiting) <= 1:
                st.caption("Only one agent passes through this cell on its path.")
            elif len(acts) == 1:
                st.caption(f"All visiting agents agree on **{acts.pop()}**.")
            else:
                drivers = ", ".join(f"{n} ({a}, driven by {d})" for n, a, d in visiting)
                st.caption(f"They diverge — {drivers}.")


if __name__ == "__main__":
    main()
