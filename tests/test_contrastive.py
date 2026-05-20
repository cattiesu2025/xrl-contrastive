"""Tests for trajectory-anchored contrastive explanations."""

from agents.train import load_agent
from agents.configs import PROFILES
from envs.multi_obj_grid import MultiObjGridEnv
from xai.contrastive import (
    find_trajectory_disagreements,
    _greedy_path_states,
    generate_explanation,
)


def _reachable_set(pairs):
    env = MultiObjGridEnv()
    reachable = set()
    for model, weights in pairs:
        for _, pos, collected in _greedy_path_states(model, weights):
            reachable.add(env.encode(pos[0], pos[1], collected))
    return reachable


def test_disagreements_are_reachable_and_off_terminal():
    """Every contrastive candidate must be a state at least one agent actually
    reaches on its greedy path (in-distribution), and never a hazard/goal cell."""
    ma, ca = load_agent("checkpoints/safety_first.pt")
    mb, cb = load_agent("checkpoints/speed_first.pt")
    pairs = [(ma, ca["weights"]), (mb, cb["weights"])]
    dis = find_trajectory_disagreements(ma, ca["weights"], mb, cb["weights"])
    assert len(dis) > 0
    env = MultiObjGridEnv()
    reachable = _reachable_set(pairs)
    for d in dis:
        assert d["position"] not in env.HAZARDS
        assert d["position"] != env.GOAL
        enc = env.encode(d["position"][0], d["position"][1], d["collected"])
        assert enc in reachable, f"{d['position']} collected={d['collected']} not reachable"


def test_same_dominant_channel_explanation_is_not_redundant():
    """When both agents share the dominant channel, the contrast must describe a
    route difference, not the vacuous 'A driven by X while B driven by X'."""
    ma, ca = load_agent("checkpoints/safety_first.pt")
    mb, cb = load_agent("checkpoints/speed_first.pt")
    dis = find_trajectory_disagreements(ma, ca["weights"], mb, cb["weights"])
    texts = [generate_explanation(d, "A", "B") for d in dis]
    # at least one disagreement has both agents goal-driven (they all navigate to goal)
    shared = [t for t in texts if "Both prioritise" in t]
    assert shared, "expected at least one shared-objective contrast"
    for t in texts:
        assert "driven by goal, while B is driven by goal" not in t
