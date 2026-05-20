"""Characterization tests for per-channel (decomposed-Q) TCAV.

These assert what is RELIABLE (high CAV separability) and explicitly document
what is NOT (near_hazard barely separable), rather than forcing naive expected
signs that the learned representations do not actually obey.
"""

from agents.train import load_agent
from xai.tcav import run_per_channel_tcav


def test_structure_and_ranges():
    model, _ = load_agent("checkpoints/balanced.pt")
    res = run_per_channel_tcav(model)
    assert set(res.keys()) == {
        "safety_x_near_hazard", "goal_x_near_goal", "coin_x_near_resource"
    }
    for v in res.values():
        assert -1.0 <= v["sensitivity"] <= 1.0
        assert 0.0 <= v["cav_accuracy"] <= 1.0


def test_goal_concept_is_separable_and_balanced_couples_positively():
    """near_goal is cleanly separable (CAV acc high), and on that reliable column
    the balanced agent's goal head positively couples to goal proximity."""
    model, _ = load_agent("checkpoints/balanced.pt")
    res = run_per_channel_tcav(model)
    assert res["goal_x_near_goal"]["cav_accuracy"] > 0.85
    assert res["goal_x_near_goal"]["sensitivity"] > 0.0


def test_near_hazard_cav_is_poorly_separable():
    """Honesty marker: near_hazard barely separates in feature space (~0.6),
    so signs on safety_x_near_hazard must not be over-interpreted."""
    model, _ = load_agent("checkpoints/balanced.pt")
    res = run_per_channel_tcav(model)
    assert res["safety_x_near_hazard"]["cav_accuracy"] < 0.75
