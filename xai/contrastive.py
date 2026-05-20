"""
Contrastive explanations: compare two agents at the same state and
generate human-readable explanations for their different decisions.

This directly follows the approach in Lee & Cruz (2025).
"""

import numpy as np
from agents.decomposed_dqn import DecomposedDQN
from agents.configs import REWARD_CHANNELS
from envs.multi_obj_grid import MultiObjGridEnv
from xai.reward_decomp import decompose_state


def find_disagreement_states(
    model_a: DecomposedDQN,
    weights_a: list[float],
    model_b: DecomposedDQN,
    weights_b: list[float],
    n_samples: int = 500,
    seed: int = 42,
) -> list[dict]:
    """
    Find states where two agents choose different actions.

    Returns list of dicts with:
        state, position, action_a, action_b, decomp_a, decomp_b
    """
    rng = np.random.RandomState(seed)
    env = MultiObjGridEnv()
    disagreements = []

    for _ in range(n_samples):
        row = rng.randint(0, env.GRID_SIZE)
        col = rng.randint(0, env.GRID_SIZE)
        collected = rng.randint(0, 256)

        # Skip hazard and goal cells
        if (row, col) in env.HAZARDS or (row, col) == env.GOAL:
            continue

        state = env.features_from_pos(row, col, collected)

        decomp_a = decompose_state(model_a, state, weights_a)
        decomp_b = decompose_state(model_b, state, weights_b)

        if decomp_a["greedy_action"] != decomp_b["greedy_action"]:
            disagreements.append({
                "state": state,
                "position": (row, col),
                "collected": collected,
                "action_a": decomp_a["greedy_action"],
                "action_b": decomp_b["greedy_action"],
                "decomp_a": decomp_a,
                "decomp_b": decomp_b,
            })

    return disagreements


def _cell_label(row: int, col: int, env: MultiObjGridEnv) -> str:
    """Return a short human-readable label for a grid cell."""
    pos = (row, col)
    if not (0 <= row < env.GRID_SIZE and 0 <= col < env.GRID_SIZE):
        return "wall"
    if pos == env.GOAL:
        return "GOAL"
    if pos in env.HAZARDS:
        return "HAZARD"
    if pos in env.RESOURCES:
        return "resource"
    return "open"


def _direction_context(action: int, row: int, col: int, env: MultiObjGridEnv) -> str:
    """Return '→ (r,c): LABEL' for the cell an action leads to."""
    dr, dc = env.ACTION_DELTAS[action]
    tr, tc = row + dr, col + dc
    label = _cell_label(tr, tc, env)
    return f"→ ({tr},{tc}): {label}"


# Maps dominant channel → plain-English reason phrase (direction inserted at {dir})
_CHANNEL_REASON = {
    "goal":   "heading {dir} brings it closer to the goal",
    "safety": "moving {dir} better avoids hazard zones",
    "coin":   "moving {dir} leads to a nearby resource",
    "step":   "moving {dir} minimises step cost",
}

# Contrastive sentence templates keyed by (dominant_a, dominant_b)
_CONTRAST_TEMPLATES = {
    ("safety", "goal"):
        "{a} prioritises safety over speed, while {b} accepts the hazard risk "
        "to approach the goal faster.",
    ("goal", "safety"):
        "{a} pushes toward the goal despite the risk, while {b} takes the "
        "safer detour.",
    ("safety", "coin"):
        "{a} avoids the hazard, while {b} detours to collect a nearby resource.",
    ("coin", "safety"):
        "{a} detours for a resource, while {b} stays on the safer path.",
    ("goal", "coin"):
        "{a} heads straight for the goal, while {b} detours to collect a resource.",
    ("coin", "goal"):
        "{a} detours for a resource, while {b} heads straight for the goal.",
}


def _dominant_channel(decomp: dict) -> tuple[str, float]:
    """Return (channel_name, percentage) for the largest contributor."""
    contribs = decomp["greedy_channel_contributions"]
    q_total = decomp["q_total"][decomp["greedy_action"]]
    dominant = max(contribs, key=lambda k: contribs[k])
    pct = (contribs[dominant] / q_total * 100) if abs(q_total) > 1e-6 else 0.0
    return dominant, pct


def _agent_sentence(
    name: str,
    action: int,
    row: int,
    col: int,
    decomp: dict,
    env: MultiObjGridEnv,
) -> str:
    """One self-contained sentence explaining a single agent's decision."""
    act_name = MultiObjGridEnv.ACTION_NAMES[action].lower()
    ctx = _direction_context(action, row, col, env)
    q_total = decomp["q_total"][action]
    dominant, pct = _dominant_channel(decomp)
    reason = _CHANNEL_REASON[dominant].format(dir=act_name)
    return (
        f"{name} chose {act_name.upper()} [{ctx}] "
        f"(Q_total = {q_total:+.2f}), "
        f"driven mainly by {dominant} ({pct:.0f}%) — {reason}."
    )


def generate_explanation(
    disagreement: dict,
    name_a: str = "Agent α",
    name_b: str = "Agent β",
) -> str:
    """
    Generate a natural-language contrastive explanation for a disagreement.

    Structure:
      1. Header — position
      2. One sentence per agent explaining its own decision (dominant channel + %)
      3. One contrastive sentence comparing the two decisions
    """
    row, col = disagreement["position"]
    env = MultiObjGridEnv()

    decomp_a = disagreement["decomp_a"]
    decomp_b = disagreement["decomp_b"]
    dominant_a, _ = _dominant_channel(decomp_a)
    dominant_b, _ = _dominant_channel(decomp_b)

    sent_a = _agent_sentence(name_a, disagreement["action_a"], row, col, decomp_a, env)
    sent_b = _agent_sentence(name_b, disagreement["action_b"], row, col, decomp_b, env)

    contrast = _CONTRAST_TEMPLATES.get(
        (dominant_a, dominant_b),
        f"{name_a} is driven by {dominant_a}, while {name_b} is driven by {dominant_b}.",
    ).format(a=name_a, b=name_b)

    return "\n".join([
        f"At position ({row},{col}):",
        f"  {sent_a}",
        f"  {sent_b}",
        f"  → {contrast}",
    ])


def contrastive_report(
    model_a: DecomposedDQN,
    weights_a: list[float],
    name_a: str,
    model_b: DecomposedDQN,
    weights_b: list[float],
    name_b: str,
    n_examples: int = 5,
    seed: int = 42,
) -> str:
    """Generate a full contrastive report between two agents."""
    disagreements = find_disagreement_states(
        model_a, weights_a, model_b, weights_b, seed=seed)

    if not disagreements:
        return f"No disagreements found between {name_a} and {name_b}."

    lines = [
        f"Contrastive Report: {name_a} vs {name_b}",
        f"Found {len(disagreements)} disagreement states.",
        "=" * 60,
        "",
    ]

    for i, d in enumerate(disagreements[:n_examples]):
        lines.append(f"--- Example {i+1} ---")
        lines.append(generate_explanation(d, name_a, name_b))
        lines.append("")

    return "\n".join(lines)


# ---------- TCAV-augmented contrastive ----------

def tcav_contrastive_summary(
    tcav_results_a: dict,
    tcav_results_b: dict,
    name_a: str = "Agent α",
    name_b: str = "Agent β",
) -> str:
    """
    Compare TCAV scores across two agents to show which concepts
    each agent is more sensitive to.
    """
    lines = [
        f"TCAV Concept Sensitivity Comparison",
        f"{'Concept':<16} {name_a:>12} {name_b:>12} {'Δ':>8}  Interpretation",
        "-" * 70,
    ]

    all_concepts = set(tcav_results_a.keys()) | set(tcav_results_b.keys())

    for concept in sorted(all_concepts):
        score_a = tcav_results_a.get(concept, {}).get("tcav_score", float("nan"))
        score_b = tcav_results_b.get(concept, {}).get("tcav_score", float("nan"))
        delta = score_a - score_b

        if abs(delta) < 0.05:
            interp = "Similar sensitivity"
        elif delta > 0:
            interp = f"{name_a} more sensitive"
        else:
            interp = f"{name_b} more sensitive"

        lines.append(
            f"{concept:<16} {score_a:>12.3f} {score_b:>12.3f} {delta:>+8.3f}  {interp}"
        )

    return "\n".join(lines)
