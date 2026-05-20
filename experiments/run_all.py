"""
End-to-end experiment pipeline.

Usage:
    python -m experiments.run_all --phase train
    python -m experiments.run_all --phase xai
    python -m experiments.run_all --phase all
"""

import argparse
import os
import json
import numpy as np

from agents.configs import PROFILES, REWARD_CHANNELS
from agents.train import train_agent, load_agent, evaluate_agent, greedy_trajectory
from xai.tcav import run_tcav_analysis
from xai.reward_decomp import trajectory_decomposition, format_decomposition
from xai.contrastive import contrastive_report, tcav_contrastive_summary


CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"


def phase_train():
    """Train all three agents."""
    print("=" * 60)
    print("PHASE 1: Training agents")
    print("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for profile_key, profile in PROFILES.items():
        print(f"\nTraining {profile['name']}...")
        print(f"  Weights: {profile['weights']}  (n_step={profile.get('n_step', 'default')})")
        agent_config = {"n_step": profile["n_step"]} if "n_step" in profile else None
        model, returns = train_agent(
            profile_name=profile_key,
            weights=profile["weights"],
            config=agent_config,
            save_dir=CHECKPOINT_DIR,
        )
        metrics = evaluate_agent(model, profile["weights"])
        print(f"  Success: {metrics['success_rate']:.0%}  "
              f"R1: {metrics['avg_R1']:.2f}  R2: {metrics['avg_R2']:.2f}  "
              f"Len: {metrics['avg_length']:.1f}  "
              f"Safety violations: {metrics['safety_violations']}")

    print("\nAll agents trained and saved.")


def phase_xai():
    """Run XAI analysis on trained agents."""
    print("=" * 60)
    print("PHASE 2: XAI Analysis")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load all agents
    agents = {}
    for profile_key, profile in PROFILES.items():
        path = os.path.join(CHECKPOINT_DIR, f"{profile_key}.pt")
        if not os.path.exists(path):
            print(f"  Skipping {profile_key}: checkpoint not found at {path}")
            continue
        model, ckpt = load_agent(path)
        agents[profile_key] = {
            "model": model,
            "weights": profile["weights"],
            "name": profile["name"],
        }

    if len(agents) < 2:
        print("Need at least 2 trained agents. Run --phase train first.")
        return

    # --- TCAV analysis ---
    print("\n--- TCAV Concept Analysis ---")
    tcav_results = {}
    for key, agent in agents.items():
        print(f"\n  Analysing {agent['name']}...")
        results = run_tcav_analysis(agent["model"], agent["weights"])
        tcav_results[key] = results
        for concept, info in results.items():
            print(f"    {concept:>15}: TCAV={info['tcav_score']:.3f}  "
                  f"CAV_acc={info['cav_accuracy']:.3f}")

    # --- Reward decomposition example ---
    print("\n--- Reward Decomposition (greedy trajectory) ---")
    for key, agent in agents.items():
        print(f"\n  {agent['name']}:")
        steps = trajectory_decomposition(agent["model"], agent["weights"])
        # Show first 3 steps
        for step in steps[:3]:
            print(format_decomposition(step))
            print()

    # --- Contrastive report ---
    print("\n--- Contrastive Explanations ---")
    keys = list(agents.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k_a, k_b = keys[i], keys[j]
            report = contrastive_report(
                agents[k_a]["model"], agents[k_a]["weights"], agents[k_a]["name"],
                agents[k_b]["model"], agents[k_b]["weights"], agents[k_b]["name"],
            )
            print(report)
            print()

    # --- TCAV contrastive ---
    print("\n--- TCAV Contrastive Summary ---")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k_a, k_b = keys[i], keys[j]
            if k_a in tcav_results and k_b in tcav_results:
                summary = tcav_contrastive_summary(
                    tcav_results[k_a], tcav_results[k_b],
                    agents[k_a]["name"], agents[k_b]["name"],
                )
                print(summary)
                print()

    # Save results
    serialisable = {}
    for key, results in tcav_results.items():
        serialisable[key] = {
            concept: {
                "tcav_score": info["tcav_score"],
                "cav_accuracy": info["cav_accuracy"],
            }
            for concept, info in results.items()
        }

    with open(os.path.join(RESULTS_DIR, "tcav_results.json"), "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="XRL-Contrastive experiments")
    parser.add_argument("--phase", choices=["train", "xai", "all"], default="all")
    args = parser.parse_args()

    if args.phase in ("train", "all"):
        phase_train()
    if args.phase in ("xai", "all"):
        phase_xai()


if __name__ == "__main__":
    main()
