# Per-Channel TCAV (Decomposed-Q Signed Sensitivity) — Design

Date: 2026-05-20

## Motivation

The current TCAV (`run_tcav_analysis`) probes the **total weighted Q** of the
greedy action. Because the four reward channels (goal, safety, coin, step) are
mixed before the gradient is taken, the resulting scores do not map cleanly to
agent personality:

- β (speed-first) shows the *lowest* `near_goal` sensitivity (0.28).
- α (safety-first) shows `near_hazard` = 0.78 (>0.5), literally reading as
  "hazard proximity increases Q" — backwards for a hazard-averse agent.
- γ (balanced) shows `near_hazard` = 0.12.

Root causes: (1) mixing 4 channels in one gradient is semantically muddy;
(2) `near_hazard` CAV separability is low (~0.6); (3) the sign of "concept
positively influences Q" is ambiguous for an aversive concept.

The project's core asset — **decomposed per-channel Q-heads** — is unused by the
current TCAV. Probing each head separately is the natural fix and a genuine
TCAV × reward-decomposition contribution.

## Metric: signed sensitivity (cosine)

For each (agent, channel `c`, concept):

**REVISION (during implementation):** the cosine-mean below was implemented and
empirically discarded. In the 64-dim feature space `g` and the CAV are nearly
orthogonal, so per-state cosines are tiny and their mean washes out to ~0 (all
agents/pairs landed in [-0.09, +0.13] — no usable signal). The **sign of the
directional derivative** is the stable signal — exactly why standard TCAV counts
a sign fraction. Final metric used:

```
for each concept state s:
    features = encoder(s)
    g        = ∂( max_a Q_channel(s, a) ) / ∂features   # channel's OWN best value
    d(s)     = sign( g · CAV )                            # ∈ {-1, 0, +1}
sensitivity  = mean_s d(s)                                # signed sign-fraction ∈ [-1, 1]
```

We use `max_a Q_channel` (not the total-Q greedy action) so the score is
independent of the policy / which action the agent happens to take.

Reading: `+0.9` = moving toward the concept consistently raises this channel's
value; `-0.85` = consistently lowers it; `~0` = no consistent coupling. Diverging
colormap: blue(+) ↔ white(0) ↔ red(−).

**Reliability caveat (key finding):** signs are only trustworthy where the CAV
separates the concept. `near_goal` CAV acc ~0.95 (reliable); `near_hazard` ~0.62
(barely above chance — its signs are NOT interpretable). After dropping naive
"expected sign" checks, the one robust, reliable contrastive finding is that
**β (speed-first)'s goal head negatively couples to near_goal (−0.68) while
α/γ are positive** — surprising and worth a focused note. Conclusions are drawn
only on high-CAV-accuracy cells; low-accuracy cells are flagged, not explained.

## Scope (focused diagonal)

Main result = 3 matched pairs × 3 agents, one heatmap:

|            | Q_safety×near_hazard | Q_goal×near_goal | Q_coin×near_resource |
|------------|----------------------|------------------|----------------------|
| α safety   | ?                    | ?                | ?                    |
| β speed    | ?                    | ?                | ?                    |
| γ balanced | ?                    | ?                | ?                    |

- **Q_step excluded** — step penalty is spatially uniform, no matching concept.
- **Total-Q TCAV kept** as a documented "naive baseline (mixed channels →
  uninterpretable)"; dashboard shows both, baseline first, per-channel as the
  main result.
- CAV reused from existing `train_cav`; **CAV_accuracy displayed** per cell so
  low-separability concepts (e.g. near_hazard ~0.6) are flagged honestly.

## Code changes (in an isolated worktree)

1. `xai/tcav.py` — add (existing functions untouched):
   - `channel_directional_sensitivity(model, channel_idx, concept_states, cav, weights)` → mean cosine ∈ [-1,1]
   - `run_per_channel_tcav(model, weights)` → 3 matched pairs + CAV_acc, structured dict
2. `experiments/run_all.py` — XAI phase prints per-channel results; save under a
   new `per_channel` field in `results/tcav_results.json` (existing structure intact).
3. `dashboard/app.py` — TCAV panel: add a second heatmap (diverging colormap),
   titles distinguishing baseline vs per-channel, show CAV_acc.

## Verification (done criteria)

- XAI phase runs without error; per-channel heatmap renders.
- **Sign sanity**: α's `Q_safety×near_hazard` clearly negative (hazard lowers
  safety value); all agents' `Q_goal×near_goal` positive (all reach goal). If
  signs are absurd, the metric or CAV is wrong — investigate, don't paper over.
- **Interpretability**: per-channel aligns with personality better than total-Q.
  If it does not, that is a real finding — record honestly, do not beautify.

## Non-goals (YAGNI)

- No full 4×4 matrix, no Q_step row, no agent retraining (reuse existing 100%
  checkpoints).
