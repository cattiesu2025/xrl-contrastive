# Per-Channel TCAV Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a decomposed-Q "signed sensitivity" TCAV that probes each reward-channel head separately, producing personality-aligned concept attributions, alongside the existing total-Q TCAV baseline.

**Architecture:** New functions in `xai/tcav.py` compute `cos(∂Q_channel/∂features, CAV)` averaged over concept states for 3 matched (channel, concept) pairs. Existing total-Q TCAV is untouched. `run_all.py` and the dashboard surface both. Reuse existing 100% checkpoints.

**Tech Stack:** PyTorch, scikit-learn (CAV logistic regression), numpy, pytest, Plotly/Streamlit (dashboard).

Reference design: `docs/plans/2026-05-20-per-channel-tcav-design.md`

---

### Task 1: `channel_directional_sensitivity`

**Files:**
- Modify: `xai/tcav.py` (add function, existing untouched)
- Test: `tests/test_per_channel_tcav.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_per_channel_tcav.py
import numpy as np
from agents.train import load_agent
from xai.tcav import collect_concept_states, collect_random_states, train_cav, channel_directional_sensitivity

def test_safety_head_drops_near_hazard():
    """α's Q_safety head should have NEGATIVE sensitivity to near_hazard:
    hazard reward is -10, so hazard proximity must lower the safety value."""
    model, ckpt = load_agent("checkpoints/safety_first.pt")
    concept = collect_concept_states("near_hazard", n_samples=120, seed=42)
    rand = collect_random_states(n_samples=120, seed=1042)
    cav, _ = train_cav(model, concept, rand)
    # channel idx 1 == "safety" in REWARD_CHANNELS
    sens = channel_directional_sensitivity(model, 1, concept[:50], cav, ckpt["weights"])
    assert -1.0 <= sens <= 1.0
    assert sens < 0.0, f"expected safety head to drop near hazards, got {sens:+.3f}"
```

**Step 2: Run test, verify it fails**

Run: `cd .worktrees/per-channel-tcav && python -m pytest tests/test_per_channel_tcav.py::test_safety_head_drops_near_hazard -v`
Expected: FAIL with `ImportError: cannot import name 'channel_directional_sensitivity'`

**Step 3: Implement**

Add to `xai/tcav.py`:

```python
def channel_directional_sensitivity(model, channel_idx, concept_states, cav, weights):
    """Mean cosine similarity between ∂Q_channel/∂features and the CAV, over
    concept states. Signed, scale-invariant, ∈ [-1, 1].
    +1 = concept strongly raises this channel's value; -1 = strongly lowers it.
    The greedy action is chosen by the TOTAL weighted Q (the policy's action),
    so we explain the channel's contribution to the value of the chosen action.
    """
    w = torch.tensor(weights, dtype=torch.float32)
    cav_t = torch.tensor(cav, dtype=torch.float32)
    cav_unit = cav_t / (cav_t.norm() + 1e-8)
    cosims = []
    for state in concept_states:
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        features = model.encoder(x)
        features.retain_grad()
        q_channels = [head(features) for head in model.heads]
        q_total = sum(wi * q for wi, q in zip(w, q_channels))
        greedy = int(q_total.argmax(dim=1).item())
        q_c = q_channels[channel_idx][0, greedy]
        model.zero_grad()
        q_c.backward()
        g = features.grad.squeeze(0)
        cos = float(torch.dot(g, cav_unit) / (g.norm() + 1e-8))
        cosims.append(cos)
    return float(np.mean(cosims))
```

**Step 4: Run test, verify it passes**

Run: `python -m pytest tests/test_per_channel_tcav.py::test_safety_head_drops_near_hazard -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xai/tcav.py tests/test_per_channel_tcav.py
git commit -m "feat: channel_directional_sensitivity (signed per-channel TCAV)"
```

---

### Task 2: `run_per_channel_tcav` (3 matched pairs)

**Files:**
- Modify: `xai/tcav.py`
- Test: `tests/test_per_channel_tcav.py`

**Step 1: Write the failing test**

```python
def test_run_per_channel_tcav_structure_and_signs():
    model, ckpt = load_agent("checkpoints/safety_first.pt")
    from xai.tcav import run_per_channel_tcav
    res = run_per_channel_tcav(model, ckpt["weights"], n_samples=120, seed=42)
    # 3 matched pairs present
    assert set(res.keys()) == {"safety_x_near_hazard", "goal_x_near_goal", "coin_x_near_resource"}
    for k, v in res.items():
        assert -1.0 <= v["sensitivity"] <= 1.0
        assert 0.0 <= v["cav_accuracy"] <= 1.0
    # sign sanity: goal head rises near goal (this agent reaches the goal)
    assert res["goal_x_near_goal"]["sensitivity"] > 0.0
```

**Step 2: Run, verify fail** — `ImportError: run_per_channel_tcav`

**Step 3: Implement** — add to `xai/tcav.py`:

```python
# (channel_name, concept_name) — semantically matched diagonal
MATCHED_PAIRS = [
    ("safety", "near_hazard"),
    ("goal", "near_goal"),
    ("coin", "near_resource"),
]

def run_per_channel_tcav(model, weights, n_samples=150, seed=42):
    """Signed sensitivity for each matched (channel, concept) pair.
    Returns {"<channel>_x_<concept>": {channel, concept, sensitivity, cav_accuracy}}.
    """
    from agents.configs import REWARD_CHANNELS
    random_states = collect_random_states(n_samples=n_samples, seed=seed + 1000)
    results = {}
    for ch_name, concept in MATCHED_PAIRS:
        ch_idx = REWARD_CHANNELS.index(ch_name)
        concept_states = collect_concept_states(concept, n_samples=n_samples, seed=seed)
        if len(concept_states) < 20:
            continue
        cav, cav_acc = train_cav(model, concept_states, random_states)
        sens = channel_directional_sensitivity(model, ch_idx, concept_states[:50], cav, weights)
        results[f"{ch_name}_x_{concept}"] = {
            "channel": ch_name,
            "concept": concept,
            "sensitivity": sens,
            "cav_accuracy": cav_acc,
        }
    return results
```

**Step 4: Run, verify pass.** Run full file: `python -m pytest tests/test_per_channel_tcav.py -v`

**Step 5: Commit**

```bash
git add xai/tcav.py tests/test_per_channel_tcav.py
git commit -m "feat: run_per_channel_tcav over 3 matched channel/concept pairs"
```

---

### Task 3: Wire into XAI phase + results JSON

**Files:**
- Modify: `experiments/run_all.py` (phase_xai)

**Step 1:** In `phase_xai`, after the existing TCAV loop, add per-channel computation + print, and add a `per_channel` field to the saved JSON. Concretely, after the `tcav_results` loop, add:

```python
    # --- Per-channel signed sensitivity (decomposed-Q TCAV) ---
    print("\n--- Per-Channel TCAV (signed sensitivity, decomposed Q-heads) ---")
    from xai.tcav import run_per_channel_tcav
    per_channel = {}
    for key, agent in agents.items():
        pc = run_per_channel_tcav(agent["model"], agent["weights"])
        per_channel[key] = pc
        print(f"\n  {agent['name']}:")
        for pair, info in pc.items():
            print(f"    {pair:24s}: sensitivity={info['sensitivity']:+.3f}  CAV_acc={info['cav_accuracy']:.3f}")
```

Then extend the serialisation block to include per-channel data under a new key (do not change the existing `serialisable` structure; add alongside):

```python
    serialisable_pc = {
        key: {pair: {"sensitivity": i["sensitivity"], "cav_accuracy": i["cav_accuracy"],
                     "channel": i["channel"], "concept": i["concept"]}
              for pair, i in pc.items()}
        for key, pc in per_channel.items()
    }
    with open(os.path.join(RESULTS_DIR, "tcav_results.json"), "w") as f:
        json.dump({"total_q": serialisable, "per_channel": serialisable_pc}, f, indent=2)
```

(Note: this changes the JSON top-level shape to `{"total_q": ..., "per_channel": ...}`. Update the dashboard loader in Task 4 accordingly.)

**Step 2: Verify** — run XAI phase:

Run: `python -m experiments.run_all --phase xai 2>&1 | sed -n '/Per-Channel TCAV/,$p' | head -30`
Expected: prints 3 pairs per agent; α `safety_x_near_hazard` negative, all `goal_x_near_goal` positive.

**Step 3: Commit**

```bash
git add experiments/run_all.py results/tcav_results.json
git commit -m "feat: surface per-channel TCAV in XAI phase + results JSON"
```

---

### Task 4: Dashboard per-channel heatmap

**Files:**
- Modify: `dashboard/app.py` (TCAV panel)

**Step 1:** Read the current TCAV panel in `dashboard/app.py` to find how the existing heatmap is built and how results are loaded. Add a SECOND heatmap below the existing one:
- Title: "Per-channel signed sensitivity (decomposed Q — main result)"; keep the existing one labelled "Total-Q TCAV (baseline, mixed channels)".
- Rows = agents, columns = the 3 matched pairs, z = sensitivity ∈ [-1, 1].
- Diverging colormap (e.g. `RdBu`), `zmid=0`, range [-1, 1].
- Annotate each cell with `sensitivity` and a small `(acc=…)` for CAV accuracy.

Use the same TCAV-compute trigger button; call `run_per_channel_tcav` for each agent (or load from `results/tcav_results.json` `per_channel`).

**Step 2: Verify** — launch dashboard, open TCAV tab, confirm both heatmaps render and the per-channel one uses a diverging colormap with α's safety×hazard cell red (negative). If the dashboard cannot be launched headless, at minimum run `python -c "import dashboard.app"` to confirm no import/syntax error, and note that UI was not visually verified.

**Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard per-channel signed-sensitivity heatmap"
```

---

### Task 5: End-to-end verification + honesty check

**Step 1:** Run the full XAI phase and capture the per-channel table for all 3 agents.

**Step 2: Sign-sanity assertions (write as a quick check script or extend the test):**
- α `safety_x_near_hazard` < 0 (and ideally the most negative across agents)
- all agents `goal_x_near_goal` > 0
- β `goal_x_near_goal` not lower than its `safety_x_near_hazard` magnitude (speed agent leans on goal)

**Step 3: Interpretability comparison.** Tabulate per-channel vs the old total-Q scores. Confirm per-channel maps to personality more cleanly. **If it does NOT, record that honestly in the design doc's verification section — do not beautify.**

**Step 4: Commit** any check script:

```bash
git add tests/ && git commit -m "test: end-to-end per-channel TCAV sign-sanity checks"
```

---

## Notes
- DRY: reuse `collect_concept_states`, `collect_random_states`, `train_cav` unchanged.
- YAGNI: 3 pairs only; no Q_step row; no 4×4 matrix; no retraining.
- Frequent commits: one per task.
- The existing total-Q TCAV is the documented baseline — never delete it.
