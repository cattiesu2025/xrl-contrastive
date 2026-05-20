# XRL-Contrastive: Concept-Guided Contrastive Explanations for Multi-Objective Deep RL

A research project exploring explainable reinforcement learning through TCAV concept probing
and reward decomposition, enabling contrastive comparisons between agents with different
value priorities.

## Quick start

```bash
pip install torch numpy matplotlib seaborn streamlit scikit-learn tqdm
cd xrl-contrastive

# 1. Train three agents (~10 min each)
python -m experiments.run_all --phase train

# 2. Run XAI analysis (~1 min)
python -m experiments.run_all --phase xai

# 3. Launch dashboard
streamlit run dashboard/app.py
```

## Project structure

```
xrl-contrastive/
├── envs/
│   ├── multi_obj_grid.py      # Multi-objective GridWorld (adapted from COMP9414 A2)
│   └── reward_wrapper.py      # Gymnasium-compatible wrapper with 4-channel rewards
├── agents/
│   ├── decomposed_dqn.py      # Multi-head DQN with per-channel Q-heads
│   ├── replay_buffer.py       # Experience replay
│   ├── train.py               # Training loop
│   └── configs.py             # Agent profiles (safety / speed / balanced)
├── xai/
│   ├── tcav.py                # TCAV concept activation vector analysis
│   ├── reward_decomp.py       # Q-value decomposition utilities
│   └── contrastive.py         # Contrastive explanation generation
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
├── experiments/
│   ├── run_all.py             # End-to-end pipeline
│   └── evaluate.py            # Quantitative metrics
└── README.md
```

## Relation to COMP9414 Assignment 2

This project extends the multi-objective RL environment from the assignment with:

- **Tabular → Neural**: Q-table replaced by multi-head DQN (PyTorch) to enable gradient-based XAI
- **2-channel → 4-channel rewards**: goal / safety / coin / step separated for finer decomposition
- **New XAI layer**: TCAV concept probing + reward decomposition + contrastive explanations
- **Interactive dashboard**: Streamlit app for comparing agent decisions side-by-side
