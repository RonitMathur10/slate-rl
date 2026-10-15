# slate-rl

Long-horizon **slate recommendation with deep RL**. Each turn, an agent recommends a slate of
items to a simulated user, and it is trained to maximise engagement over the whole session
(clicks, watch-through, diversity and novelty) instead of the next click.

The user simulator is learned from **MovieLens-1M**:
- a sequence encoder of the user's history
- a conditional-logit choice model that includes a no-click option
- latent fatigue and satisfaction states that decide when the user leaves the session

Agents include greedy-click, item-wise DQN, SlateQ and PPO with Wolpertinger-style
proto-actions.

> Status: work in progress. Milestone 1 (scaffold and data pipeline) is done.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a W&B login (`uvx wandb login`).

```bash
make install   # uv sync + pre-commit hooks
make data      # download MovieLens-1M, index, split -> data/processed/ml-1m
make test
```

Pass `wandb.mode=disabled` to any script to run without W&B logging.

## Data

- **Chronological per-user split:** the last 10% of each user's ratings is test, the 10% before
  that is validation, and the rest is train.
- **Held-out users:** 10% of users are kept out entirely. They are used only for the final RL
  evaluation.
- **No popularity leakage:** item popularity is computed from model-group training data only.
- **Index 0 is padding:** real items are indexed from 1.
- **Modelling caveat:** MovieLens has no impressions or watch time. Watch-through is modelled from
  ratings, and unclicked impressions are sampled negatives. Results describe behaviour inside the
  simulator, not real users.

## Layout

```
configs/    Hydra configs (data/, wandb/, ...)
slate_rl/   data/, utils/ (retrieval/, simulator/, envs/, agents/, eval/ coming)
scripts/    entry points
tests/
```
