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

> Status: work in progress. Milestones 1–2 (data pipeline, two-tower retrieval) are done.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a W&B login (`uvx wandb login`).

```bash
make install   # uv sync + pre-commit hooks
make data      # download MovieLens-1M, index, split -> data/processed/ml-1m
make retrieval # train the two-tower model + FAISS index -> artifacts/retrieval/ml-1m
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

## Candidate retrieval (two-tower)

Each step, the agent chooses from K=100 candidates proposed by a two-tower model.
- **User tower:** encodes recent history plus demographics, with **no user-ID embedding**. This
  lets it work for held-out users and inside simulated sessions.
- **Item tower:** encodes item ID, genres and release year.
- **Training:** in-batch sampled softmax with logQ correction and duplicate-target masking.
- **Search:** exact inner-product search with FAISS.

Results on MovieLens-1M. Recall and NDCG count all of a user's next ratings in the split, with
already-seen items excluded:

| Model | Split | Recall@10 | Recall@100 | NDCG@10 | NDCG@100 |
|---|---|---|---|---|---|
| Popularity | test | 0.038 | 0.218 | 0.064 | 0.116 |
| Two-tower | test | 0.180 | 0.551 | 0.223 | 0.336 |
| Two-tower | held-out users, test | 0.182 | 0.545 | 0.219 | 0.330 |

Held-out users perform about as well as training users, so the history-based user tower
generalises to users it never saw.

## Layout

```
configs/    Hydra configs (data/, wandb/, ...)
slate_rl/   data/, retrieval/, utils/ (simulator/, envs/, agents/, eval/ coming)
scripts/    entry points
tests/
```
