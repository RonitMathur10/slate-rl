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

> Status: work in progress. Milestones 1–4 (data pipeline, two-tower retrieval, user simulator, RL environment) are done.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a W&B login (`uvx wandb login`).

```bash
make install   # uv sync + pre-commit hooks
make data      # download MovieLens-1M, index, split -> data/processed/ml-1m
make retrieval # train the two-tower model + FAISS index -> artifacts/retrieval/ml-1m
make simulator # train the simulator's choice model -> artifacts/simulator/ml-1m
make validate  # roll reference policies through the simulator -> reports/simulator/
make check-env # Gymnasium API check, an example episode, env throughput
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

## User simulator

Each simulated user has two parts:
- **Learned preferences:** a SASRec-style transformer reads the user's history. It is trained on
  MovieLens with a full-catalogue softmax (a conditional-logit choice model) plus a rating head.
- **Hand-set session dynamics:** a choice among the slate's items plus a *no-click* option,
  position bias, a **fatigue** state, a **satisfaction** state and a probability of leaving.
  - Fatigue rises when the user consumes items similar to what they just watched.
  - Satisfaction is a moving average of watch-through.
  - The leave probability goes up with fatigue and down with satisfaction.

Some design choices:
- **Comparable utilities:** utilities are expressed as *log-lift over uniform*,
  `log P(i) + log N`. Because a conditional logit has the IIA property, restricting choice to
  the slate stays consistent with the model fit on the whole catalogue.
- **Calibrated fatigue reference:** the similarity level at which fatigue neither rises nor falls
  is set from real data (the mean similarity between consecutive real consumptions).
- **Separate models:** the simulator has its own embeddings, separate from the retrieval model the
  agent uses.

The full report is in [reports/simulator/report.md](reports/simulator/report.md). Results on
held-out users:

| Policy | Session length | Watch / session | CTR at step 0 |
|---|---|---|---|
| Random (from candidates) | 10.4 | 1.88 | 0.27 |
| Retrieval greedy | 10.8 | 3.52 | 0.61 |
| Oracle greedy (simulator affinity) | 11.6 | 4.70 | **0.73** |
| Oracle novelty-aware | **13.6** | **5.73** | 0.72 |

Greedy wins on immediate CTR but loses on session length (−15%) and total watch (−18%) to a
policy that steers away from what the user just consumed. This is the long-term trade-off
the RL agents have to discover, and they must do it *without* access to the simulator's
affinity. The `oracle_*` rows exist only as probes.

**Limitations:**
- No impressions or watch time: everything except the learned affinity and rating model is
  hand-set.
- The rating head is compressed towards the mean (true 1★ → 2.7, 5★ → 3.9).
- Simulated consumption is slightly more popular and less genre-diverse than real consumption.

## Environment

`BatchedSlateEnv` steps N simulated users in parallel on the GPU and auto-resets finished
sessions. `SlateRecEnv` is a single-user Gymnasium wrapper; it passes
`gymnasium.utils.env_checker` and is used for the viewer and serving.

- **Observation:** only what a real recommender could know.
  - the user's recent history and their retrieval user embedding
  - K=100 candidates, each with its retrieval embedding, retrieval score, popularity, genres
    and similarity to the user's recent items
  - the recommender's own session log: progress, click-rate EMA, last click, last watch,
    no-click streak

  Simulator internals such as fatigue and satisfaction appear only in `info["sim"]`, for
  analysis.
- **Action:** k=5 distinct indices into the candidates. Duplicate indices are replaced by the
  lowest unused candidate and flagged in `info["action_fixed"]`.
- **Reward:** `click + watch + 0.1·ILD(slate) + 0.1·novelty(clicked) − 0.2·repeat`.
  - ILD is genre diversity within the shown slate.
  - Novelty counts only for *consumed* items, so showing obscure items nobody clicks earns
    nothing.
  - `repeat` is the share of the slate already shown earlier in the session.
  - Each component is reported separately in `info["reward_components"]`.
- **Episode end:** a user leaving is `terminated`; hitting the 50-step cap is `truncated`.
  `info["final_obs"]` supports bootstrapping on truncation.
- **Throughput:** about 22k user-steps/s on an M3 (MPS), and about 9k on CPU.

## Layout

```
configs/    Hydra configs (data/, wandb/, ...)
slate_rl/   data/, retrieval/, simulator/, envs/, agents/, eval/, utils/
reports/    generated validation reports
scripts/    entry points
tests/
```
