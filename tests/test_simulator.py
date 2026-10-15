import pytest
import torch

from slate_rl.agents import mmr_slate, random_slate, topk_slate
from slate_rl.data.features import ItemFeatures, UserFeatures
from slate_rl.data.sequences import UserSequences
from slate_rl.eval import genre_entropy, intra_list_diversity, novelty
from slate_rl.retrieval import CandidateGenerator, TwoTower
from slate_rl.simulator import (
    SimParams,
    SimTrainConfig,
    UserChoiceModel,
    UserSimulator,
    evaluate,
    fit,
    initial_state,
    load_simulator,
    save_choice_model,
)

HIST = 8
K = 4


@pytest.fixture
def seqs(processed):
    return UserSequences(processed, HIST)


@pytest.fixture
def choice_model(processed):
    torch.manual_seed(0)
    return UserChoiceModel(processed.meta["n_items"], max_hist=HIST, emb_dim=16, dropout=0.0)


@pytest.fixture
def sim(choice_model):
    return UserSimulator(choice_model, SimParams(max_steps=6, sim_ref=0.1), seed=0)


@pytest.fixture
def started(sim, seqs):
    users = torch.arange(6)
    sim.reset(*initial_state(seqs, users))
    return sim, users


def _unseen_slates(sim, k=K):
    scores = torch.rand(sim.seen.shape).masked_fill(sim.seen, -1.0)
    scores[:, 0] = -1.0
    return scores.topk(k, dim=1).indices


# ----- data plumbing -----------------------------------------------------------------------
def test_split_pairs_cover_val_positions(seqs):
    users, pos = seqs.split_pairs("val")
    assert (pos >= seqs.n_train[users]).all()
    assert (pos < seqs.n_train[users] + seqs.n_val[users]).all()
    assert len(users) == int(seqs.n_val[seqs.model_users()].sum())


# ----- choice model ------------------------------------------------------------------------
def test_choice_model_shapes_and_ranges(choice_model, seqs):
    users = torch.arange(3)
    hist = seqs.history(users, seqs.n_train[users])
    u = choice_model.encode(hist)
    logits = choice_model.logits(u)
    assert logits.shape == (3, seqs.n_items + 1)
    assert torch.isinf(logits[:, 0]).all()
    r = choice_model.predict_rating(u, torch.tensor([[1, 2], [3, 4], [5, 6]]))
    assert ((r >= 1) & (r <= 5)).all()


def test_empty_history_does_not_produce_nan(choice_model):
    u = choice_model.encode(torch.zeros(2, HIST, dtype=torch.long))
    assert torch.isfinite(u).all()


def test_fit_improves_validation_nll(choice_model, seqs):
    before = evaluate(choice_model, seqs, "val")["nll"]
    after = fit(choice_model, seqs, SimTrainConfig(batch_size=64, epochs=5, lr=3e-3))["nll"]
    assert after < before


# ----- simulator dynamics ------------------------------------------------------------------
def test_click_logits_and_probabilities(started):
    sim, _ = started
    logits, sim_mat = sim.click_logits(_unseen_slates(sim))
    assert logits.shape == (6, K + 1) and sim_mat.shape == (6, K)
    assert torch.allclose(logits[:, -1], torch.full((6,), sim.p.no_click_bias))
    assert torch.allclose(torch.softmax(logits, 1).sum(1), torch.ones(6))


def test_step_updates_state_consistently(started):
    sim, _ = started
    slates = _unseen_slates(sim)
    hist_before, fat_before = sim.history.clone(), sim.fatigue.clone()
    sim_before = sim.recent_similarity(slates)
    r = sim.step(slates)

    assert ((r.watch >= 0) & (r.watch <= 1)).all()
    assert (r.watch[~r.clicked] == 0).all() and (r.item[~r.clicked] == 0).all()
    assert (r.position[~r.clicked] == K).all()
    for b in range(6):
        if r.clicked[b]:
            assert sim.history[b, -1] == r.item[b]
            assert torch.equal(sim.history[b, :-1], hist_before[b, 1:])
            assert sim.seen[b, r.item[b]]
            s = sim_before[b, r.position[b]]
            expected = sim.p.fatigue_decay * fat_before[b] + sim.p.fatigue_lr * (s - sim.p.sim_ref)
            assert r.fatigue[b] == pytest.approx(
                expected.clamp(0, sim.p.fatigue_max).item(), abs=1e-5
            )
        else:
            assert torch.equal(sim.history[b], hist_before[b])
    assert (r.fatigue >= 0).all() and (sim.t == 1).all()


def test_seen_items_are_effectively_never_clicked(started):
    sim, _ = started
    seen_items = torch.stack([torch.nonzero(row).squeeze(1)[:K] for row in sim.seen])
    logits, _ = sim.click_logits(seen_items)
    probs = torch.softmax(logits, 1)
    assert (probs[:, :K].sum(1) < 1e-3).all()


def test_sessions_end_by_max_steps(started):
    sim, _ = started
    done = torch.zeros(6, dtype=torch.bool)
    for _ in range(sim.p.max_steps):
        done |= sim.step(_unseen_slates(sim)).done
    assert done.all()


def test_same_seed_same_trajectory(choice_model, seqs):
    users = torch.arange(6)
    outs = []
    for _ in range(2):
        s = UserSimulator(choice_model, SimParams(sim_ref=0.1), seed=123)
        s.reset(*initial_state(seqs, users))
        torch.manual_seed(0)
        slates = _unseen_slates(s)
        outs.append([s.step(slates).position for _ in range(3)])
    for a, b in zip(*outs, strict=True):
        assert torch.equal(a, b)


def test_partial_reset_only_touches_given_rows(started, seqs):
    sim, users = started
    sim.step(_unseen_slates(sim))
    before = sim.state_dict()
    idx = torch.tensor([1, 4])
    sim.reset(*initial_state(seqs, users[idx]), idx=idx)
    assert (sim.t[idx] == 0).all() and (sim.fatigue[idx] == 0).all()
    keep = torch.tensor([0, 2, 3, 5])
    assert torch.equal(sim.t[keep], before["t"][keep])
    assert torch.equal(sim.history[keep], before["history"][keep])


def test_save_and_load_simulator(choice_model, tmp_path):
    save_choice_model(choice_model, tmp_path, sim_ref=0.123)
    loaded = load_simulator(tmp_path, {"no_click_bias": 1.5})
    assert loaded.p.sim_ref == pytest.approx(0.123)
    assert loaded.p.no_click_bias == 1.5
    for a, b in zip(loaded.model.parameters(), choice_model.parameters(), strict=True):
        assert torch.equal(a, b)


# ----- slates, candidates, metrics ---------------------------------------------------------
def test_slate_builders():
    cands = torch.tensor([[10, 11, 12, 13]])
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.3]])
    assert topk_slate(cands, scores, 2).tolist() == [[11, 12]]
    vecs = torch.nn.functional.normalize(torch.randn(20, 4), dim=1)
    assert torch.equal(
        mmr_slate(cands, scores, vecs, 3, diversity=0.0), topk_slate(cands, scores, 3)
    )
    r = random_slate(cands, 3, torch.Generator().manual_seed(0))
    assert len(set(r[0].tolist())) == 3 and set(r[0].tolist()) <= set(cands[0].tolist())


def test_mmr_avoids_duplicates_of_the_same_direction():
    vecs = torch.zeros(5, 2)
    vecs[1] = vecs[2] = torch.tensor([1.0, 0.0])  # items 1 and 2 are identical
    vecs[3] = torch.tensor([0.0, 1.0])
    cands = torch.tensor([[1, 2, 3]])
    rel = torch.tensor([[1.0, 0.99, 0.5]])
    assert mmr_slate(cands, rel, vecs, 2, diversity=1.0).tolist() == [[1, 3]]


def test_candidate_generator_excludes_seen(processed, seqs):
    feats = ItemFeatures.from_processed(processed)
    gen = CandidateGenerator(
        TwoTower.from_features(feats, emb_dim=8, hidden_dim=16),
        UserFeatures.from_processed(processed).demo,
    )
    users = torch.arange(5)
    hist, seen = initial_state(seqs, users)
    cands, scores = gen(users, hist, seen, k=10)
    assert cands.shape == (5, 10)
    assert not seen.gather(1, cands).any() and (cands > 0).all()
    assert (scores[:, :-1] >= scores[:, 1:]).all()


def test_metrics():
    genres = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
    ild = intra_list_diversity(torch.tensor([[1, 2], [1, 3]]), genres)
    assert ild.tolist() == pytest.approx([0.0, 1.0])
    ent = genre_entropy(torch.tensor([[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))
    assert ent.tolist() == pytest.approx([1.0, 0.0])
    pop = torch.tensor([0.0, 100.0, 1.0])
    nov = novelty(torch.tensor([1, 2]), pop)
    assert nov[1] > nov[0]


def test_rollout_and_summary(processed, seqs, sim):
    from slate_rl.simulator.validation import make_policies, per_step_curves, rollout, summarize

    feats = ItemFeatures.from_processed(processed)
    gen = CandidateGenerator(
        TwoTower.from_features(feats, emb_dim=8, hidden_dim=16),
        UserFeatures.from_processed(processed).demo,
    )
    users = torch.arange(6)
    policies = make_policies(feats.popularity, mmr_diversity=1.0, novelty_weight=1.0)
    for policy in policies.values():
        ro = rollout(sim, gen, policy, seqs, users, feats.genres, n_candidates=10, slate_size=K)
        assert ro.alive[0].all()
        assert (ro.length >= 1).all() and (ro.length <= sim.p.max_steps).all()
        assert not (ro.clicked & ~ro.alive).any()
        s = summarize(ro, feats.genres, feats.popularity)
        assert 0.0 <= s["ctr"] <= 1.0 and s["session_length"] >= 1.0
        assert per_step_curves(ro)["survival"][0] == 1.0
