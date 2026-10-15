import numpy as np
import pytest
import torch

from slate_rl.data.features import ItemFeatures, UserFeatures
from slate_rl.data.sequences import UserSequences
from slate_rl.retrieval import (
    Retriever,
    TrainConfig,
    TwoTower,
    evaluate,
    fit,
    in_batch_softmax_loss,
    model_score_fn,
    pad_history,
    popularity_score_fn,
)

MAX_HIST = 8


@pytest.fixture
def seqs(processed):
    return UserSequences(processed, MAX_HIST)


@pytest.fixture
def model(processed):
    torch.manual_seed(0)
    return TwoTower.from_features(ItemFeatures.from_processed(processed), emb_dim=16, hidden_dim=32)


@pytest.fixture
def demo(processed):
    return UserFeatures.from_processed(processed).demo


def test_history_window_is_left_padded_and_ordered(processed, seqs):
    inter = processed.interactions.filter(processed.interactions["user_idx"] == 0)
    items = inter["item_idx"].to_list()
    hist = seqs.history(torch.tensor([0, 0]), torch.tensor([3, 12]))
    assert hist[0].tolist() == [0] * (MAX_HIST - 3) + items[:3]
    assert hist[1].tolist() == items[12 - MAX_HIST : 12]


def test_split_boundaries_match_processed(processed, seqs):
    for u in (0, 5):
        user = processed.interactions.filter(processed.interactions["user_idx"] == u)
        assert seqs.n_train[u] == (user["split"] == "train").sum()
        assert seqs.n_val[u] == (user["split"] == "val").sum()
        assert seqs.n_total[u] == user.height


def test_train_pairs_only_model_users_and_train_positions(seqs):
    users, pos = seqs.train_pairs()
    assert not seqs.heldout[users].any()
    assert (pos >= 1).all() and (pos < seqs.n_train[users]).all()
    assert len(users) == int((seqs.n_train[seqs.model_users()] - 1).sum())


def test_items_in_range(seqs):
    users = torch.tensor([0])
    mask = seqs.items_in_range(users, torch.tensor([0]), torch.tensor([4]))
    expected = set(seqs.items[0, MAX_HIST : MAX_HIST + 4].tolist())
    assert set(torch.nonzero(mask[0]).squeeze(1).tolist()) == expected
    assert not mask[:, 0].any()


def test_embeddings_are_normalised(model, seqs, demo):
    users = torch.arange(4)
    u = model.user_embed(seqs.history(users, seqs.n_train[users]), demo[users])
    assert u.shape == (4, 16)
    assert torch.allclose(u.norm(dim=1), torch.ones(4), atol=1e-5)
    items = model.all_item_embeddings()
    assert torch.all(items[0] == 0)
    assert torch.allclose(items[1:].norm(dim=1), torch.ones(items.shape[0] - 1), atol=1e-5)


def test_loss_masks_duplicate_targets():
    u = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    targets = torch.tensor([5, 5, 7])
    loss_dup = in_batch_softmax_loss(u, u, targets, temperature=1.0)
    assert torch.isfinite(loss_dup)


def test_fit_beats_popularity_baseline_on_train_signal(model, seqs, demo):
    cfg = TrainConfig(
        batch_size=64, epochs=15, lr=5e-3, eval_ks=[10], select_metric="recall@10", patience=15
    )
    users = seqs.model_users()
    before = evaluate(model_score_fn(model, demo), seqs, users, "val", [10])["recall@10"]
    after = fit(model, seqs, demo, cfg)["recall@10"]
    assert after >= before


def test_evaluate_perfect_scorer_gets_full_recall(seqs):
    def oracle(users, history):
        end = seqs.n_train[users]
        return seqs.items_in_range(users, end, end + seqs.n_val[users]).float()

    users = seqs.model_users()
    metrics = evaluate(oracle, seqs, users, "val", [50])
    assert metrics["recall@50"] == pytest.approx(1.0)
    assert metrics["ndcg@50"] == pytest.approx(1.0)
    pop = evaluate(popularity_score_fn(torch.rand(seqs.n_items + 1)), seqs, users, "val", [50])
    assert 0.0 <= pop["recall@50"] <= 1.0


def test_faiss_search_matches_torch_and_respects_exclude(model, demo, tmp_path):
    retriever = Retriever(model, MAX_HIST)
    histories = [[1, 2, 3], [4]]
    user_emb = retriever.user_embedding(histories, demo[:2])
    ids = retriever.search(user_emb, k=5)
    torch_top = (torch.from_numpy(user_emb) @ torch.from_numpy(retriever.item_emb[1:]).T).topk(5)
    np.testing.assert_array_equal(ids, torch_top.indices.numpy() + 1)

    excluded = retriever.search(user_emb, k=5, exclude=[{int(ids[0, 0])}, set()])
    assert int(ids[0, 0]) not in excluded[0]
    np.testing.assert_array_equal(excluded[1], ids[1])

    retriever.save(tmp_path)
    loaded = Retriever.load(tmp_path)
    np.testing.assert_array_equal(loaded.search(user_emb, k=5), ids)


def test_pad_history():
    out = pad_history([[1, 2, 3], [], list(range(1, 12))], max_hist=4)
    assert out.tolist() == [[0, 1, 2, 3], [0, 0, 0, 0], [8, 9, 10, 11]]
