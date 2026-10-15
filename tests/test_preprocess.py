import numpy as np
import polars as pl
import pytest

from slate_rl.data.preprocess import (
    PAD_ITEM,
    SplitConfig,
    load_processed,
    preprocess,
    save_processed,
)


@pytest.fixture
def raw_tables():
    rng = np.random.default_rng(0)
    rows = []
    for user in range(1, 41):
        n = 5 if user == 40 else int(rng.integers(20, 60))  # user 40 falls below the minimum
        movies = rng.choice(np.arange(100, 160), size=n, replace=False)
        times = np.sort(rng.integers(0, 10, size=n))  # many ties on purpose
        rows += [
            (user, int(m), int(rng.integers(1, 6)), int(t))
            for m, t in zip(movies, times, strict=True)
        ]
    ratings = pl.DataFrame(
        rows, schema=["user_id", "movie_id", "rating", "timestamp"], orient="row"
    )
    movies = pl.DataFrame(
        {
            "movie_id": list(range(100, 160)),
            "title": [f"Movie {i}" for i in range(60)],
            "year": [1990 + i % 10 for i in range(60)],
            "genres": [["Drama", "Comedy"] if i % 2 else ["Action"] for i in range(60)],
        }
    )
    users = pl.DataFrame(
        {
            "user_id": list(range(1, 41)),
            "gender": ["M", "F"] * 20,
            "age": [25] * 40,
            "occupation": [1] * 40,
            "zip": ["94720"] * 40,
        }
    )
    return ratings, movies, users


@pytest.fixture
def processed(raw_tables):
    return preprocess(*raw_tables, SplitConfig(heldout_user_frac=0.2, min_interactions=20))


def test_min_interaction_filter(processed):
    assert processed.users.height == 39
    assert 40 not in processed.users["user_id"].to_list()


def test_item_index_is_contiguous_and_skips_padding(processed):
    idx = processed.items["item_idx"].to_list()
    assert idx == list(range(1, len(idx) + 1))
    assert PAD_ITEM not in processed.interactions["item_idx"].to_list()


def test_split_is_chronological_per_user(processed):
    order = {"train": 0, "val": 1, "test": 2}
    df = processed.interactions.with_columns(
        pl.col("split").replace_strict(order, return_dtype=pl.Int64).alias("rank")
    )
    for _, user_df in df.group_by("user_idx"):
        user_df = user_df.sort("seq_pos")
        assert user_df["rank"].is_sorted(), "splits out of time order"
        assert user_df["timestamp"].is_sorted()
        assert {"train", "val", "test"} <= set(user_df["split"].to_list())


def test_heldout_users_are_disjoint_and_sized(processed):
    groups = processed.users.group_by("group").len()
    counts = dict(groups.iter_rows())
    assert counts["heldout"] == round(39 * 0.2)
    per_user = processed.interactions.group_by("user_idx").agg(pl.col("group").n_unique())
    assert per_user["group"].max() == 1


def test_popularity_uses_only_model_train_data(processed):
    expected = processed.interactions.filter(
        (pl.col("group") == "model") & (pl.col("split") == "train")
    ).height
    assert processed.items["train_count"].sum() == expected


def test_genre_ids(processed):
    row = processed.items.filter(pl.col("movie_id") == 101).row(0, named=True)
    assert row["genres"] == ["Drama", "Comedy"]
    assert row["genre_ids"] == [7, 4]


def test_split_is_deterministic(raw_tables):
    cfg = SplitConfig(heldout_user_frac=0.2, min_interactions=20, seed=3)
    a, b = preprocess(*raw_tables, cfg), preprocess(*raw_tables, cfg)
    assert a.interactions.equals(b.interactions)
    assert a.users.equals(b.users)


def test_save_load_roundtrip(processed, tmp_path):
    save_processed(processed, tmp_path)
    loaded = load_processed(tmp_path)
    assert loaded.interactions.equals(processed.interactions)
    assert loaded.items.equals(processed.items)
    assert loaded.meta["n_items"] == processed.meta["n_items"]
