"""Turn raw MovieLens tables into indexed, split interaction data.

Output layout (all parquet, plus ``meta.json``):
- ``interactions``: user_idx, item_idx, rating, timestamp, seq_pos, split, group
- ``items``: item_idx, movie_id, title, year, genres, genre_ids, train_count
- ``users``: user_idx, user_id, gender, age, occupation, group

Item index 0 is reserved for padding, so real items are 1..n_items. ``group`` is ``model``
(used to fit retrieval/simulator/agents) or ``heldout`` (reserved for final RL evaluation).
Within each user, ``split`` is chronological; ties in timestamp are broken by movie id.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from slate_rl.data.movielens import GENRES

PAD_ITEM = 0
GENRE_TO_ID = {g: i for i, g in enumerate(GENRES)}


@dataclass
class SplitConfig:
    val_frac: float = 0.1
    test_frac: float = 0.1
    heldout_user_frac: float = 0.1
    min_interactions: int = 20
    seed: int = 0


@dataclass
class ProcessedData:
    interactions: pl.DataFrame
    items: pl.DataFrame
    users: pl.DataFrame
    meta: dict


def assign_user_groups(user_ids: pl.Series, heldout_frac: float, seed: int) -> pl.DataFrame:
    """Randomly reserve ``heldout_frac`` of users for final evaluation."""
    ids = np.sort(user_ids.unique().to_numpy())
    rng = np.random.default_rng(seed)
    heldout = set(rng.choice(ids, size=round(len(ids) * heldout_frac), replace=False).tolist())
    groups = ["heldout" if u in heldout else "model" for u in ids]
    return pl.DataFrame({"user_id": ids, "group": groups})


def chronological_split(ratings: pl.DataFrame, val_frac: float, test_frac: float) -> pl.DataFrame:
    """Label each user's ratings train/val/test by time order (at least one val and test each)."""
    n = pl.len().over("user_id")
    n_test = pl.max_horizontal(pl.lit(1), (n * test_frac).round()).cast(pl.Int64)
    n_val = pl.max_horizontal(pl.lit(1), (n * val_frac).round()).cast(pl.Int64)
    df = ratings.sort(["user_id", "timestamp", "movie_id"]).with_columns(
        pl.int_range(pl.len()).over("user_id").alias("seq_pos"),
        n_test.alias("_n_test"),
        n_val.alias("_n_val"),
        n.alias("_n"),
    )
    split = (
        pl.when(pl.col("seq_pos") >= pl.col("_n") - pl.col("_n_test"))
        .then(pl.lit("test"))
        .when(pl.col("seq_pos") >= pl.col("_n") - pl.col("_n_test") - pl.col("_n_val"))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )
    return df.with_columns(split.alias("split")).drop("_n_test", "_n_val", "_n")


def preprocess(
    ratings: pl.DataFrame, movies: pl.DataFrame, users: pl.DataFrame, cfg: SplitConfig
) -> ProcessedData:
    counts = ratings.group_by("user_id").len()
    keep = counts.filter(pl.col("len") >= cfg.min_interactions)["user_id"]
    ratings = ratings.filter(pl.col("user_id").is_in(keep.implode()))

    groups = assign_user_groups(ratings["user_id"], cfg.heldout_user_frac, cfg.seed)
    user_index = (
        groups.sort("user_id")
        .with_row_index("user_idx")
        .with_columns(pl.col("user_idx").cast(pl.Int64))
    )
    item_index = (
        ratings.select(pl.col("movie_id").unique().sort())
        .with_row_index("item_idx", offset=PAD_ITEM + 1)
        .with_columns(pl.col("item_idx").cast(pl.Int64))
    )

    interactions = (
        chronological_split(ratings, cfg.val_frac, cfg.test_frac)
        .join(user_index, on="user_id")
        .join(item_index, on="movie_id")
        .select("user_idx", "item_idx", "rating", "timestamp", "seq_pos", "split", "group")
        .sort("user_idx", "seq_pos")
    )

    # Popularity is measured on model-group training data only, so it never leaks eval data.
    train_counts = (
        interactions.filter((pl.col("group") == "model") & (pl.col("split") == "train"))
        .group_by("item_idx")
        .len("train_count")
    )
    items = (
        item_index.join(movies, on="movie_id", how="left")
        .join(train_counts, on="item_idx", how="left")
        .with_columns(
            pl.col("train_count").fill_null(0).cast(pl.Int64),
            pl.col("genres")
            .list.eval(pl.element().replace_strict(GENRE_TO_ID, return_dtype=pl.Int64))
            .alias("genre_ids"),
        )
        .select("item_idx", "movie_id", "title", "year", "genres", "genre_ids", "train_count")
        .sort("item_idx")
    )
    users_out = (
        user_index.join(users, on="user_id", how="left")
        .select("user_idx", "user_id", "gender", "age", "occupation", "group")
        .sort("user_idx")
    )

    meta = {
        "n_users": users_out.height,
        "n_items": items.height,
        "pad_item": PAD_ITEM,
        "genres": GENRES,
        "n_interactions": interactions.height,
        "split_counts": dict(interactions.group_by("split").len().iter_rows()),
        "group_user_counts": dict(users_out.group_by("group").len().iter_rows()),
        "config": asdict(cfg),
    }
    return ProcessedData(interactions, items, users_out, meta)


def save_processed(data: ProcessedData, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.interactions.write_parquet(out_dir / "interactions.parquet")
    data.items.write_parquet(out_dir / "items.parquet")
    data.users.write_parquet(out_dir / "users.parquet")
    (out_dir / "meta.json").write_text(json.dumps(data.meta, indent=2))


def load_processed(out_dir: str | Path) -> ProcessedData:
    out_dir = Path(out_dir)
    return ProcessedData(
        interactions=pl.read_parquet(out_dir / "interactions.parquet"),
        items=pl.read_parquet(out_dir / "items.parquet"),
        users=pl.read_parquet(out_dir / "users.parquet"),
        meta=json.loads((out_dir / "meta.json").read_text()),
    )
