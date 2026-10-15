import numpy as np
import polars as pl
import pytest

from slate_rl.data.preprocess import SplitConfig, preprocess


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
