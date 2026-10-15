"""Download and parse the raw MovieLens-1M files."""

import io
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def download_ml1m(raw_dir: str | Path) -> Path:
    """Download and extract ML-1M into ``raw_dir/ml-1m`` (no-op if already present)."""
    raw_dir = Path(raw_dir)
    target = raw_dir / "ml-1m"
    if (target / "ratings.dat").exists():
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ML1M_URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(raw_dir)
    return target


def _read_dat(path: Path, columns: list[str]) -> pl.DataFrame:
    # The ``::`` separator is multi-byte (unsupported by polars' CSV reader) and titles
    # can contain ``:``, so split lines in Python. ML-1M is latin-1 encoded.
    lines = path.read_text(encoding="latin-1").splitlines()
    rows = [line.split("::") for line in lines if line]
    return pl.DataFrame(rows, schema=columns, orient="row")


def load_ratings(ml_dir: str | Path) -> pl.DataFrame:
    df = _read_dat(Path(ml_dir) / "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
    return df.select(pl.all().cast(pl.Int64))


def load_users(ml_dir: str | Path) -> pl.DataFrame:
    df = _read_dat(Path(ml_dir) / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    return df.with_columns(pl.col("user_id", "age", "occupation").cast(pl.Int64))


def load_movies(ml_dir: str | Path) -> pl.DataFrame:
    df = _read_dat(Path(ml_dir) / "movies.dat", ["movie_id", "raw_title", "genre_str"])
    return df.select(
        pl.col("movie_id").cast(pl.Int64),
        pl.col("raw_title").str.replace(r"\s*\(\d{4}\)\s*$", "").alias("title"),
        pl.col("raw_title").str.extract(r"\((\d{4})\)\s*$", 1).cast(pl.Int64).alias("year"),
        pl.col("genre_str").str.split("|").alias("genres"),
    )
