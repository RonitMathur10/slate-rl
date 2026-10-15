"""Dense tensor features for items and users, indexed by item_idx / user_idx."""

from dataclasses import dataclass

import torch

from slate_rl.data.preprocess import ProcessedData

AGE_BUCKETS = [1, 18, 25, 35, 45, 50, 56]  # ML-1M age codes
N_OCCUPATIONS = 21
YEAR_MIN, YEAR_BUCKET_SIZE, N_YEAR_BUCKETS = 1915, 5, 18


@dataclass
class ItemFeatures:
    genres: torch.Tensor  # [n_items + 1, n_genres] multi-hot; row 0 is padding
    year_bucket: torch.Tensor  # [n_items + 1] long
    popularity: torch.Tensor  # [n_items + 1] model-group train counts

    @classmethod
    def from_processed(cls, data: ProcessedData) -> "ItemFeatures":
        n = data.meta["n_items"] + 1
        genres = torch.zeros(n, len(data.meta["genres"]))
        year_bucket = torch.zeros(n, dtype=torch.long)
        popularity = torch.zeros(n)
        items = data.items
        for idx, gids, year, count in zip(
            items["item_idx"], items["genre_ids"], items["year"], items["train_count"], strict=True
        ):
            genres[idx, gids] = 1.0
            bucket = ((year or YEAR_MIN) - YEAR_MIN) // YEAR_BUCKET_SIZE
            year_bucket[idx] = min(max(bucket, 0), N_YEAR_BUCKETS - 1)
            popularity[idx] = count
        return cls(genres, year_bucket, popularity)

    def to(self, device) -> "ItemFeatures":
        return ItemFeatures(*(t.to(device) for t in (self.genres, self.year_bucket, self.popularity)))


@dataclass
class UserFeatures:
    demo: torch.Tensor  # [n_users, 3] long: (gender, age bucket, occupation)

    @classmethod
    def from_processed(cls, data: ProcessedData) -> "UserFeatures":
        users = data.users.sort("user_idx")
        age_index = {a: i for i, a in enumerate(AGE_BUCKETS)}
        gender = [int(g == "F") for g in users["gender"]]
        age = [age_index.get(a, 0) for a in users["age"]]
        occupation = [min(max(o, 0), N_OCCUPATIONS - 1) for o in users["occupation"]]
        return cls(torch.tensor([gender, age, occupation], dtype=torch.long).T.contiguous())

    def to(self, device) -> "UserFeatures":
        return UserFeatures(self.demo.to(device))
