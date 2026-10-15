"""Uncertainty estimates for policy comparisons."""

import torch


def bootstrap_ci(
    values: torch.Tensor, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap (1 - alpha) CI of ``values`` [n]."""
    values = values.double().cpu()
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(values), (n_boot, len(values)), generator=gen)
    means = values[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64))
    return values.mean().item(), lo.item(), hi.item()


def paired_difference(
    a: torch.Tensor, b: torch.Tensor, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a - b over the same users with a 95% bootstrap CI (users resampled jointly)."""
    return bootstrap_ci(a - b, n_boot=n_boot, seed=seed)


# Two-sided 97.5% Student-t quantiles for small numbers of seeds (df = n - 1).
_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def seed_ci(values: list[float] | torch.Tensor) -> tuple[float, float, float]:
    """Mean and 95% t-interval across independent training seeds (captures training variance).

    With a single seed the interval collapses to the mean.
    """
    v = torch.as_tensor(values, dtype=torch.float64)
    mean = v.mean().item()
    if len(v) < 2:
        return mean, mean, mean
    half = _T975.get(len(v) - 1, 1.96) * (v.std(unbiased=True) / len(v) ** 0.5).item()
    return mean, mean - half, mean + half
