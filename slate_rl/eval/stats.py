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
