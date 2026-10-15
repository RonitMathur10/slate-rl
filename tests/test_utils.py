import torch

from slate_rl.utils import get_device, seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(7)
    a = torch.rand(3)
    seed_everything(7)
    assert torch.equal(a, torch.rand(3))


def test_get_device_explicit():
    assert get_device("cpu") == torch.device("cpu")
    assert isinstance(get_device(), torch.device)
