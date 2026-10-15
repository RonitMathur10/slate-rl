import json
from pathlib import Path

import torch

from slate_rl.simulator.model import UserChoiceModel
from slate_rl.simulator.simulator import SimParams, UserSimulator


def save_choice_model(
    model: UserChoiceModel, out_dir: str | Path, sim_ref: float, metrics: dict | None = None
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config, "sim_ref": sim_ref},
        out_dir / "choice_model.pt",
    )
    if metrics is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_choice_model(
    out_dir: str | Path, device: str | torch.device = "cpu"
) -> tuple[UserChoiceModel, float]:
    """Returns (model, calibrated sim_ref)."""
    ckpt = torch.load(Path(out_dir) / "choice_model.pt", map_location=device)
    model = UserChoiceModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["sim_ref"]


def load_simulator(
    out_dir: str | Path, params: dict, device: str | torch.device = "cpu", seed: int = 0
) -> UserSimulator:
    """Build a simulator from a saved choice model; ``sim_ref`` comes from the checkpoint."""
    model, sim_ref = load_choice_model(out_dir, device)
    sim_params = SimParams.from_dict({**params, "sim_ref": sim_ref})
    return UserSimulator(model, sim_params, device=device, seed=seed)
