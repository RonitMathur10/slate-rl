"""FastAPI app around ``RecommenderService``.

Run locally:  uv run --group serve uvicorn slate_rl.serving.api:app --port 8000
Environment:  SLATE_RL_DEVICE (default cpu), SLATE_RL_AGENT_SEED (default 0).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from slate_rl.serving.service import Demographics, RecommenderService


class DemographicsIn(BaseModel):
    gender: str = Field("M", pattern="^[MF]$")
    age: int = Field(25, description="ML-1M age code: 1, 18, 25, 35, 45, 50 or 56")
    occupation: int = Field(0, ge=0, le=20)


class RecommendRequest(BaseModel):
    policy: str = "slateq"
    history_movie_ids: list[int] = Field(
        ..., min_length=1, description="MovieLens movie ids the user watched, oldest first"
    )
    shown_movie_ids: list[int] = Field(
        default_factory=list, description="items already shown earlier in this session"
    )
    step: int = Field(0, ge=0, description="steps elapsed in the current session")
    demographics: DemographicsIn | None = None


class SimulateRequest(BaseModel):
    policy: str = "slateq"
    user_idx: int | None = Field(None, description="a held-out user; omit to pick one by seed")
    seed: int = 0


def build_service() -> RecommenderService:
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="evaluate")
    return RecommenderService.from_config(
        cfg,
        device=os.environ.get("SLATE_RL_DEVICE", "cpu"),
        agent_seed=int(os.environ.get("SLATE_RL_AGENT_SEED", "0")),
    )


def create_app(service: RecommenderService | None = None) -> FastAPI:
    """With ``service=None`` the real artifacts are loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_service()
        yield

    app = FastAPI(
        title="slate-rl",
        description="Slate recommendations from RL agents trained in a MovieLens user simulator.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/policies")
    def policies() -> dict:
        return {"policies": sorted(app.state.service.policies)}

    @app.post("/recommend")
    def recommend(req: RecommendRequest) -> dict:
        demo = Demographics(**req.demographics.model_dump()) if req.demographics else None
        try:
            slate = app.state.service.recommend(
                req.policy, req.history_movie_ids, req.shown_movie_ids, demo, req.step
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e).strip("'\"")) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return {"policy": req.policy, "slate": slate}

    @app.post("/simulate")
    def simulate(req: SimulateRequest) -> dict:
        try:
            return app.state.service.simulate(req.policy, req.user_idx, req.seed)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e).strip("'\"")) from e

    return app


app = create_app()
