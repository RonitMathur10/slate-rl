import pytest
import torch
from fastapi.testclient import TestClient

from slate_rl.agents.baselines import make_baselines
from slate_rl.envs import EnvAssets, EnvConfig
from slate_rl.serving.api import create_app
from slate_rl.serving.service import Demographics, RecommenderService
from slate_rl.simulator import SimParams

K, SLATE = 6, 3


@pytest.fixture
def service(parts, processed):
    feats, seqs, cand_gen, model = parts
    titles = [""] + [f"Movie {i}" for i in range(1, processed.meta["n_items"] + 1)]
    assets = EnvAssets(seqs, cand_gen, feats, model, SimParams(max_steps=5, sim_ref=0.1), titles)
    movie_ids = [0] * len(titles)
    for idx, movie in processed.items.select("item_idx", "movie_id").iter_rows():
        movie_ids[idx] = movie
    svc = RecommenderService(
        assets, EnvConfig(n_candidates=K, slate_size=SLATE), {}, movie_ids, "cpu"
    )
    svc.policies = make_baselines(svc._template, None)
    return svc


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def test_recommend_returns_unseen_distinct_items(service):
    history = [100, 101, 102]
    slate = service.recommend("retrieval", history)
    assert [s["rank"] for s in slate] == [1, 2, 3]
    ids = [s["movie_id"] for s in slate]
    assert len(set(ids)) == SLATE and not set(ids) & set(history)
    assert all(s["title"].startswith("Movie") for s in slate)


def test_recommend_marks_shown_items_for_the_policy(service):
    first = service.recommend("retrieval", [100, 101])
    again = service.recommend(
        "retrieval", [100, 101], shown_movie_ids=[s["movie_id"] for s in first], step=1
    )
    assert [s["movie_id"] for s in again] == [s["movie_id"] for s in first]  # retrieval ignores it


def test_recommend_errors(service):
    with pytest.raises(KeyError):
        service.recommend("no_such_policy", [100])
    with pytest.raises(KeyError):
        service.recommend("retrieval", [999_999])


def test_demographics_tensor():
    assert Demographics("F", 35, 7).tensor("cpu").tolist() == [[1, 3, 7]]
    assert Demographics("M", 99, 50).tensor("cpu").tolist() == [[0, 2, 20]]


def test_simulate_returns_a_full_session(service):
    user = int(service.assets.seqs.heldout_users()[0])
    out = service.simulate("retrieval", user_idx=user, seed=0)
    assert out["user_idx"] == user and out["history"]
    assert 1 <= len(out["steps"]) <= 5
    assert out["total_reward"] == pytest.approx(sum(s["reward"] for s in out["steps"]), abs=1e-3)
    assert all(len(s["slate"]) == SLATE for s in out["steps"])
    with pytest.raises(KeyError):
        service.simulate("retrieval", user_idx=int(service.assets.seqs.model_users()[0]))


def test_api_endpoints(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "retrieval" in client.get("/policies").json()["policies"]
    r = client.post("/recommend", json={"policy": "retrieval", "history_movie_ids": [100, 101]})
    assert r.status_code == 200 and len(r.json()["slate"]) == SLATE
    r = client.post(
        "/recommend",
        json={
            "policy": "retrieval",
            "history_movie_ids": [100],
            "demographics": {"gender": "F", "age": 35, "occupation": 3},
        },
    )
    assert r.status_code == 200
    assert (
        client.post("/recommend", json={"policy": "nope", "history_movie_ids": [100]}).status_code
        == 404
    )
    assert (
        client.post(
            "/recommend", json={"policy": "retrieval", "history_movie_ids": [1]}
        ).status_code
        == 404
    )
    assert client.post("/recommend", json={"history_movie_ids": []}).status_code == 422
    r = client.post("/simulate", json={"policy": "retrieval", "seed": 1})
    assert r.status_code == 200 and r.json()["steps"]


def test_recommend_with_learned_agent(service):
    from slate_rl.agents.dqn import DQNAgent, DQNConfig

    service.policies["dqn"] = DQNAgent.for_env(service._template, DQNConfig(hidden=16))
    slate = service.recommend("dqn", [100, 101, 102])
    assert len({s["movie_id"] for s in slate}) == SLATE
    assert torch.is_tensor(service._template.item_static)
