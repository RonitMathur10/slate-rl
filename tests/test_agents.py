import pytest
import torch

from slate_rl.agents.baselines import make_baselines
from slate_rl.agents.click_model import (
    ClickModel,
    collect_logs,
    evaluate_click_model,
    fit_click_model,
    load_click_model,
    save_click_model,
)
from slate_rl.agents.common import CompactObs, encoder_kwargs
from slate_rl.agents.dqn import DQNAgent, DQNConfig, train_dqn
from slate_rl.agents.replay import NStepAccumulator, ReplayBuffer
from slate_rl.eval.evaluate import evaluate_policy

from .test_env import SLATE, make_env


def fake_obs(n: int, value: float) -> CompactObs:
    return CompactObs(
        torch.full((n, 4), float(value)),
        torch.full((n, 6), float(value)),
        torch.full((n, 3), int(value), dtype=torch.long),
        torch.full((n, 3, 2), float(value)),
    )


# ----- n-step returns and replay ------------------------------------------------------------
def test_nstep_returns_handle_termination_and_truncation():
    acc = NStepAccumulator(n=3, gamma=0.5)
    rewards = [
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([2.0, 2.0, 2.0]),
        torch.tensor([4.0, 4.0, 4.0]),
    ]
    # env 0: no episode end; env 1: terminates at step 1; env 2: truncated at step 1
    term = [
        torch.tensor([False, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([False, False, False]),
    ]
    done = [
        torch.tensor([False, False, False]),
        torch.tensor([False, True, True]),
        torch.tensor([False, False, False]),
    ]
    out = None
    for t in range(3):
        out = acc.push(
            fake_obs(3, t),
            torch.zeros(3, 2, dtype=torch.long),
            rewards[t],
            term[t],
            done[t],
            fake_obs(3, 10 + t),
        )
        if t < 2:
            assert out is None
    obs0, _, ret, discount, next_obs = out
    assert torch.allclose(obs0.user_emb, torch.zeros(3, 4))
    assert ret.tolist() == pytest.approx([1 + 0.5 * 2 + 0.25 * 4, 1 + 0.5 * 2, 1 + 0.5 * 2])
    assert discount.tolist() == pytest.approx([0.125, 0.0, 0.25])
    assert next_obs.user_emb[:, 0].tolist() == pytest.approx([12.0, 11.0, 11.0])


def test_nstep_with_n1_is_one_step_td():
    acc = NStepAccumulator(n=1, gamma=0.9)
    out = acc.push(
        fake_obs(2, 0),
        torch.zeros(2, 2, dtype=torch.long),
        torch.tensor([1.0, 2.0]),
        torch.tensor([False, True]),
        torch.tensor([False, True]),
        fake_obs(2, 5),
    )
    _, _, ret, discount, _ = out
    assert ret.tolist() == [1.0, 2.0]
    assert discount.tolist() == pytest.approx([0.9, 0.0])


def test_replay_buffer_roundtrip_and_wraparound():
    buf = ReplayBuffer(capacity=5, example=fake_obs(1, 0), slate_size=2, device="cpu")
    for v in range(3):
        buf.add(
            fake_obs(2, v),
            torch.full((2, 2), v),
            torch.full((2,), float(v)),
            torch.full((2,), 0.5),
            fake_obs(2, v + 1),
        )
    assert buf.size == 5 and buf.ptr == 1
    # Slots 0..4 were written as v0 v0 v1 v1 v2, then v2's second row wrapped onto slot 0.
    assert buf.reward.tolist() == [2.0, 0.0, 1.0, 1.0, 2.0]
    obs, action, reward, discount, next_obs = buf.sample(64, torch.Generator().manual_seed(0))
    assert torch.allclose(next_obs.user_emb[:, 0], obs.user_emb[:, 0] + 1)
    assert (action[:, 0].float() == reward).all() and (discount == 0.5).all()


# ----- click model and baselines ------------------------------------------------------------
@pytest.fixture
def env(parts):
    return make_env(parts, num_envs=16)


def make_click_model(env):
    model = ClickModel(**encoder_kwargs(env), slate_size=env.k, hidden=32)
    model.encoder.item_static.copy_(env.item_static)
    return model


def test_click_model_learns_from_logs(env, tmp_path):
    logs = collect_logs(env, n_iters=60, seed=0)
    assert len(logs) == 60 * 16
    assert ((logs.position >= 0) & (logs.position <= env.k)).all()
    gen = torch.Generator().manual_seed(0)
    train, val = logs.split(0.2, gen)
    torch.manual_seed(0)
    model = make_click_model(env)
    before = evaluate_click_model(model, val)["nll"]
    after = fit_click_model(model, train, val, epochs=15, batch_size=128, lr=3e-3, gen=gen)["nll"]
    assert after < before
    save_click_model(model, tmp_path / "cm.pt")
    loaded = load_click_model(tmp_path / "cm.pt", "cpu")
    assert evaluate_click_model(loaded, val)["nll"] == pytest.approx(after, rel=1e-5)


def test_baselines_return_valid_slates(env):
    obs = env.reset(seed=0)
    for name, policy in make_baselines(env, make_click_model(env)).items():
        action = policy.act(obs)
        assert action.shape == (16, SLATE), name
        assert ((action >= 0) & (action < env.K)).all(), name
        for row in action:
            assert len(set(row.tolist())) == SLATE, name


def test_evaluate_policy_runs_one_session_per_user(parts):
    eval_env = make_env(parts, num_envs=8, autoreset=False)
    users = eval_env.user_pool[:8]
    policy = make_baselines(eval_env, None)["retrieval"]
    a = evaluate_policy(eval_env, policy, users, seed=0)
    b = evaluate_policy(eval_env, policy, users, seed=0)
    assert a["return"] == pytest.approx(b["return"]), "same users + seed must be reproducible"
    assert 1.0 <= a["session_length"] <= eval_env.max_steps
    assert a["per_user_return"].shape == (8,)
    assert 0.0 <= a["ctr"] <= 1.0
    with pytest.raises(ValueError):
        evaluate_policy(make_env(parts, num_envs=8, autoreset=True), policy, users)


# ----- DQN ----------------------------------------------------------------------------------
def test_dqn_trains_without_nans_and_checkpoints(parts, tmp_path):
    env = make_env(parts, num_envs=16, seed=0)
    eval_env = make_env(parts, num_envs=8, autoreset=False)
    cfg = DQNConfig(
        total_iters=40,
        buffer_size=2000,
        batch_size=32,
        learning_starts=64,
        updates_per_iter=1,
        hidden=32,
        eval_interval=20,
        log_interval=10,
    )
    agent = DQNAgent.for_env(env, cfg, seed=0)
    logged = []
    best = train_dqn(
        agent,
        env,
        cfg,
        evaluate=lambda a: evaluate_policy(eval_env, a, eval_env.user_pool[:8], seed=0),
        log=lambda m, step: logged.append(m),
        checkpoint=tmp_path / "dqn.pt",
    )
    losses = [m["train/loss"] for m in logged if "train/loss" in m]
    assert losses and all(torch.isfinite(torch.tensor(losses)))
    assert any("eval/return" in m for m in logged)
    assert "return" in best
    loaded = DQNAgent.load(tmp_path / "dqn.pt", "cpu")
    loaded.q.encoder.item_static.copy_(env.item_static)
    obs = env.reset(seed=0)
    action = loaded.act(obs)
    assert action.shape == (16, SLATE)


def test_dqn_epsilon_one_is_random_and_zero_is_greedy(env):
    agent = DQNAgent.for_env(env, DQNConfig(hidden=32), seed=0)
    obs = env.reset(seed=0)
    agent.epsilon = 0.0
    greedy = agent.act(obs)
    q = agent.q(CompactObs.from_obs(obs))
    assert torch.equal(greedy, q.topk(SLATE, dim=1).indices)
    agent.epsilon = 1.0
    assert not torch.equal(agent.act(obs), greedy)


def test_bootstrap_ci_and_paired_difference():
    from slate_rl.eval.stats import bootstrap_ci, paired_difference

    mean, lo, hi = bootstrap_ci(torch.ones(50) * 3.0)
    assert mean == lo == hi == pytest.approx(3.0)
    gen = torch.Generator().manual_seed(0)
    base = torch.randn(400, generator=gen)
    diff, lo, hi = paired_difference(base + 0.5, base)
    assert diff == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    mean, lo, hi = bootstrap_ci(torch.randn(400, generator=gen))
    assert lo < mean < hi and lo < 0 < hi
