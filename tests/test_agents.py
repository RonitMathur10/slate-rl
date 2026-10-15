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


# ----- SlateQ -------------------------------------------------------------------------------
def make_slateq(env, cfg=None):
    from slate_rl.agents.slateq import SlateQAgent, SlateQConfig

    return SlateQAgent.for_env(env, cfg or SlateQConfig(hidden=32), make_click_model(env), seed=0)


def test_slateq_slate_reward_matches_env_reward_components(env):
    agent = make_slateq(env)
    obs = env.reset(seed=0)
    for _ in range(3):  # later steps include re-shown items
        action = agent.act(obs)
        compact = CompactObs.from_obs(obs)
        predicted = agent.slate.slate_reward(compact, action, agent.item_static)
        obs, _, _, _, info = env.step(action)
        c, w = info["reward_components"], env.cfg.reward
        assert torch.allclose(
            predicted, w.diversity * c["diversity"] - w.repeat * c["repeat"], atol=1e-5
        )


def test_slateq_greedy_value_is_consistent_with_slate_value(env):
    from slate_rl.agents.slateq import slate_q_value

    agent = make_slateq(env)
    compact = CompactObs.from_obs(env.reset(seed=0))
    slate, best = agent._greedy(agent.net, compact)
    for row in slate:
        assert len(set(row.tolist())) == SLATE
    qbar, q_none = agent.net(compact)
    scores, none_logit = agent.click_model.scores(compact)
    direct = slate_q_value(
        slate, qbar, q_none, scores, none_logit, agent.click_model.position_bias
    ) + agent.slate.slate_reward(compact, slate, agent.item_static)
    assert torch.allclose(best, direct, atol=1e-4)


def test_slateq_greedy_prefers_high_value_items():
    from slate_rl.agents.slateq import greedy_slate
    from slate_rl.envs import RewardConfig

    qbar = torch.tensor([[0.0, 5.0, 1.0, 4.0, 0.5]])
    scores = torch.zeros(1, 5)
    slate, _ = greedy_slate(
        qbar,
        torch.zeros(1),
        scores,
        torch.zeros(1),
        torch.zeros(2),
        dist=torch.zeros(1, 5, 5),
        shown=torch.zeros(1, 5),
        reward=RewardConfig(diversity=0.0, repeat=0.0),
        k=2,
    )
    assert slate.tolist() == [[1, 3]]


def test_slateq_trains_and_roundtrips(parts, tmp_path):
    from slate_rl.agents.slateq import SlateQAgent, SlateQConfig, train_slateq

    env = make_env(parts, num_envs=16, seed=0)
    eval_env = make_env(parts, num_envs=8, autoreset=False)
    cfg = SlateQConfig(
        total_iters=40,
        buffer_size=2000,
        batch_size=32,
        learning_starts=64,
        updates_per_iter=1,
        hidden=32,
        eval_interval=20,
        log_interval=10,
    )
    agent = make_slateq(env, cfg)
    logged = []
    train_slateq(
        agent,
        env,
        cfg,
        evaluate=lambda a: evaluate_policy(eval_env, a, eval_env.user_pool[:8], seed=0),
        log=lambda m, step: logged.append(m),
        checkpoint=tmp_path / "slateq.pt",
    )
    losses = [m["train/loss"] for m in logged if "train/loss" in m]
    assert losses and all(torch.isfinite(torch.tensor(losses)))
    assert (tmp_path / "slateq.pt").exists()
    agent.epsilon = 0.0
    agent.save(tmp_path / "final.pt")
    loaded = SlateQAgent.load(tmp_path / "final.pt", "cpu")
    obs = env.reset(seed=1)
    assert torch.equal(loaded.act(obs), agent.act(obs))


# ----- PPO ----------------------------------------------------------------------------------
def test_plackett_luce_log_probs_sum_to_one_over_all_ordered_slates():
    import itertools

    from slate_rl.agents.ppo import plackett_luce_log_prob

    scores = torch.tensor([[0.3, -1.0, 2.0, 0.5]])
    slates = torch.tensor(list(itertools.permutations(range(4), 2)))  # all 12 ordered pairs
    logp = plackett_luce_log_prob(scores.expand(len(slates), -1), slates)
    assert logp.exp().sum().item() == pytest.approx(1.0, abs=1e-6)


def test_gumbel_top_k_matches_plackett_luce_frequencies():
    from slate_rl.agents.ppo import gumbel_top_k, plackett_luce_log_prob

    scores = torch.tensor([[1.0, 0.0, -1.0]]).expand(20000, -1)
    slates = gumbel_top_k(scores, 2, torch.Generator().manual_seed(0))
    target = torch.tensor([[0, 1]])
    empirical = (slates == target).all(1).float().mean().item()
    expected = plackett_luce_log_prob(scores[:1], target).exp().item()
    assert empirical == pytest.approx(expected, abs=0.01)


def test_nearest_unused_picks_closest_distinct_candidates():
    from slate_rl.agents.ppo import nearest_unused

    cand = torch.nn.functional.normalize(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]]), dim=-1
    )
    proto = torch.tensor([[[1.0, 0.1], [1.0, 0.0]]])  # both closest to candidate 0
    assert nearest_unused(proto, cand).tolist() == [[0, 2]]


def test_gae_matches_hand_computation():
    from slate_rl.agents.ppo import compute_gae

    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.tensor([[False], [True], [False]])
    adv, ret = compute_gae(rewards, values, dones, torch.tensor([1.0]), gamma=0.9, lam=0.8)
    d2 = 3.0 + 0.9 * 1.0 - 0.5
    d1 = 2.0 - 0.5  # episode ends after step 1: no bootstrap, no carry
    d0 = 1.0 + 0.9 * 0.5 - 0.5
    expected = [d0 + 0.9 * 0.8 * d1, d1, d2]
    assert adv.squeeze(1).tolist() == pytest.approx(expected)
    assert torch.allclose(ret, adv + values)


@pytest.mark.parametrize("head", ["proto", "plackett_luce"])
def test_ppo_trains_and_roundtrips(parts, tmp_path, head):
    from slate_rl.agents.ppo import PPOAgent, PPOConfig, train_ppo

    env = make_env(parts, num_envs=16, seed=0)
    eval_env = make_env(parts, num_envs=8, autoreset=False)
    cfg = PPOConfig(
        head=head, total_steps=16 * 8 * 4, rollout_len=8, minibatches=4, hidden=32, eval_interval=2
    )
    agent = PPOAgent.for_env(env, cfg, seed=0)
    logged = []
    best = train_ppo(
        agent,
        env,
        cfg,
        evaluate=lambda a: evaluate_policy(eval_env, a, eval_env.user_pool[:8], seed=0),
        log=lambda m, step: logged.append(m),
        checkpoint=tmp_path / "ppo.pt",
    )
    losses = [m["train/policy_loss"] for m in logged if "train/policy_loss" in m]
    assert len(losses) == 4 and all(torch.isfinite(torch.tensor(losses)))
    assert "return" in best
    agent.save(tmp_path / "final.pt")
    loaded = PPOAgent.load(tmp_path / "final.pt", "cpu")
    obs = env.reset(seed=1)
    action = loaded.act(obs)
    assert torch.equal(action, agent.act(obs))
    for row in action:
        assert len(set(row.tolist())) == SLATE


def test_anchored_proto_policy_starts_as_retrieval_ranking(env):
    from slate_rl.agents.ppo import PPOAgent, PPOConfig

    agent = PPOAgent.for_env(env, PPOConfig(head="proto", proto_anchor=True, hidden=32))
    obs = env.reset(seed=0)
    retrieval_order = torch.arange(SLATE).expand(env.num_envs, -1)  # candidates arrive sorted
    assert torch.equal(agent.act(obs), retrieval_order)


# ----- milestone 8 evaluation helpers --------------------------------------------------------
def test_seed_ci():
    from slate_rl.eval.stats import seed_ci

    mean, lo, hi = seed_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0
    assert (hi - mean) == pytest.approx(2.776 * (2.5**0.5) / 5**0.5, rel=1e-3)
    assert seed_ci([2.0]) == (2.0, 2.0, 2.0)


def test_record_trajectories_matches_evaluation(parts):
    from slate_rl.eval.trajectories import record_trajectories

    env = make_env(parts, num_envs=6, autoreset=False)
    users = env.user_pool[:6]
    policy = make_baselines(env, None)["retrieval"]
    titles = [f"title {i}" for i in range(env.item_static.shape[0])]
    traj = record_trajectories(env, policy, users, titles, "retrieval", seed=0)
    result = evaluate_policy(env, policy, users, seed=0)
    per_user = traj.group_by("user_idx").agg(
        __import__("polars").col("reward").sum(), __import__("polars").len().alias("n")
    )
    assert per_user["reward"].mean() == pytest.approx(result["return"], rel=1e-5)
    assert per_user["n"].mean() == pytest.approx(result["session_length"])
    row = traj.row(0, named=True)
    assert len(row["slate"]) == SLATE and len(row["slate_titles"]) == SLATE
    assert (row["clicked_title"] is None) == (row["clicked_position"] == -1)
