import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from slate_rl.data.features import ItemFeatures, UserFeatures
from slate_rl.data.sequences import UserSequences
from slate_rl.envs import N_SESSION_FEATURES, BatchedSlateEnv, EnvConfig, SlateRecEnv
from slate_rl.retrieval import CandidateGenerator, TwoTower
from slate_rl.simulator import SimParams, UserChoiceModel, UserSimulator

HIST, K, SLATE, N = 8, 6, 3, 5


@pytest.fixture
def parts(processed):
    torch.manual_seed(0)
    feats = ItemFeatures.from_processed(processed)
    seqs = UserSequences(processed, HIST)
    cand_gen = CandidateGenerator(
        TwoTower.from_features(feats, emb_dim=8, hidden_dim=16),
        UserFeatures.from_processed(processed).demo,
    )
    model = UserChoiceModel(processed.meta["n_items"], max_hist=HIST, emb_dim=16, dropout=0.0)
    return feats, seqs, cand_gen, model


def make_env(parts, num_envs=N, autoreset=True, seed=0, **sim_kwargs):
    feats, seqs, cand_gen, model = parts
    params = SimParams(**{"max_steps": 5, "sim_ref": 0.1, **sim_kwargs})
    sim = UserSimulator(model, params, seed=seed + 1)
    cfg = EnvConfig(num_envs=num_envs, n_candidates=K, slate_size=SLATE, autoreset=autoreset)
    return BatchedSlateEnv(
        sim, cand_gen, seqs, seqs.model_users(), feats.genres, feats.popularity, cfg, seed=seed
    )


def first_k_action(n=N):
    return torch.arange(SLATE).repeat(n, 1)


def test_observation_shapes_and_validity(parts):
    env = make_env(parts)
    obs = env.reset(seed=0)
    assert obs.user_emb.shape == (N, 8)
    assert obs.session.shape == (N, N_SESSION_FEATURES)
    assert obs.history.shape == (N, HIST)
    assert obs.cand_items.shape == (N, K)
    assert obs.cand_feats.shape == (N, K, env.cand_feat_dim)
    assert torch.isfinite(obs.cand_feats).all() and torch.isfinite(obs.session).all()
    assert not env.sim.seen.gather(1, obs.cand_items).any(), "candidates must be unseen"
    for row in obs.cand_items:
        assert len(set(row.tolist())) == K


def test_step_outputs_and_reward_decomposition(parts):
    env = make_env(parts)
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(first_k_action())
    assert reward.shape == term.shape == trunc.shape == (N,)
    c, w = info["reward_components"], env.cfg.reward
    expected = (
        w.click * c["click"]
        + w.watch * c["watch"]
        + w.diversity * c["diversity"]
        + w.novelty * c["novelty"]
        - w.repeat * c["repeat"]
    )
    assert torch.allclose(reward, expected)
    assert torch.equal(c["click"].bool(), info["clicked"])
    lo = -w.repeat
    hi = w.click + w.watch + w.diversity + w.novelty
    assert ((reward >= lo - 1e-6) & (reward <= hi + 1e-6)).all()


def test_duplicate_actions_are_fixed(parts):
    env = make_env(parts)
    env.reset(seed=0)
    action = torch.zeros(N, SLATE, dtype=torch.long)  # every slot picks candidate 0
    _, _, _, _, info = env.step(action)
    assert info["action_fixed"].all()
    for row in info["slate"]:
        assert len(set(row.tolist())) == SLATE


def test_invalid_actions_raise(parts):
    env = make_env(parts)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(torch.full((N, SLATE), K))
    with pytest.raises(ValueError):
        env.step(torch.zeros(N, SLATE + 1, dtype=torch.long))


def test_repeat_penalty_tracks_shown_items(parts):
    env = make_env(parts, no_click_bias=50.0, leave_bias=-50.0)  # never click, never leave
    env.reset(seed=0)
    _, _, _, _, info = env.step(first_k_action())
    assert (info["reward_components"]["repeat"] == 0).all()
    assert env.shown.gather(1, info["slate"]).all()
    # Nothing was clicked, so the candidates are unchanged and the same slate is re-shown.
    _, _, _, _, info2 = env.step(first_k_action())
    assert torch.equal(info2["slate"], info["slate"])
    assert (info2["reward_components"]["repeat"] == 1).all()


def test_truncation_vs_termination_and_autoreset(parts):
    env = make_env(parts, leave_bias=-50.0)  # nobody leaves -> sessions end by truncation
    env.reset(seed=0)
    for t in range(5):
        obs, _, term, trunc, info = env.step(first_k_action())
        assert not term.any()
        if t < 4:
            assert not trunc.any()
    assert trunc.all()
    assert "final_obs" in info and info["final_obs"].cand_items.shape == (N, K)
    assert torch.equal(info["episode"]["length"], torch.full((N,), 5))
    assert (env.sim.t == 0).all() and (env.ep_len == 0).all(), "rows should be auto-reset"
    assert (obs.session[:, 0] == 0).all()


def test_leaving_is_termination(parts):
    env = make_env(parts, leave_bias=50.0)  # everyone leaves after the first step
    env.reset(seed=0)
    _, _, term, trunc, info = env.step(first_k_action())
    assert term.all() and not trunc.any()
    assert "final_obs" not in info
    assert (info["episode"]["length"] == 1).all()


def test_seeded_determinism(parts):
    runs = []
    for _ in range(2):
        env = make_env(parts, seed=7)
        env.reset(seed=7)
        runs.append([env.step(first_k_action())[1] for _ in range(6)])
    for a, b in zip(*runs, strict=True):
        assert torch.equal(a, b)


def test_episode_stats_match_step_rewards(parts):
    env = make_env(parts, leave_bias=-50.0)
    env.reset(seed=0)
    total = torch.zeros(N)
    for _ in range(5):
        _, reward, _, _, info = env.step(first_k_action())
        total += reward
    assert torch.allclose(info["episode"]["return"], total)


def test_gym_wrapper_passes_env_checker(parts):
    env = SlateRecEnv(make_env(parts, num_envs=1, autoreset=False))
    check_env(env, skip_render_check=True)


def test_gym_wrapper_matches_batched_env(parts):
    gym_env = SlateRecEnv(make_env(parts, num_envs=1, autoreset=False, seed=3))
    batched = make_env(parts, num_envs=1, autoreset=False, seed=3)
    obs_g, _ = gym_env.reset(seed=3)
    obs_b = batched.reset(seed=3)
    np.testing.assert_allclose(obs_g["cand_feats"], obs_b.cand_feats[0].numpy(), rtol=1e-6)
    for _ in range(4):
        _, r_g, term_g, trunc_g, _ = gym_env.step(np.arange(SLATE))
        _, r_b, term_b, trunc_b, _ = batched.step(first_k_action(1))
        assert r_g == pytest.approx(float(r_b[0]))
        assert (term_g, trunc_g) == (bool(term_b[0]), bool(trunc_b[0]))
        if term_g or trunc_g:
            break


def test_gym_wrapper_rejects_autoreset_env(parts):
    with pytest.raises(ValueError):
        SlateRecEnv(make_env(parts, num_envs=1, autoreset=True))
