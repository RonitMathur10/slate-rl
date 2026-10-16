"""Streamlit viewer: replay simulated sessions side by side, browse results, run live sessions.

Run:  uv run streamlit run viewer/app.py
Reads reports/ (committed), so it works without any trained artifacts. The "Live session"
tab calls the API at $API_URL (default http://localhost:8000).
"""

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import polars as pl
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
API_URL = os.environ.get("API_URL", "http://localhost:8000")
LABELS = {
    "slateq": "SlateQ",
    "ppo_pl": "PPO (Plackett-Luce)",
    "dqn": "DQN",
    "ppo": "PPO (proto-actions)",
    "greedy_click": "greedy-click",
    "mmr": "MMR",
    "retrieval": "retrieval",
    "popularity": "popularity",
    "random": "random",
}
# Reference categorical palette (fixed order): left policy = slot 1, right policy = slot 2.
LEFT, RIGHT = "#2a78d6", "#eb6834"

st.set_page_config(page_title="slate-rl viewer", page_icon="🎬", layout="wide")


def label(policy: str) -> str:
    return LABELS.get(policy, policy)


@st.cache_data
def load_trajectories() -> tuple[pd.DataFrame, pd.DataFrame]:
    traj = pl.read_parquet(REPORTS / "trajectories" / "trajectories.parquet")
    users = pl.read_parquet(REPORTS / "trajectories" / "users.parquet")
    return traj.to_pandas(), users.to_pandas()


@st.cache_data
def load_results() -> dict | None:
    path = REPORTS / "eval_seeds" / "results.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    return saved.get("results", saved)


def session_of(traj: pd.DataFrame, policy: str, user: int) -> pd.DataFrame:
    s = traj[(traj.policy == policy) & (traj.user_idx == user)].sort_values("step").copy()
    s["cumulative_reward"] = s["reward"].cumsum()
    return s.reset_index(drop=True)


def render_slate(session: pd.DataFrame, step: int, color: str) -> None:
    if step >= len(session):
        last = session.iloc[-1]
        reason = "left the session" if last.terminated else "reached the step limit"
        st.info(f"Session over after {len(session)} steps: the user {reason}.")
        return
    row = session.iloc[step]
    shown_before = {t for titles in session.slate_titles[:step] for t in titles}
    for pos, title in enumerate(row.slate_titles):
        repeat = " · *shown before*" if title in shown_before else ""
        if pos == row.clicked_position:
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:6px 10px;margin:4px 0;"
                f"background:rgba(127,127,127,0.10);border-radius:4px'>"
                f"<b>{pos + 1}. {title}</b> &nbsp;✅ clicked · watched {row.watch:.0%}{repeat}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding:6px 10px;margin:4px 0;color:gray'>{pos + 1}. {title}"
                f"{repeat}</div>",
                unsafe_allow_html=True,
            )
    if row.clicked_position < 0:
        st.caption("No click on this slate.")
    st.caption(
        f"step reward {row.reward:+.2f} · fatigue {row.fatigue:.2f} · "
        f"satisfaction {row.satisfaction:.2f}"
    )


def line_chart(df: pd.DataFrame, column: str, title: str, colors: dict[str, str]) -> alt.Chart:
    return (
        alt.Chart(df, title=title)
        .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title="step", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y(f"{column}:Q", title=None),
            color=alt.Color(
                "policy_label:N",
                scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["policy_label:N", "step:Q", alt.Tooltip(f"{column}:Q", format=".2f")],
        )
        .properties(height=220)
    )


# ----- tabs ---------------------------------------------------------------------------------
st.title("🎬 slate-rl")
st.caption(
    "RL agents recommend 5-movie slates to simulated MovieLens users and are trained for "
    "long-term engagement. Fatigue and satisfaction are the simulator's hidden state, which "
    "the agents never see."
)
replay_tab, results_tab, live_tab = st.tabs(["Session replay", "Results", "Live session"])

with replay_tab:
    traj, users = load_trajectories()
    policies = sorted(traj.policy.unique(), key=lambda p: list(LABELS).index(p))
    returns = traj.groupby(["policy", "user_idx"]).reward.sum().unstack(0)
    c1, c2, c3 = st.columns([2, 1, 1])
    left = c2.selectbox("Left policy", policies, index=policies.index("slateq"), format_func=label)
    right = c3.selectbox(
        "Right policy", policies, index=policies.index("greedy_click"), format_func=label
    )
    gap = returns[left] - returns[right]
    order = st.radio(
        "Order users by",
        ["user id", f"biggest {label(left)} advantage", f"biggest {label(right)} advantage"],
        horizontal=True,
        help="Sessions are stochastic: the extremes include luck in when users leave. "
        "The 'All exported users' table below shows the averages.",
    )
    if order == "user id":
        ordered = sorted(gap.index)
    else:
        ordered = gap.sort_values(ascending=not order.endswith(f"{label(left)} advantage")).index
    user = c1.selectbox(
        "Held-out user",
        list(ordered),
        format_func=lambda u: (
            f"user {u}  ·  {label(left)} {returns.loc[u, left]:.1f} vs "
            f"{label(right)} {returns.loc[u, right]:.1f}"
        ),
    )
    history = users.set_index("user_idx").loc[user, "history_titles"]
    st.markdown("**Recently watched (real MovieLens history):** " + " · ".join(history))

    sl, sr = session_of(traj, left, user), session_of(traj, right, user)
    n_steps = max(len(sl), len(sr))
    step = st.slider("Step", 0, n_steps - 1, 0) if n_steps > 1 else 0

    col_l, col_r = st.columns(2)
    for col, session, policy, color in ((col_l, sl, left, LEFT), (col_r, sr, right, RIGHT)):
        with col:
            st.subheader(label(policy))
            m1, m2, m3 = st.columns(3)
            m1.metric("Session length", len(session))
            m2.metric("Total reward", f"{session.reward.sum():.2f}")
            m3.metric("Clicks", int((session.clicked_position >= 0).sum()))
            render_slate(session, step, color)

    both = pd.concat([sl.assign(policy_label=label(left)), sr.assign(policy_label=label(right))])
    colors = {label(left): LEFT, label(right): RIGHT}
    g1, g2, g3 = st.columns(3)
    g1.altair_chart(
        line_chart(both, "cumulative_reward", "Cumulative reward", colors), use_container_width=True
    )
    g2.altair_chart(
        line_chart(both, "fatigue", "Fatigue (hidden)", colors), use_container_width=True
    )
    g3.altair_chart(
        line_chart(both, "satisfaction", "Satisfaction (hidden)", colors), use_container_width=True
    )

    with st.expander(f"All {len(returns)} exported users"):
        summary = (
            traj.groupby(["policy", "user_idx"])
            .agg(
                reward=("reward", "sum"),
                length=("step", "size"),
                clicks=("clicked_position", lambda c: (c >= 0).sum()),
            )
            .groupby(level=0)
            .mean()
        )
        summary.index = [label(p) for p in summary.index]
        st.dataframe(summary.sort_values("reward", ascending=False).style.format("{:.2f}"))

with results_tab:
    results = load_results()
    st.markdown("Held-out users, 5 training seeds per agent (95% CI across seeds).")
    st.image(str(REPORTS / "eval_seeds" / "nominal_returns.png"))
    st.image(str(REPORTS / "eval_seeds" / "robustness.png"))
    curves = REPORTS / "eval" / "learning_curves.png"
    if curves.exists():
        st.image(str(curves), caption="Validation learning curves (seed 0)")
    if results:
        rows = []
        for scen, by_policy in results.items():
            for policy, runs in by_policy.items():
                rows.append(
                    {
                        "scenario": scen,
                        "policy": label(policy),
                        "return": sum(r["return"] for r in runs) / len(runs),
                    }
                )
        table = pd.DataFrame(rows).pivot(index="policy", columns="scenario", values="return")
        st.markdown("**Mean return per scenario**")
        st.dataframe(table.sort_values("nominal", ascending=False).style.format("{:.2f}"))

with live_tab:
    st.markdown(f"Runs a brand-new simulated session through the API at `{API_URL}`.")
    try:
        available = requests.get(f"{API_URL}/policies", timeout=3).json()["policies"]
    except requests.RequestException:
        available = None
        st.warning(
            "The API is not reachable. Start it with "
            "`uv run uvicorn slate_rl.serving.api:app --port 8000` or `docker compose up`."
        )
    if available:
        a, b = st.columns(2)
        policy = a.selectbox(
            "Policy",
            available,
            index=available.index("slateq") if "slateq" in available else 0,
            format_func=label,
        )
        seed = b.number_input("Seed (picks the user and the randomness)", 0, 10_000, 7)
        if st.button("Run session", type="primary"):
            with st.spinner("Simulating..."):
                out = requests.post(
                    f"{API_URL}/simulate", json={"policy": policy, "seed": int(seed)}, timeout=60
                ).json()
            st.markdown(
                f"**User {out['user_idx']}** recently watched: "
                + " · ".join(h["title"] for h in out["history"])
            )
            st.metric(
                "Total reward", f"{out['total_reward']:.2f}", help=f"{len(out['steps'])} steps"
            )
            for s in out["steps"]:
                c = s["clicked_position"]
                picked = (
                    f"✅ **{s['slate'][c]['title']}** (watched {s['watch']:.0%})"
                    if c >= 0
                    else "no click"
                )
                st.markdown(
                    f"**Step {s['step'] + 1}**: {picked}  \n"
                    + " · ".join(x["title"] for x in s["slate"])
                    + f"  \n<span style='color:gray'>reward {s['reward']:+.2f} · fatigue "
                    f"{s['fatigue']:.2f}</span>",
                    unsafe_allow_html=True,
                )
