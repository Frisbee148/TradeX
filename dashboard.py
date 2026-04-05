"""TradeX Market Surveillance Dashboard — Gradio UI."""

from __future__ import annotations

import json
import random
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from meverse.server.meverse_environment import MarketSurveillanceEnvironment
from meverse.models import SurveillanceAction
from meverse.tasks import list_task_names, task_definition, compute_task_grade
from meverse.baseline_policy import choose_surveillance_action

# ---------------------------------------------------------------------------
# Theme & colors — no purple, clean dark look
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0f1117",
    "surface": "#1a1d27",
    "border": "#2a2d3a",
    "text": "#e4e6eb",
    "muted": "#8b8fa3",
    "accent": "#00c9a7",       # teal
    "accent2": "#0088cc",      # blue
    "danger": "#ff4757",       # red
    "warning": "#ffa502",      # amber
    "success": "#2ed573",      # green
    "info": "#3498db",         # light blue
}

ACTION_COLORS = {
    "ALLOW": "#2ed573",
    "FLAG": "#ffa502",
    "BLOCK": "#ff4757",
    "MONITOR": "#3498db",
}

LABEL_COLORS = {
    "suspicious": "#ff4757",
    "normal": "#2ed573",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,39,0.8)",
    font=dict(color="#e4e6eb", family="Inter, system-ui, sans-serif"),
    margin=dict(l=50, r=30, t=40, b=40),
    xaxis=dict(gridcolor="#2a2d3a", zerolinecolor="#2a2d3a"),
    yaxis=dict(gridcolor="#2a2d3a", zerolinecolor="#2a2d3a"),
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class EpisodeState:
    """Holds the mutable state for a running episode."""

    def __init__(self):
        self.env: MarketSurveillanceEnvironment | None = None
        self.observation = None
        self.step_history: list[dict[str, Any]] = []
        self.actions: list[str] = []
        self.labels: list[str] = []
        self.rewards: list[float] = []
        self.amm_prices: list[float] = []
        self.amm_liquidity: list[float] = []
        self.amm_bot_conf: list[float] = []
        self.amm_volatility: list[float] = []
        self.amm_health: list[float] = []
        self.signal_matrix: list[list[float]] = []  # for heatmap
        self.done = False
        self.task_name = ""


def _build_signal_row(obs) -> list[float]:
    return [
        obs.burst_indicator,
        obs.pattern_indicator,
        obs.suspiciousness_score,
        obs.manipulation_score,
        obs.trade_frequency / 10.0,  # normalized
        obs.recent_slippage_impact * 10.0,  # scaled for visibility
    ]


SIGNAL_NAMES = [
    "Burst",
    "Pattern",
    "Suspicion",
    "Manipulation",
    "Frequency",
    "Slippage",
]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_full_episode(task_name: str, policy: str, seed: int | None) -> tuple:
    """Run a complete episode and return all visualizations."""
    state = EpisodeState()
    state.task_name = task_name

    if seed == 0:
        seed = None

    env = MarketSurveillanceEnvironment(
        task=task_name, eval_mode=(seed is not None), demo_mode=(seed is None)
    )
    obs = env.reset(task=task_name, seed=seed if seed else None)
    state.env = env
    state.observation = obs

    # Record initial AMM state
    snap = env.debug_snapshot()
    state.amm_prices.append(snap["amm_state"]["price"])
    state.amm_liquidity.append(snap["amm_state"]["liquidity"])
    state.amm_bot_conf.append(snap["amm_state"]["bot_confidence"])
    state.amm_volatility.append(snap["amm_state"]["volatility"])
    state.amm_health.append(snap["amm_state"]["health_index"])
    state.signal_matrix.append(_build_signal_row(obs))

    step_count = 0
    while not obs.done:
        # Select action
        if policy == "Heuristic":
            action = choose_surveillance_action(obs)
        elif policy == "Random":
            action = random.choice(["ALLOW", "FLAG", "BLOCK", "MONITOR"])
        else:
            # Always-allow baseline
            action = "ALLOW"

        pre_snap = env.debug_snapshot()
        label = pre_snap["current_step"]["label"] if pre_snap["current_step"] else "normal"

        obs = env.step(SurveillanceAction(action_type=action))
        step_count += 1
        reward = float(obs.reward or 0.0)

        state.actions.append(action)
        state.labels.append(label)
        state.rewards.append(reward)

        post_snap = env.debug_snapshot()
        state.amm_prices.append(post_snap["amm_state"]["price"])
        state.amm_liquidity.append(post_snap["amm_state"]["liquidity"])
        state.amm_bot_conf.append(post_snap["amm_state"]["bot_confidence"])
        state.amm_volatility.append(post_snap["amm_state"]["volatility"])
        state.amm_health.append(post_snap["amm_state"]["health_index"])

        if not obs.done:
            state.signal_matrix.append(_build_signal_row(obs))

        state.step_history.append({
            "step": step_count,
            "action": action,
            "label": label,
            "reward": reward,
            "burst": obs.burst_indicator,
            "pattern": obs.pattern_indicator,
            "suspicion": obs.suspiciousness_score,
            "manipulation": obs.manipulation_score,
        })

    grade = env.grade()
    state.done = True

    return (
        _make_reward_chart(state),
        _make_action_dist_chart(state),
        _make_signal_heatmap(state),
        _make_amm_chart(state),
        _make_grade_chart(grade),
        _make_confusion_chart(state),
        _make_episode_summary(state, grade, policy, seed),
        _make_step_table(state),
        _make_amm_gauges(state),
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _make_reward_chart(state: EpisodeState) -> go.Figure:
    steps = list(range(1, len(state.rewards) + 1))
    colors = [ACTION_COLORS.get(a, "#888") for a in state.actions]
    label_colors = [LABEL_COLORS.get(l, "#888") for l in state.labels]

    fig = go.Figure()

    # Reward bars colored by action
    fig.add_trace(go.Bar(
        x=steps, y=state.rewards,
        marker_color=colors,
        name="Reward",
        hovertemplate="Step %{x}<br>Reward: %{y:.3f}<br>Action: %{customdata[0]}<br>Label: %{customdata[1]}",
        customdata=list(zip(state.actions, state.labels)),
    ))

    # Cumulative reward line
    cumulative = np.cumsum(state.rewards).tolist()
    fig.add_trace(go.Scatter(
        x=steps, y=cumulative,
        mode="lines",
        name="Cumulative",
        line=dict(color=COLORS["accent"], width=2, dash="dot"),
        yaxis="y2",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Reward per Step", font=dict(size=14)),
        xaxis_title="Step",
        yaxis_title="Reward",
        yaxis2=dict(
            title="Cumulative",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            color=COLORS["accent"],
        ),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=340,
        bargap=0.15,
    )
    return fig


def _make_action_dist_chart(state: EpisodeState) -> go.Figure:
    actions = ["ALLOW", "FLAG", "BLOCK", "MONITOR"]

    # Count by label
    suspicious_counts = []
    normal_counts = []
    for a in actions:
        s_count = sum(1 for act, lbl in zip(state.actions, state.labels) if act == a and lbl == "suspicious")
        n_count = sum(1 for act, lbl in zip(state.actions, state.labels) if act == a and lbl == "normal")
        suspicious_counts.append(s_count)
        normal_counts.append(n_count)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=actions, y=normal_counts,
        name="Normal",
        marker_color=COLORS["success"],
        marker_line=dict(width=0),
    ))
    fig.add_trace(go.Bar(
        x=actions, y=suspicious_counts,
        name="Suspicious",
        marker_color=COLORS["danger"],
        marker_line=dict(width=0),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Action Distribution by Label", font=dict(size=14)),
        barmode="stack",
        xaxis_title="Action",
        yaxis_title="Count",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=340,
    )
    return fig


def _make_signal_heatmap(state: EpisodeState) -> go.Figure:
    matrix = np.array(state.signal_matrix).T
    steps = list(range(len(state.signal_matrix)))

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=steps,
        y=SIGNAL_NAMES,
        colorscale=[
            [0.0, "#0f1117"],
            [0.25, "#0a3d5c"],
            [0.5, "#0088cc"],
            [0.75, "#ffa502"],
            [1.0, "#ff4757"],
        ],
        hovertemplate="Step %{x}<br>%{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="Intensity", font=dict(size=11))),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Signal Heatmap Over Time", font=dict(size=14)),
        xaxis_title="Step",
        height=300,
    )
    return fig


def _make_amm_chart(state: EpisodeState) -> go.Figure:
    steps = list(range(len(state.amm_prices)))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("AMM Price & Liquidity", "Bot Confidence & Volatility"),
    )

    fig.add_trace(go.Scatter(
        x=steps, y=state.amm_prices,
        name="Price", line=dict(color=COLORS["accent2"], width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=state.amm_liquidity,
        name="Liquidity", line=dict(color=COLORS["accent"], width=2),
        yaxis="y2",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=state.amm_bot_conf,
        name="Bot Confidence", line=dict(color=COLORS["danger"], width=2),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=state.amm_volatility,
        name="Volatility", line=dict(color=COLORS["warning"], width=2),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=state.amm_health,
        name="Health Index", line=dict(color=COLORS["success"], width=2),
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,39,0.8)",
        font=dict(color="#e4e6eb", family="Inter, system-ui, sans-serif"),
        margin=dict(l=50, r=30, t=40, b=40),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        height=480,
    )

    for i in range(1, 3):
        fig.update_xaxes(gridcolor="#2a2d3a", row=i, col=1)
        fig.update_yaxes(gridcolor="#2a2d3a", row=i, col=1)

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#e4e6eb")

    return fig


def _make_grade_chart(grade: dict) -> go.Figure:
    categories = ["Detection", "False Positive", "False Negative", "Health", "Overblocking"]
    values = [
        grade["detection_score"],
        grade["false_positive_score"],
        grade["false_negative_score"],
        grade["health_score"],
        grade["overblocking_score"],
    ]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(0,200,167,0.15)",
        line=dict(color=COLORS["accent"], width=2),
        name="Score",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(26,29,39,0.8)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="#2a2d3a",
                tickfont=dict(size=10, color="#8b8fa3"),
            ),
            angularaxis=dict(
                gridcolor="#2a2d3a",
                tickfont=dict(size=11, color="#e4e6eb"),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e4e6eb", family="Inter, system-ui, sans-serif"),
        title=dict(text="Grade Breakdown (Radar)", font=dict(size=14)),
        margin=dict(l=60, r=60, t=50, b=40),
        height=360,
        showlegend=False,
    )
    return fig


def _make_confusion_chart(state: EpisodeState) -> go.Figure:
    """Action x Label confusion matrix."""
    actions_order = ["ALLOW", "MONITOR", "FLAG", "BLOCK"]
    labels_order = ["normal", "suspicious"]

    matrix = []
    for label in labels_order:
        row = []
        for action in actions_order:
            count = sum(1 for a, l in zip(state.actions, state.labels) if a == action and l == label)
            row.append(count)
        matrix.append(row)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=actions_order,
        y=labels_order,
        colorscale=[
            [0.0, "#1a1d27"],
            [0.5, "#0088cc"],
            [1.0, "#00c9a7"],
        ],
        text=matrix,
        texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        hovertemplate="Action: %{x}<br>Label: %{y}<br>Count: %{z}<extra></extra>",
        showscale=False,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Action vs Ground Truth", font=dict(size=14)),
        xaxis_title="Action Taken",
        yaxis_title="True Label",
        height=260,
    )
    return fig


def _make_episode_summary(state: EpisodeState, grade: dict, policy: str, seed: int | None) -> str:
    total_reward = sum(state.rewards)
    avg_reward = total_reward / max(len(state.rewards), 1)
    task_def = task_definition(state.task_name)

    score_bar = _score_bar(grade["score"])

    return f"""### Episode Complete

| Metric | Value |
|--------|-------|
| **Task** | {task_def.title} ({task_def.difficulty}) |
| **Policy** | {policy} |
| **Seed** | {seed if seed else "random"} |
| **Steps** | {len(state.actions)} |
| **Total Reward** | {total_reward:.3f} |
| **Avg Reward** | {avg_reward:.3f} |

---

### Final Score: {grade['score']:.4f}

{score_bar}

| Component | Score | Weight |
|-----------|-------|--------|
| Detection | {grade['detection_score']:.4f} | 50% |
| False Positive | {grade['false_positive_score']:.4f} | 20% |
| False Negative | {grade['false_negative_score']:.4f} | 15% |
| Health | {grade['health_score']:.4f} | 10% |
| Overblocking | {grade['overblocking_score']:.4f} | 5% |
"""


def _score_bar(score: float) -> str:
    filled = int(score * 20)
    empty = 20 - filled
    return f"`[{'=' * filled}{'-' * empty}]` {score:.1%}"


def _make_step_table(state: EpisodeState) -> list[list]:
    rows = []
    for h in state.step_history:
        rows.append([
            h["step"],
            h["action"],
            h["label"],
            f"{h['reward']:.3f}",
            f"{h['burst']:.3f}",
            f"{h['pattern']:.3f}",
            f"{h['suspicion']:.3f}",
            f"{h['manipulation']:.3f}",
        ])
    return rows


def _make_amm_gauges(state: EpisodeState) -> go.Figure:
    """Final AMM state as indicator gauges."""
    final_price = state.amm_prices[-1]
    final_bot = state.amm_bot_conf[-1]
    final_health = state.amm_health[-1]
    final_vol = state.amm_volatility[-1]

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
    )

    gauges = [
        ("Price", final_price, 50, 150, COLORS["accent2"]),
        ("Bot Conf", final_bot, 0, 1, COLORS["danger"]),
        ("Health", final_health, 0, 1, COLORS["success"]),
        ("Volatility", final_vol, 0, 0.5, COLORS["warning"]),
    ]

    for i, (title, value, lo, hi, color) in enumerate(gauges, 1):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=title, font=dict(size=13)),
            number=dict(font=dict(size=18), valueformat=".3f"),
            gauge=dict(
                axis=dict(range=[lo, hi], tickfont=dict(size=9, color="#8b8fa3")),
                bar=dict(color=color),
                bgcolor="#1a1d27",
                borderwidth=0,
                steps=[
                    dict(range=[lo, hi], color="#2a2d3a"),
                ],
            ),
        ), row=1, col=i)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e4e6eb", family="Inter, system-ui, sans-serif"),
        margin=dict(l=20, r=20, t=30, b=10),
        height=180,
    )
    return fig


# ---------------------------------------------------------------------------
# Compare policies
# ---------------------------------------------------------------------------

def compare_policies(task_name: str, seed: int | None) -> tuple:
    if seed == 0:
        seed = 42

    results = {}
    for policy in ["Heuristic", "Always Allow", "Random"]:
        env = MarketSurveillanceEnvironment(task=task_name, eval_mode=True, demo_mode=False)
        obs = env.reset(task=task_name, seed=seed)
        actions_list = []
        labels_list = []
        rewards_list = []

        while not obs.done:
            snap = env.debug_snapshot()
            label = snap["current_step"]["label"] if snap["current_step"] else "normal"

            if policy == "Heuristic":
                action = choose_surveillance_action(obs)
            elif policy == "Random":
                action = random.choice(["ALLOW", "FLAG", "BLOCK", "MONITOR"])
            else:
                action = "ALLOW"

            obs = env.step(SurveillanceAction(action_type=action))
            actions_list.append(action)
            labels_list.append(label)
            rewards_list.append(float(obs.reward or 0.0))

        grade = env.grade()
        results[policy] = {
            "score": grade["score"],
            "detection": grade["detection_score"],
            "fp": grade["false_positive_score"],
            "fn": grade["false_negative_score"],
            "health": grade["health_score"],
            "overblock": grade["overblocking_score"],
            "total_reward": sum(rewards_list),
        }

    # Bar chart comparison
    policies = list(results.keys())
    metrics = ["score", "detection", "fp", "fn", "health", "overblock"]
    metric_labels = ["Final Score", "Detection", "False Pos.", "False Neg.", "Health", "Overblocking"]
    bar_colors = [COLORS["accent"], COLORS["accent2"], COLORS["success"], COLORS["danger"], COLORS["info"], COLORS["warning"]]

    fig = go.Figure()
    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        fig.add_trace(go.Bar(
            x=policies,
            y=[results[p][m] for p in policies],
            name=ml,
            marker_color=bar_colors[i],
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Policy Comparison — {task_name}", font=dict(size=14)),
        barmode="group",
        yaxis_title="Score",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        height=400,
    )

    # Summary table
    summary = f"""### Policy Comparison (seed={seed})

| Policy | Score | Detection | FP | FN | Health | Total Reward |
|--------|-------|-----------|----|----|--------|--------------|
"""
    for p in policies:
        r = results[p]
        summary += f"| {p} | {r['score']:.4f} | {r['detection']:.4f} | {r['fp']:.4f} | {r['fn']:.4f} | {r['health']:.4f} | {r['total_reward']:.2f} |\n"

    return fig, summary


# ---------------------------------------------------------------------------
# Telemetry viewer
# ---------------------------------------------------------------------------

def load_telemetry(file) -> tuple:
    if file is None:
        return go.Figure(), "Upload a `.jsonl` telemetry file to visualize."

    content = file.decode("utf-8") if isinstance(file, bytes) else open(file.name, "r").read()
    lines = [json.loads(l) for l in content.strip().split("\n") if l.strip()]

    steps = [e for e in lines if e.get("event") == "step"]
    if not steps:
        return go.Figure(), "No step events found in telemetry file."

    rewards = [s.get("reward", 0) for s in steps]
    actions = [s.get("action", "?") for s in steps]
    step_nums = list(range(1, len(steps) + 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=step_nums, y=rewards,
        marker_color=[ACTION_COLORS.get(a, "#888") for a in actions],
        hovertemplate="Step %{x}<br>Reward: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Telemetry Replay — Rewards", font=dict(size=14)),
        xaxis_title="Step",
        yaxis_title="Reward",
        height=350,
    )

    # Extract grade if available
    end_events = [e for e in lines if e.get("event") == "episode_end"]
    summary = f"**Steps:** {len(steps)} | **Total reward:** {sum(rewards):.3f}"
    if end_events:
        grade = end_events[0].get("grade", {})
        if grade:
            summary += f"\n\n**Final score:** {grade.get('score', 'N/A')}"

    return fig, summary


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Global overrides — kill all purple/violet tones */
:root {
    --body-background-fill: #0f1117 !important;
    --block-background-fill: #1a1d27 !important;
    --block-border-color: #2a2d3a !important;
    --block-label-text-color: #e4e6eb !important;
    --body-text-color: #e4e6eb !important;
    --button-primary-background-fill: #00c9a7 !important;
    --button-primary-text-color: #0f1117 !important;
    --button-primary-background-fill-hover: #00b396 !important;
    --input-background-fill: #1a1d27 !important;
    --input-border-color: #2a2d3a !important;
    --color-accent: #00c9a7 !important;
    --color-accent-soft: rgba(0, 201, 167, 0.15) !important;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 8px;
}
.header-banner h1 {
    color: #00c9a7 !important;
    font-size: 28px !important;
    margin: 0 0 4px 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #8b8fa3 !important;
    font-size: 14px !important;
    margin: 0 !important;
}

/* Tabs */
.tabs > .tab-nav {
    background: #1a1d27 !important;
    border-bottom: 1px solid #2a2d3a !important;
}
.tabs > .tab-nav button {
    color: #8b8fa3 !important;
    font-weight: 500 !important;
    border: none !important;
}
.tabs > .tab-nav button.selected {
    color: #00c9a7 !important;
    border-bottom: 2px solid #00c9a7 !important;
    background: transparent !important;
}

/* Cards */
.gr-group {
    border: 1px solid #2a2d3a !important;
    border-radius: 10px !important;
    background: #1a1d27 !important;
}

/* Dataframe */
table {
    border-collapse: collapse !important;
}
table th {
    background: #0f1117 !important;
    color: #00c9a7 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #2a2d3a !important;
}
table td {
    border-bottom: 1px solid #2a2d3a !important;
    color: #e4e6eb !important;
}
table tr:hover td {
    background: rgba(0, 201, 167, 0.05) !important;
}

/* Markdown inside the summary */
.prose h3 {
    color: #00c9a7 !important;
}
.prose table th {
    background: #0f1117 !important;
}

/* Number input */
input[type="number"] {
    background: #1a1d27 !important;
    color: #e4e6eb !important;
    border: 1px solid #2a2d3a !important;
}

/* Dropdown */
.gr-dropdown {
    background: #1a1d27 !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

THEME = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e6faf5", c100="#b3f0e0", c200="#80e6cc",
            c300="#4ddcb8", c400="#26d4a8", c500="#00c9a7",
            c600="#00b396", c700="#009d85", c800="#008774",
            c900="#006b5d", c950="#004f46",
        ),
        secondary_hue=gr.themes.Color(
            c50="#e6f3ff", c100="#b3d9ff", c200="#80bfff",
            c300="#4da6ff", c400="#2696ff", c500="#0088cc",
            c600="#007ab8", c700="#006ca3", c800="#005e8f",
            c900="#004a70", c950="#003652",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f0f1f5", c100="#d4d6e0", c200="#b8bbcb",
            c300="#9ca0b6", c400="#8b8fa3", c500="#6b7089",
            c600="#555970", c700="#3f4257", c800="#2a2d3a",
            c900="#1a1d27", c950="#0f1117",
        ),
        font=["Inter", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0f1117",
        block_background_fill="#1a1d27",
        block_border_color="#2a2d3a",
        block_label_text_color="#e4e6eb",
        body_text_color="#e4e6eb",
        button_primary_background_fill="#00c9a7",
        button_primary_text_color="#0f1117",
        input_background_fill="#1a1d27",
        input_border_color="#2a2d3a",
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="TradeX Surveillance Dashboard") as app:

        # Header
        gr.HTML("""
        <div class="header-banner">
            <h1>TradeX Surveillance Dashboard</h1>
            <p>Bot-aware Market Surveillance in Simulated AMM Trading &mdash; Interactive Analysis & Benchmarking</p>
        </div>
        """)

        with gr.Tabs():

            # =============== TAB 1: Episode Runner ===============
            with gr.Tab("Episode Runner"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("#### Configuration")
                        task_dd = gr.Dropdown(
                            choices=list_task_names(),
                            value="full_market_surveillance",
                            label="Task",
                        )
                        policy_dd = gr.Dropdown(
                            choices=["Heuristic", "Always Allow", "Random"],
                            value="Heuristic",
                            label="Policy",
                        )
                        seed_input = gr.Number(
                            value=42, label="Seed (0 = random)", precision=0,
                        )
                        run_btn = gr.Button("Run Episode", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        summary_md = gr.Markdown("Run an episode to see results.")
                        amm_gauges = gr.Plot(label="AMM Final State")

                with gr.Row():
                    reward_chart = gr.Plot(label="Reward Timeline")
                    action_chart = gr.Plot(label="Action Distribution")

                with gr.Row():
                    signal_heatmap = gr.Plot(label="Signal Heatmap")

                with gr.Row():
                    amm_chart = gr.Plot(label="AMM State Evolution")

                with gr.Row():
                    with gr.Column(scale=1):
                        grade_chart = gr.Plot(label="Grade Radar")
                    with gr.Column(scale=1):
                        confusion_chart = gr.Plot(label="Action vs Truth")

                gr.Markdown("#### Step-by-Step Log")
                step_table = gr.Dataframe(
                    headers=["Step", "Action", "Label", "Reward", "Burst", "Pattern", "Suspicion", "Manipulation"],
                    datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

                run_btn.click(
                    fn=run_full_episode,
                    inputs=[task_dd, policy_dd, seed_input],
                    outputs=[
                        reward_chart, action_chart, signal_heatmap, amm_chart,
                        grade_chart, confusion_chart, summary_md, step_table, amm_gauges,
                    ],
                )

            # =============== TAB 2: Policy Comparison ===============
            with gr.Tab("Policy Comparison"):
                gr.Markdown("#### Compare Heuristic, Always-Allow, and Random policies on the same episode")
                with gr.Row():
                    cmp_task = gr.Dropdown(
                        choices=list_task_names(),
                        value="full_market_surveillance",
                        label="Task",
                    )
                    cmp_seed = gr.Number(value=42, label="Seed", precision=0)
                    cmp_btn = gr.Button("Compare", variant="primary")

                cmp_chart = gr.Plot(label="Comparison Chart")
                cmp_summary = gr.Markdown()

                cmp_btn.click(
                    fn=compare_policies,
                    inputs=[cmp_task, cmp_seed],
                    outputs=[cmp_chart, cmp_summary],
                )

            # =============== TAB 3: Telemetry Viewer ===============
            with gr.Tab("Telemetry Viewer"):
                gr.Markdown("#### Upload a JSONL telemetry file to replay and visualize")
                telem_file = gr.File(label="Upload .jsonl", file_types=[".jsonl"])
                telem_chart = gr.Plot(label="Telemetry Rewards")
                telem_summary = gr.Markdown()

                telem_file.change(
                    fn=load_telemetry,
                    inputs=[telem_file],
                    outputs=[telem_chart, telem_summary],
                )

            # =============== TAB 4: About ===============
            with gr.Tab("About"):
                gr.Markdown("""
#### About TradeX

**TradeX** is a bot-aware market surveillance benchmark built on a simulated AMM (Automated Market Maker) environment.

**Tasks:**
- **Burst Detection** (Easy) — Identify sudden bursts of aggressive activity
- **Pattern Manipulation Detection** (Medium) — Detect repeated timing and size signatures
- **Full Market Surveillance** (Hard) — Balance all detection types with false-positive control

**Actions:**
- `ALLOW` — Normal activity, let it through
- `MONITOR` — Watch more closely
- `FLAG` — Mark as suspicious for review
- `BLOCK` — Block the activity

**Scoring Components:**
| Component | Weight | Measures |
|-----------|--------|----------|
| Detection | 50% | Correct identification of suspicious activity |
| False Positive | 20% | Avoiding flagging normal activity |
| False Negative | 15% | Avoiding missing suspicious activity |
| Health | 10% | Preserving healthy market behavior |
| Overblocking | 5% | Not over-blocking normal users |

**AMM Dynamics:**
The environment uses a constant-product AMM (`x * y = k`). Agent actions affect the AMM state — blocking suspicious activity reduces bot confidence and volatility, while allowing it increases both. This creates a feedback loop where early decisions shape future observations.
""")

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=THEME,
        css=CUSTOM_CSS,
        share=True,
    )
