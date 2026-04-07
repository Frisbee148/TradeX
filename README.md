---
title: TradeX Surveillance Dashboard
emoji: "📊"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# TradeX: Bot Aware Market Surveillance in Simulated AMM Trading

TradeX is a reinforcement learning benchmark environment built on the [OpenEnv](https://github.com/openenv) framework. It simulates a constant product Automated Market Maker (AMM) pool and asks an AI agent to act as a **market surveillance controller** — detecting suspicious bot driven trading activity in real time while preserving healthy market participation.

This is **not** a trading bot, DeFi product, wallet, liquidity manager, or blockchain integration demo. It is a decision intelligence environment where the agent's only job is to classify each window of market activity and choose the correct intervention.

---

## Table of Contents

- [Key Terminology](#key-terminology)
- [How It Works](#how-it-works)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Logic and Scoring](#reward-logic-and-scoring)
- [Tasks](#tasks)
- [Baseline Policies](#baseline-policies)
- [Runnable Files — What to Run and When](#runnable-files-what-to-run-and-when)
- [Environment Variables](#environment-variables)
- [Debug Telemetry](#debug-telemetry)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Quick Start](#quick-start)
- [Futuristic Roadmap](#futuristic-roadmap)

---

## Key Terminology

Understanding the following terms is essential to reading this codebase and interpreting its outputs.

### AMM (Automated Market Maker)

An AMM is a decentralized exchange mechanism that uses a mathematical formula to price assets instead of a traditional order book. TradeX simulates a **constant product AMM** (`x * y = k`), where `reserve_x` and `reserve_y` are the two token reserves in the pool. The price of the asset is determined by the ratio `reserve_y / reserve_x`. When a trade occurs, tokens are added to one side and removed from the other, keeping the product `k` constant. This is the same model used by protocols like Uniswap v2.

### Liquidity

Liquidity measures how much capital is available in the AMM pool to absorb trades without large price swings. In this simulation, liquidity is calculated as `2 * sqrt(k)` where `k = reserve_x * reserve_y`. Higher liquidity means the pool can handle larger trades with smaller price impact. **In the current version, liquidity is initialized at fixed reserves (1000 x, 100000 y) and evolves only through simulated trades — there is no real-time market data feed.**

### Slippage

Slippage is the difference between the expected price of a trade and the actual execution price. In a constant product AMM, every trade moves the price  larger trades cause more slippage. The field `recent_slippage_impact` in the observation space captures the magnitude of this price displacement across recent trades. High slippage on small trades is a red flag for manipulation, because it suggests the pool is being deliberately pushed to extreme price points.

### MEV (Maximal Extractable Value)

MEV refers to the profit that can be extracted by reordering, inserting, or censoring transactions within a block before it is finalized on-chain. Common MEV strategies include:

- **Sandwich attacks**: A bot sees a pending large trade, places a buy order before it (frontrun) and a sell order after it (backrun), profiting from the price movement the victim's trade causes.
- **Frontrunning**: A bot copies or front-runs a profitable pending transaction to capture the profit first.
- **JIT (Just-In-Time) liquidity**: A bot adds liquidity to a pool moments before a large trade to earn fees, then removes it immediately after.
- **Backrunning / arbitrage**: A bot places a trade immediately after a large price-moving transaction to capture the resulting arbitrage.

In TradeX, the simulated bots exhibit **MEV-like behavior patterns** — rapid bursts of trades, suspiciously regular timing intervals, and coordinated size signatures that the surveillance agent must learn to detect.

### Bot Confidence

An internal simulation parameter (`bot_confidence`, range 0.0–1.0) that controls how aggressively the simulated bot trades in a given episode. Higher bot confidence means more frequent suspicious activity, larger trade sizes, and stronger signal indicators. This value evolves dynamically: successful `BLOCK` actions reduce it (the bot "backs off"), while missed detections (`ALLOW` on suspicious activity) increase it (the bot "gets bolder"). It also controls a **stealth mechanic** — as the episode progresses, bots learn to suppress their visible signal footprint, making later steps harder to classify.

### Burst Indicator

A float (0.0–1.0) representing how much the current trading window looks like an acute, high-frequency burst many trades crammed into a short time window. High burst values suggest a bot hammering the pool rapidly, typical of sandwich attacks or aggressive frontrunning.

### Pattern Indicator

A float (0.0–1.0) representing how rhythmic or coordinated the recent trading pattern looks. High pattern values suggest repeated timing intervals and size signatures that are unlikely to occur organically characteristic of algorithmic manipulation bots operating on a fixed schedule.

### Suspiciousness Score

A composite float (0.0–1.0) combining multiple signals into a single overall suspicion metric. It blends burst activity, pattern regularity, and bot confidence into one number. It is the broadest signal available to the agent.

### Manipulation Score

A float (0.0–1.0) that specifically captures how much the current activity looks like deliberate market manipulation (as opposed to just being noisy or unusual). It correlates strongly with the pattern indicator and is the highest-confidence signal for coordinated attacks.

---

## How It Works

1. **The AMM pool initializes** with fixed reserves (`reserve_x=1000`, `reserve_y=100000`) and a task-specific `bot_confidence` level.
2. **Each step**, the environment procedurally generates a trading window: either organic normal trades or bot-driven suspicious trades, depending on a probabilistic roll against `bot_confidence`.
3. **The agent receives a structured observation** containing 16 surveillance signals (price, liquidity, trade stats, time gaps, burst/pattern/suspicion/manipulation indicators).
4. **The agent chooses an action**: `ALLOW`, `FLAG`, `BLOCK`, or `MONITOR`.
5. **The environment updates**: the AMM state evolves (reserves shift from simulated trades), `bot_confidence` adjusts based on whether the agent's action was correct, and the next observation is generated.
6. **After all steps**, the episode is scored across five weighted dimensions.

The key challenge is that **bots adapt**: successful blocks reduce bot confidence (fewer future attacks), but missed detections embolden the bot. Additionally, normal market activity can produce **noise spikes** (14-15% chance per step) that mimic suspicious signals, creating a false-positive trap.

---

## Observation Space

Each step returns a fixed size structured observation with these surveillance signals:

| Field | Type | Description |
|---|---|---|
| `current_amm_price` | float | Current price ratio `reserve_y / reserve_x` |
| `liquidity_snapshot` | float | Pool liquidity `2 * sqrt(k)` |
| `recent_trade_count` | int | Number of trades in the current window |
| `trades_in_window` | list[float] | Last 5 trade sizes |
| `trade_frequency` | float | Trades per unit time in the window |
| `average_trade_size` | float | Mean trade size in the window |
| `maximum_trade_size` | float | Largest single trade in the window |
| `recent_slippage_impact` | float | Aggregate price displacement from recent trades |
| `time_gap_mean` | float | Average time between consecutive trades |
| `time_gap_min` | float | Shortest time gap (low = rapid-fire bot activity) |
| `recent_time_gaps` | list[float] | Last 5 inter-trade time intervals |
| `recent_price_impacts` | list[float] | Last 5 per-trade price impact values |
| `burst_indicator` | float | Acute high-frequency burst score (0.0–1.0) |
| `pattern_indicator` | float | Rhythmic coordination score (0.0–1.0) |
| `suspiciousness_score` | float | Composite suspicion metric (0.0–1.0) |
| `manipulation_score` | float | Deliberate manipulation confidence (0.0–1.0) |

---

## Action Space

The agent must choose exactly one of four actions per step:

| Action | Meaning | When to Use |
|---|---|---|
| `ALLOW` | Let the activity pass as normal | Organic, healthy trading no red flags |
| `MONITOR` | Watch more closely without intervening | Mildly elevated signals not enough to escalate |
| `FLAG` | Mark as suspicious for review | Clear suspicious indicators but not extreme |
| `BLOCK` | Stop the activity strongly suspicious or harmful | High-confidence manipulation or burst attack |

Legacy trading and liquidity-management actions have been removed from the environment.

---

## Reward Logic and Scoring

Reward is shaped for **partial credit** — the agent does not need a perfect binary classifier to score well.

**On suspicious activity:**
| Action | Credit |
|---|---|
| `BLOCK` | 1.00 (full detection) |
| `FLAG` | 0.82 |
| `MONITOR` | 0.55 |
| `ALLOW` | 0.05 (miss — also counts as false negative) |

**On normal activity:**
| Action | Effect |
|---|---|
| `ALLOW` | +1.0 market health |
| `MONITOR` | +0.92 market health |
| `FLAG` | +0.72 market health, counts as false positive |
| `BLOCK` | +0.45 market health, counts as false positive + overblock |

### Final Episode Grade Weights

| Component | Weight | What It Measures |
|---|---|---|
| **Detection** | 50% | How well the agent caught suspicious activity |
| **False Positive** | 20% | How well the agent avoided flagging/blocking normal activity |
| **False Negative** | 15% | How well the agent avoided missing suspicious activity |
| **Health** | 10% | How well the agent preserved healthy market participation |
| **Overblocking** | 5% | How well the agent avoided blocking normal users |

A score of `>= 0.6` is considered a passing episode.

---

## Tasks

The benchmark includes three deterministic tasks with escalating difficulty. **Running `python inference.py` executes all three tasks sequentially**, producing `[START]`, `[STEP]`, and `[END]` stdout logs for each task.

### `burst_detection` 

The agent faces sudden, acute spikes — a bot hammering the pool with high volume trades in a short window. The primary signal is `burst_indicator` going high. Activity is concentrated in time. A simple threshold rule can mostly catch it. Initial bot confidence: **0.25**.

### `pattern_manipulation_detection`

The agent faces sustained, rhythmic coordination trades that aren't necessarily bursty but repeat at suspicious timing intervals or with suspicious size signatures. The signal is `pattern_indicator` combined with `time_gap_min` being suspiciously regular. A bot can have low burst but high pattern. This is harder because individually each trade looks benign. Initial bot confidence: **0.35**.

### `full_market_surveillance`

Both threats at once, mixed with normal traffic. The agent must simultaneously avoid false positives on organic noise and catch both burst type and pattern type attacks. Initial bot confidence: **0.30**. This task produces 60 `[STEP]` logs.

---

## Baseline Policies

### LLM Policy (Official Baseline)

The primary baseline in `inference.py`. It sends each observation as a JSON payload to an LLM (via OpenAI-compatible API) with a structured system prompt containing surveillance rules. The LLM returns a JSON action. If the LLM call fails or returns an invalid action, it falls back to the heuristic policy.

### Heuristic Policy (Fallback)

A simple threshold based rule engine in `meverse/baseline_policy.py`:

- `pattern_indicator >= 0.72` AND `slippage >= 0.055` → `BLOCK`
- `manipulation_score >= 0.78` → `BLOCK`
- `burst_indicator >= 0.70` OR `trade_frequency >= 7.5` → `FLAG`
- `suspiciousness_score >= 0.52` → `MONITOR`
- Otherwise → `ALLOW`

This exists for crash recovery and as a benchmark floor.

### Dashboard Comparison Baselines

These are edge cases that we should cover in our policy not only the heuristic policy.
- **Always Allow**: Returns `ALLOW` every step. Sanity-check lower bound.
- **Random**: Picks randomly from all four actions. Noise baseline.

---

## Runnable Files — What to Run and When

### `inference.py` — Run the official benchmark

The competition entrypoint. Runs all three tasks sequentially (burst_detection, pattern_manipulation_detection, full_market_surveillance) and prints structured `[START]`, `[STEP]`, `[END]` logs to stdout for each task.

```bash
# Run all three tasks
python inference.py

# With debug telemetry
DEBUG_TELEMETRY=1 python inference.py

# Full competition mode
EVAL_MODE=true DEMO_MODE=false python inference.py
```

### `dashboard.py` — Visual debugging and policy comparison

Interactive Gradio UI for running episodes, comparing baselines side-by-side, and replaying telemetry files.

```bash
pip install gradio plotly numpy   # if not already installed
python dashboard.py
# Open http://127.0.0.1:7860 or use the printed Gradio share link
```

The dashboard provides three tabs:
- **Episode Runner**: Run a single episode with Heuristic, Always Allow, or Random
- **Policy Comparison**: Compare all three baselines on the same seeded episode
- **Telemetry Viewer**: Upload a JSONL telemetry file and replay rewards visually

### `app.py` — OpenEnv server / HF Spaces entrypoint

Serves the OpenEnv FastAPI app locally or as a Hugging Face Space (auto-detected). In Space mode, it wraps both the OpenEnv Playground and the Dashboard into a tabbed Gradio interface.

```bash
python app.py
```

### `compare_policies.py` — Benchmark depth analysis

Compares the heuristic policy against the LLM baseline across all tasks. Used to verify that the environment has headroom beyond simple thresholds. **Requires `HF_TOKEN`.**

```bash
python compare_policies.py
```

### `python -m meverse.validation` — Score validation suite

Runs every task with the heuristic policy and asserts all scores fall within `[0.0, 1.0]`. Quick smoke test for environment integrity.

```bash
python -m meverse.validation
```

### `openenv validate` — OpenEnv package validation

Validates the environment package against the OpenEnv spec.

```bash
openenv validate
```

### Viewing Logs

Inference logs follow a structured format on stdout:

```
[START] task=burst_detection env=amm-market-surveillance model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=ALLOW reward=0.85 done=false error=null
[STEP] step=2 action=BLOCK reward=1.00 done=false error=null
...
[END] success=true steps=50 score=0.9997 rewards=0.85,1.00,...

[START] task=pattern_manipulation_detection env=amm-market-surveillance model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=50 score=0.6041 rewards=...

[START] task=full_market_surveillance env=amm-market-surveillance model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=60 score=0.6185 rewards=...
```

For richer step by step data (observations, hidden labels, AMM state transitions), enable debug telemetry:

```bash
DEBUG_TELEMETRY=1 DEBUG_TELEMETRY_PATH=telemetry/debug.jsonl python inference.py
```

Telemetry files are JSONL and can be replayed in the dashboard's Telemetry Viewer tab.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | For LLM | `https://router.huggingface.co/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | For LLM | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for the LLM policy |
| `HF_TOKEN` | For LLM | — | Hugging Face API token. Without this, the run falls back to heuristic. |
| `MEVERSE_TASK` | No | all tasks | When set, runs only the specified task (also accepts `TASK_NAME`). By default, all three tasks run sequentially. |
| `EVAL_MODE` | No | `true` | Fixed-seed deterministic mode for reproducible scores |
| `DEMO_MODE` | No | `false` | Adds bounded variation for local exploration. **Overrides EVAL_MODE.** |
| `DEBUG_TELEMETRY` | No | `false` | Writes step-by-step JSONL telemetry to disk |
| `DEBUG_TELEMETRY_PATH` | No | `telemetry/<task>-<timestamp>.jsonl` | Custom telemetry output path |

---

## Debug Telemetry

When `DEBUG_TELEMETRY=1`, each run writes a JSONL file capturing:

- The observation seen by the policy **before** each action
- The observation returned **after** each action
- Hidden environment labels and scenario metadata for the active step
- AMM state transitions (`bot_confidence`, `volatility`, `health_index`)
- The final episode grade breakdown

This data is essential for diagnosing policy failures, understanding why the agent made a particular decision, and verifying that the simulation is behaving correctly.

---

## Project Structure

```
TradeX1/
├── inference.py              # Competition entrypoint — LLM surveillance policy
├── dashboard.py              # Gradio UI — episode runner, comparison, telemetry viewer
├── app.py                    # OpenEnv server / HF Spaces entrypoint
├── compare_policies.py       # LLM vs heuristic benchmark analysis
├── openenv.yaml              # OpenEnv metadata and task definitions
├── Dockerfile                # Multi-stage Docker build (uv-based)
├── requirements.txt          # Root-level Python dependencies
├── validate_submission.sh    # Environment validation script
├── telemetry/                # Debug telemetry output directory
├── docs/                     # Detailed documentation
│   ├── Dashboard.md          # Dashboard visualizations reference
│   └── playground.md         # Playground interface reference
└── meverse/                  # Core environment package
    ├── __init__.py           # Package exports
    ├── amm.py                # Constant-product AMM state machine and procedural generation
    ├── models.py             # Pydantic models (SurveillanceAction, SurveillanceObservation)
    ├── tasks.py              # Task definitions, step generation, and scoring logic
    ├── baseline_policy.py    # Threshold-based heuristic fallback policy
    ├── policy.py             # LLM policy config, client builder, action selection
    ├── client.py             # Environment client wrapper
    ├── env.py                # .env file loader
    ├── validation.py         # Score validation suite
    └── server/               # FastAPI OpenEnv server implementation
        └── meverse_environment.py  # MarketSurveillanceEnvironment class
```

---

## Documentation

Detailed documentation for the UI components is available in the `docs/` folder:

- **[Dashboard.md](docs/Dashboard.md)** — Complete reference for all 10+ visualizations, charts, gauges, heatmaps, and data tables in the Gradio dashboard. Covers every element's purpose, data source, color coding, and interpretation guide.
- **[playground.md](docs/playground.md)** — Reference for the OpenEnv Playground interface, including action/observation models, data flow, HF Space integration, and comparison with the Dashboard.

---

## Quick Start

- **Entrypoint**: `python inference.py` — runs all three tasks in deterministic mode
- **Artifacts**: `Dockerfile` and `openenv.yaml` (root)
- **Required env vars**: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- **Mode**: `EVAL_MODE=true` and `DEMO_MODE=false` for fixed-seed reproducibility
- The **LLM-driven policy** is the primary baseline
- The **heuristic policy** is a crash fallback only
- `dashboard.py` provides visualization and debugging for single episodes

---

## Futuristic Roadmap

### End-to-End MEV Attack Replay

Currently, TradeX generates synthetic bot behavior through procedural simulation probabilistic rolls against `bot_confidence` produce burst and pattern signals that approximate real MEV strategies. The natural next step is to **replay actual on-chain MEV transactions** against a live or forked AMM state. By feeding real sandwich attacks, JIT liquidity events, and frontrunning sequences through the environment, the benchmark would shift from "detect synthetic bots" to "detect real adversarial MEV strategies as they appeared on mainnet." This would require integrating with an archive node or a mempool replay service and mapping raw transaction traces into the observation schema the agent already understands.

### Closing the RL Loop — From Evaluation to Training

Right now, the LLM agent is **evaluated but not trained** within this environment. The reward signal is computed and logged, but it never flows back into the model's weights. The logical next step is closing that loop using the per-step reward signal to fine-tune or RLHF-align the agent's surveillance policy, so the model itself improves from repeated exposure to the environment. This turns TradeX from a static benchmark into an active training harness where the agent's detection accuracy compounds over episodes. Integration with the MEV environment we have built would allow the agent to train against increasingly adversarial bot strategies in a curriculum-learning setup.

### Honest Design Constraints

It is important to be transparent about what the current version does **not** do. Liquidity is initialized at fixed reserves and evolves only through simulated trades there is no real time market data integration, no live price feeds, and no connection to on-chain state. Seeded determinism is used for reproducibility, which means that in `EVAL_MODE` every run produces the exact same observation sequence. Predictions are based entirely on simulation parameters, not actual market conditions. The bot behavior, while adaptive (stealth mechanics, confidence based intensity), is still procedurally generated and does not capture the full complexity of real-world MEV actors who coordinate across multiple pools, tokens, and protocols simultaneously.

### Learnable Reward Weights

The current scoring function uses hardcoded weights: Detection 50%, False Positive 20%, False Negative 15%, Health 10%, Overblocking 5%. These weights encode a specific value judgment about what matters most in surveillance. A future version could allow the **reward function itself to become a learned component** the agent (or a meta-learning loop) would propose and refine these weights based on trajectory outcomes, environmental feedback, and task-specific objectives. This moves toward reward-shaping as a first-class optimization target rather than a fixed design decision.

### The Broader Vision — MEV Surveillance as a Benchmark Domain

MEV surveillance sits at the intersection of **adversarial machine learning**, **DeFi security**, and **agent evaluation** three fields that are individually well-studied but rarely combined into a single benchmark. No standardized OpenEnv benchmark currently exists for this problem class. TradeX is a step toward establishing MEV detection as a repeatable, scorable, and comparable evaluation domain where different agent architectures (LLMs, RL policies, hybrid systems) can be measured against the same adversarial scenarios. The long-term goal is a benchmark suite where the environment, the attacker, and the defender all co-evolve producing increasingly realistic and challenging surveillance problems that push the frontier of what AI agents can detect in decentralized financial systems.
