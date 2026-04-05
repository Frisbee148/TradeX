# Bot-aware Market Surveillance in Simulated AMM Trading

This project is a simulation environment for reinforcement learning and decision intelligence. It is not a trading bot, DeFi product, wallet, liquidity manager, or blockchain integration demo.

The environment models an AMM-style market and asks an agent to act as a market surveillance controller. At each step, the agent reviews structured signals about recent trading activity and chooses one of four responses:

- `ALLOW`
- `FLAG`
- `BLOCK`
- `MONITOR`

The goal is to identify suspicious bot-like behavior while minimizing harm to normal users and preserving healthy market behavior.

## For Judges

This repository is intended to be evaluated in deterministic competition mode.

- Use the root [inference.py](/home/casp1an/Code/TradeX1/inference.py) as the submission entrypoint.
- Use the repo-root [Dockerfile](/home/casp1an/Code/TradeX1/Dockerfile) and repo-root [openenv.yaml](/home/casp1an/Code/TradeX1/openenv.yaml) as the primary submission artifacts.
- Provide the required model environment variables: `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.
- Run with `EVAL_MODE=true` and `DEMO_MODE=false` so the environment uses a fixed seed and produces reproducible scores.
- Treat the LLM-driven policy as the official baseline path.
- Treat the heuristic policy only as a crash fallback and internal testing aid, not as the intended submitted baseline.
- Treat [dashboard.py](/home/casp1an/Code/TradeX1/dashboard.py) as an optional visualization and debugging UI, not as the official judging entrypoint.
- To inspect behavior locally, run `python dashboard.py` from the repo root and open the Gradio UI on port `7860`. The launch config enables `share=True`, so Gradio will also print a public share link when tunneling is available.

This matches the project’s intended hackathon evaluation flow: a single inference entrypoint, deterministic task execution, and programmatic grading across all tasks.

## What The Benchmark Measures

This benchmark is designed as a real-world surveillance and anomaly-detection task:

- detect suspicious bursts of trading activity
- detect repeated manipulation patterns
- avoid false positives on normal activity
- avoid false negatives on harmful activity
- preserve healthy market participation

The benchmark is reward-shaped for partial progress. It does not optimize for profit.

## Observation Space

Each step returns a fixed-size structured observation with surveillance signals:

- `current_amm_price`
- `liquidity_snapshot`
- `recent_trade_count`
- `trades_in_window`
- `trade_frequency`
- `average_trade_size`
- `maximum_trade_size`
- `recent_slippage_impact`
- `time_gap_mean`
- `time_gap_min`
- `recent_time_gaps`
- `recent_price_impacts`
- `burst_indicator`
- `pattern_indicator`
- `suspiciousness_score`
- `manipulation_score`

## Action Space

Only these actions are valid:

- `ALLOW`
- `FLAG`
- `BLOCK`
- `MONITOR`

Legacy trading and liquidity-management actions have been removed from the environment logic.

Action meaning:

- `ALLOW`: treat the activity as normal and let it pass.
- `MONITOR`: keep the activity under closer watch without escalating to a hard intervention.
- `FLAG`: mark the activity as suspicious for review.
- `BLOCK`: stop the activity because it appears strongly suspicious or harmful.

## Reward Logic

Reward combines:

- positive reward for correctly detecting suspicious behavior
- positive reward for correctly allowing normal activity
- false positive penalties
- false negative penalties
- severity bonuses on harmful suspicious activity
- overblocking penalties to protect healthy market behavior

At a high level:

- on suspicious activity, `BLOCK` scores best, `FLAG` is strong, `MONITOR` gets partial credit, and `ALLOW` is usually a miss
- on normal activity, `ALLOW` scores best, `MONITOR` is a softer fallback, and `FLAG` or `BLOCK` hurt false-positive and market-health metrics

The final episode grade is weighted as:

- `Detection`: 50%
- `False Positive`: 20%
- `False Negative`: 15%
- `Health`: 10%
- `Overblocking`: 5%

## Tasks

The repo includes three deterministic tasks with distinct difficulty levels:

1. `burst_detection`
2. `pattern_manipulation_detection`
3. `full_market_surveillance`

- `burst_detection`: learn to catch abrupt high-frequency bursts.
- `pattern_manipulation_detection`: learn repeated timing and size signatures.
- `full_market_surveillance`: balance burst detection, pattern detection, and false-positive control in mixed traffic.

## Baseline Policies

The official competition baseline is the LLM-driven policy in [inference.py](/home/casp1an/Code/TradeX1/inference.py). It uses the OpenAI client together with the required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

The rule-based heuristic in [meverse/baseline_policy.py](/home/casp1an/Code/TradeX1/meverse/baseline_policy.py) is kept for two narrower jobs:

- crash fallback if the LLM call fails
- benchmark comparison to test whether the environment has headroom beyond simple thresholds

It is not the intended submitted baseline. It exists to support internal testing, benchmark analysis, and graceful degradation if the upstream model call fails during a run.

The heuristic policy follows simple surveillance rules:

- if pattern score is high and slippage is high, `BLOCK`
- elif manipulation score is high, `BLOCK`
- elif burst score or trade frequency is high, `FLAG`
- elif suspiciousness is moderate, `MONITOR`
- else `ALLOW`

The dashboard also exposes two simple comparison baselines:

- `Always Allow`: ignores the signals and returns `ALLOW` on every step.
- `Random`: ignores the signals and picks randomly from `ALLOW`, `FLAG`, `BLOCK`, and `MONITOR` on each step.

These are useful as sanity-check baselines for visualization, but they are not official submitted policies.

## Running The Environment

Serve the OpenEnv app from the repo root:

```bash
python app.py
```

Validate the environment package directly:

```bash
openenv validate
```

## Running Inference

The root inference runner is [inference.py](/home/casp1an/Code/TradeX1/inference.py). It loads the surveillance environment, runs the LLM baseline, and prints clean competition-style logs. If the upstream model call fails, it falls back to the heuristic policy so the run can still complete.

```bash
python inference.py
```

Optional task selection:

```powershell
$env:MEVERSE_TASK="full_market_surveillance"
python inference.py
```

## Running The Dashboard

The interactive dashboard lives at [dashboard.py](/home/casp1an/Code/TradeX1/dashboard.py). Run it from the repo root so the local `meverse` package imports resolve correctly.

If your environment does not already have the UI dependencies installed, install them first:

```bash
pip install gradio plotly numpy
```

Then launch the dashboard:

```bash
python dashboard.py
```

After startup:

- open `http://127.0.0.1:7860` for the local UI
- check the terminal output for the Gradio public share URL because the app launches with `share=True`

The dashboard provides:

- `Episode Runner`: run one episode with `Heuristic`, `Always Allow`, or `Random`
- `Policy Comparison`: compare those three baselines on the same seeded episode
- `Telemetry Viewer`: upload a JSONL telemetry file and replay rewards visually

## Required Environment Variables

`inference.py` reads these variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `EVAL_MODE`
- `DEMO_MODE`
- `DEBUG_TELEMETRY`
- `DEBUG_TELEMETRY_PATH`

Example PowerShell setup:

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="your-token"
python inference.py
```

If `HF_TOKEN` is not set, the script cannot use the primary LLM baseline and will label the run as `heuristic-fallback`. For submission, the required API variables should be present. The fallback exists to reduce crash risk, not as the intended submitted baseline.

Mode behavior:

- `EVAL_MODE=true` is the official judging mode. The environment resets with a fixed seed so the observation sequence is deterministic and scores are reproducible across runs.
- `DEMO_MODE=true` is an exploratory mode for local testing only. It introduces small bounded variation into the scenario so contributors can sanity-check that the policy is not overfitting one exact replay.
- `DEMO_MODE=true` takes precedence over `EVAL_MODE`, so it should not be used when reporting baseline scores or when preparing a competition submission.
- Judges should evaluate the project with `EVAL_MODE=true` and `DEMO_MODE=false`.
- `DEBUG_TELEMETRY=true` writes step-by-step JSONL telemetry to disk without changing the required `[START]`, `[STEP]`, and `[END]` stdout format.
- `DEBUG_TELEMETRY_PATH` overrides the telemetry output path. If not set, telemetry is written under `telemetry/<task>-<timestamp>.jsonl`.

Telemetry captures:

- the observation seen by the policy before each action
- the observation returned after each action
- hidden environment labels and scenario metadata for the active step
- AMM state transitions such as `bot_confidence`, `volatility`, and `health_index`
- the final episode grade breakdown

Example telemetry run:

```bash
DEBUG_TELEMETRY=1 DEBUG_TELEMETRY_PATH=telemetry/debug.jsonl python inference.py
```

## Validation And Graders

Validation logic is implemented in [meverse/validation.py](/home/casp1an/Code/TradeX1/meverse/validation.py). It:

- enumerates all tasks
- runs each task independently
- runs the deterministic grader independently
- prints task-wise scores
- asserts every score satisfies `0.0 <= score <= 1.0`

Run it with:

```bash
python -m meverse.validation
```

To compare the heuristic policy against the official LLM baseline and check whether the environment is too shallow, run:

```bash
python compare_policies.py
```

That script is analysis tooling for benchmark quality. It is not the competition inference entrypoint.

## Verifying Score Range

When you run the validation suite, each task prints a normalized score and the script asserts the range check `0.0 <= score <= 1.0`.

## OpenEnv Metadata

Project metadata for OpenEnv lives in [openenv.yaml](/home/casp1an/Code/TradeX1/openenv.yaml). It describes the repository as a market surveillance benchmark rather than a trading or liquidity-management environment.
