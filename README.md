# Bot-aware Market Surveillance in Simulated AMM Trading

This project is a simulation environment for reinforcement learning and decision intelligence. It is not a trading bot, DeFi product, wallet, liquidity manager, or blockchain integration demo.

The environment models an AMM-style market and asks an agent to act as a market surveillance controller. At each step, the agent reviews structured signals about recent trading activity and chooses one of four responses:

- `ALLOW`
- `FLAG`
- `BLOCK`
- `MONITOR`

The goal is to identify suspicious bot-like behavior while minimizing harm to normal users and preserving healthy market behavior.

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

## Reward Logic

Reward combines:

- positive reward for correctly detecting suspicious behavior
- positive reward for correctly allowing normal activity
- false positive penalties
- false negative penalties
- severity bonuses on harmful suspicious activity
- overblocking penalties to protect healthy market behavior

## Tasks

The repo includes three deterministic tasks with distinct difficulty levels:

1. `burst_detection`
2. `pattern_manipulation_detection`
3. `full_market_surveillance`

- `burst_detection`: learn to catch abrupt high-frequency bursts.
- `pattern_manipulation_detection`: learn repeated timing and size signatures.
- `full_market_surveillance`: balance burst detection, pattern detection, and false-positive control in mixed traffic.

## Baseline Policy

The baseline policy follows simple surveillance rules:

- if pattern score is high and slippage is high, `BLOCK`
- elif burst score or trade frequency is high, `FLAG`
- elif suspiciousness is moderate, `MONITOR`
- else `ALLOW`

Implementation lives in [meverse/baseline_policy.py](/d:/TradeX/meverse/baseline_policy.py).

## Why The Baseline Beats Edge Cases

This benchmark is designed so fixed one-action policies perform poorly even if they look safe in one dimension.

- An always-`ALLOW` policy preserves healthy traffic, but misses suspicious behavior and creates many false negatives.
- An always-`BLOCK` policy catches suspicious behavior, but harms normal users and creates many false positives and overblocking penalties.
- An always-`FLAG` policy is less severe than blocking, but still reacts too aggressively to normal activity.
- An always-`MONITOR` policy is gentler, but still leaves severe suspicious behavior under-addressed.

The baseline performs better because it adapts its response to the observed surveillance signals. It allows clearly normal activity, flags bursty elevated activity, monitors borderline cases, and blocks only the strongest manipulation patterns. That balance is exactly what the grader rewards.

## Why Adaptive Policy Matters

Real surveillance is not solved by one fixed action. The same response should not be used for every market condition.

- Normal market flow should usually be allowed.
- Bursty but not fully proven suspicious behavior may deserve a flag rather than a block.
- Borderline cases often benefit from monitoring instead of immediate escalation.
- Strong manipulation patterns should be blocked decisively.

This is why the benchmark emphasizes adaptive decision-making instead of a single hard-coded extreme. The agent is rewarded for matching the intensity of its response to the severity of the observed market behavior while preserving healthy market participation.

## Benchmark Results

The table below shows deterministic eval-mode scores for several fixed edge-case policies compared with the adaptive baseline:

| Policy | burst | pattern | full |
|---|---:|---:|---:|
| Always `ALLOW` | 0.3707 | 0.3705 | 0.3692 |
| Always `BLOCK` | 0.6931 | 0.6930 | 0.6924 |
| Always `FLAG` | 0.6789 | 0.6788 | 0.6778 |
| Always `MONITOR` | 0.7631 | 0.7629 | 0.7617 |
| Baseline (adaptive) | 0.9057 | 0.9764 | 0.9788 |

These results make the benchmark design visible at a glance:

- the fixed edge-case policies each do well on one dimension but fail badly on others
- the adaptive baseline is consistently stronger because it balances detection with false-positive control and healthy market preservation
- the biggest gap appears on the harder pattern and full-surveillance tasks, where a one-action strategy is especially brittle

## Real-World Relevance

This environment is intentionally framed as a market surveillance and anomaly-detection benchmark, but the decision problem maps closely to real operational systems.

It simulates challenges faced by:

- centralized exchange risk and abuse monitoring teams
- on-chain MEV and transaction-monitoring systems
- fraud and anomaly-detection pipelines in digital marketplaces
- market integrity and surveillance groups at trading venues
- regulatory and compliance teams that need explainable escalation decisions

The core real-world tension is the same across those settings:

- react too slowly, and harmful behavior is allowed to continue
- react too aggressively, and normal users are harmed by false positives
- apply the same action everywhere, and the system becomes brittle

That is why this benchmark does not reward a single extreme strategy. It rewards systems that can interpret changing signals, calibrate the severity of their response, and preserve healthy market behavior while still catching adversarial activity.

In practical terms, the benchmark reflects real decision-making tradeoffs around:

- security versus user experience
- sensitivity versus precision
- fast escalation versus overblocking
- explainable rules versus adaptive behavior

The goal is not just to detect suspicious activity in the abstract. The goal is to benchmark decision-making systems that must make context-sensitive interventions under uncertainty, which is exactly the kind of problem faced by modern surveillance, trust, and market safety systems.

## Running The Environment

Serve the OpenEnv app from the repo root:

```bash
python app.py
```

Validate the environment package directly:

```bash
cd meverse
openenv validate
```

## Running Inference

The root inference runner is [inference.py](/d:/TradeX/inference.py). It loads the surveillance environment, runs a baseline or OpenAI-backed policy, and prints clean competition-style logs.

```bash
python inference.py
```

Optional task selection:

```powershell
$env:MEVERSE_TASK="full_market_surveillance"
python inference.py
```

## Required Environment Variables

`inference.py` reads these variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `EVAL_MODE`
- `DEMO_MODE`

Example PowerShell setup:

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="your-token"
python inference.py
```

If `HF_TOKEN` is not set or the model call fails, the script falls back to a deterministic local baseline.

Mode behavior:

- `EVAL_MODE=true` keeps runs reproducible for competition-style evaluation.
- `DEMO_MODE=true` uses the episode seed to introduce small bounded observation variation for manual testing.
- If `DEMO_MODE=true`, it takes precedence over `EVAL_MODE`.

## Validation And Graders

Validation logic is implemented in [meverse/validation.py](/d:/TradeX/meverse/validation.py). It:

- enumerates all tasks
- runs each task independently
- runs the deterministic grader independently
- prints task-wise scores
- asserts every score satisfies `0.0 <= score <= 1.0`

Run it with:

```bash
python -m meverse.validation
```

## Verifying Score Range

When you run the validation suite, each task prints a normalized score and the script asserts the range check `0.0 <= score <= 1.0`.

## OpenEnv Metadata

Project metadata for OpenEnv lives in [meverse/openenv.yaml](/d:/TradeX/meverse/openenv.yaml). It now describes the repository as a market surveillance benchmark rather than a trading or liquidity-management environment.
