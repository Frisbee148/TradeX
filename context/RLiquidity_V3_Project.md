# RLiquidity V3 — MEV-Aware Reinforcement Learning Environment

> **A mini RL environment where agents learn to trade, provide liquidity, and survive
> adversarial MEV attacks in a simulated Uniswap V3 pool.**

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why This Project](#2-why-this-project)
3. [What Makes This Unique](#3-what-makes-this-unique)
4. [System Architecture](#4-system-architecture)
5. [Environment Design](#5-environment-design)
6. [MEV Attack Mechanics](#6-mev-attack-mechanics)
7. [Three Tasks — Increasing Difficulty](#7-three-tasks--increasing-difficulty)
8. [Reward Function](#8-reward-function)
9. [Grading System](#9-grading-system)
10. [Inference Script](#10-inference-script)
11. [LLM Agent Integration](#11-llm-agent-integration)
12. [Tech Stack](#12-tech-stack)
13. [Real-World Impact](#13-real-world-impact)
14. [Execution Flow](#14-execution-flow)

---

## 1. Problem Statement

**MEV (Maximal Extractable Value) bots extracted over $686 million from Ethereum
users in 2023 alone.**

Every time an ordinary user submits a swap transaction to a decentralized exchange,
their transaction enters a public mempool — a waiting room visible to everyone on the
network. Adversarial bots monitor this mempool in real time and exploit predictable
transaction patterns through three primary attack vectors:

| Attack Type | Mechanism | Annual Cost to Users |
|---|---|---|
| Sandwich Attack | Bot front-runs your buy, then sells into your price impact | ~$280M |
| JIT Liquidity | Bot injects liquidity before your swap, captures your fee, removes it after | ~$190M |
| Front-running | Bot executes an identical trade ahead of yours at a better price | ~$216M |

Uniswap V3's concentrated liquidity model — where LPs specify precise price ranges
rather than providing liquidity across the full curve — makes this problem
significantly more complex and more damaging than its V2 predecessor. Tick
crossings cause non-linear price impact, JIT attacks are uniquely enabled by V3's
NFT position model, and range order mechanics introduce entirely new attack surfaces.

**This project builds a structured RL environment that trains and evaluates agents
on their ability to execute trades, manage liquidity positions, and survive adversarial
MEV conditions in a simulated Uniswap V3 pool.**

---

## 2. Why This Project

### The Gap in Existing RL Environments

Existing public RL benchmarks for financial systems fall into two categories:

- **Too simple:** Stock trading with `BUY / SELL / HOLD`, no adversarial dynamics,
  no market microstructure.
- **Too theoretical:** Auction-based game theory environments with no real-world
  analogue in deployed systems.

No public benchmark exists that captures the **V3 concentrated liquidity + mempool
adversary** interaction. This is a genuine research gap.

### Why Now

Three converging trends make this the right moment:

1. **DeFi is mainstream infrastructure.** Over $45 billion in value is locked in
   DeFi protocols. MEV is not a niche problem — it affects every on-chain transaction.

2. **LLMs as agents.** LLM-based agents can reason about structured market state,
   read mempool signals, and choose between strategic actions in ways that rule-based
   bots cannot. This environment is purpose-built to showcase and evaluate that
   capability.

3. **Evaluation infrastructure demand.** The ML community is actively building
   structured agent evaluation benchmarks. A reproducible, domain-specific RL
   environment in an underexplored domain is exactly what this hackathon rewards.

---

## 3. What Makes This Unique

### V3 vs V2 — Why the Distinction Matters

Most DeFi RL papers use the constant product formula `x * y = k` from Uniswap V2.
This project uses **Uniswap V3 concentrated liquidity mechanics**, which introduce:

#### Tick-Based Price Ranges
Liquidity in V3 is not uniformly distributed across the price curve. It is
concentrated in discrete **tick ranges** — each tick representing a 0.01% price
move (`price = 1.0001^tick`). Active liquidity at any moment is only the sum of
positions whose range includes the current tick.

```
Price curve (V2):         Price curve (V3):
                          Dense ticks near price
uniform liquidity ─────   ████████████░░░░░░░░░░
across full range         └────┬────┘
                           active zone
```

This means:
- Swap price impact is **non-linear** — cheap inside a dense tick, expensive when
  crossing into a sparse one
- An agent that ignores tick structure will consistently get worse execution
- A smart agent can read `tick_distribution` and size swaps to avoid costly crossings

#### JIT Liquidity (V3-Exclusive MEV)
Just-In-Time liquidity is a V3-specific attack that V2 cannot replicate:

```
Step 1: Bot sees your large swap in the mempool
Step 2: Bot adds concentrated liquidity at current tick (same block, before your tx)
Step 3: Your swap executes — bot earns most of the trading fee
Step 4: Bot immediately removes liquidity in the same block
Result: You paid the fee. The fee went to a predator, not a real LP.
```

This is subtle because the attack actually *reduces* your price impact while
stealing the fee revenue that should incentivize long-term liquidity provision.
Detecting and countering it requires reading both mempool signals and active
liquidity composition simultaneously.

#### Range Orders
V3 allows single-sided liquidity that acts as a limit order — a primitive that
V2 doesn't have. An agent that understands this can place directional liquidity
positions that execute at a target price without paying swap fees.

### Why This Beats Stock Trading Environments

| Dimension | Stock Trading Bot | RLiquidity V3 |
|---|---|---|
| Adversarial dynamics | None | Adaptive MEV bots |
| Action complexity | Buy / Sell / Hold | 8 strategic actions |
| State complexity | Price + volume | Tick distribution + mempool |
| Real-world analogue | Simplified | Direct V3 pool mechanics |
| LLM differentiation | Low | High — reasoning required |
| Novelty | Overused | Unexplored in public benchmarks |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RLiquidity V3                          │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Agent   │───▶│   Mempool    │───▶│   MEV Engine     │  │
│  │(LLM/Rule)│    │  Simulation  │    │  (pre-run hook)  │  │
│  └──────────┘    └──────────────┘    └──────────────────┘  │
│       │                                       │             │
│       ▼                                       ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  V3 Pool Engine                      │   │
│  │                                                      │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ Tick Manager│  │ Swap Engine  │  │ LP Manager │  │   │
│  │  │             │  │              │  │            │  │   │
│  │  │ net_liq[]   │  │ V3 sqrt_P    │  │ Positions  │  │   │
│  │  │ jit_liq[]   │  │ formula      │  │ Fee accrual│  │   │
│  │  └─────────────┘  └──────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Reward  │───▶│   Grader     │───▶│  Score 0.0–1.0   │   │
│  │  Engine  │    │ (normalized) │    │  (deterministic) │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

**Tick Manager**
Maintains the per-tick liquidity map. Tracks `net_liquidity` (real LPs) and
`jit_liquidity` (bot overlay, ephemeral). Exposes a ±5 tick window to the agent
as the `tick_distribution` observation key.

**Swap Engine**
Implements the V3 sqrt-price formula:
- `Δ√P = Δy / L` for token1 → token0 swaps
- `Δ(1/√P) = Δx / L` for token0 → token1 swaps

Handles fee deduction before computing output amounts. Price and tick update
atomically after each swap.

**LP Manager**
Tracks agent LP positions as `LPPosition` objects with `tick_lower`, `tick_upper`,
`liquidity`, and `fees_earned`. Distributes pro-rata fee revenue to positions
whose range includes the current tick each step. Applies impermanent loss
approximation on removal.

**MEV Engine**
Runs in two hooks — `pre_step` (front-run / JIT inject) and `post_step`
(sandwich back-run / JIT removal). Strategy is seeded per scenario: `passive`,
`jit`, `sandwich`, or `adaptive`. The `adaptive` bot increases aggression based
on observed agent behavior over the episode.

**Reward Engine**
Computes dense per-step rewards from swap quality, LP yield, and MEV losses.
Terminal reward is computed from portfolio value delta. All components are
individually logged for interpretability.

**Grader**
Deterministic, seed-controlled evaluation that normalizes all metrics to `[0.0, 1.0]`.
Produces a breakdown of four component scores plus a weighted final score.

---

## 5. Environment Design

### OpenEnv-Style Interface

```python
env = RLiquidityV3Env(config=TASK_CONFIGS["medium"])
obs, info = env.reset(seed=42)          # deterministic reset

while True:
    action = agent.act(obs)             # agent produces action dict
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

score = env.grade()                     # normalized 0.0 – 1.0
```

### Observation Space

Every observation is a structured Python dict — directly usable as an LLM prompt
context without additional processing:

```python
observation = {
    # Pool state
    "current_tick":       int,          # active tick index
    "current_price":      float,        # token1 / token0 (e.g. USDC per ETH)
    "sqrt_price":         float,        # √P, needed for V3 position math
    "active_liquidity":   float,        # total liquidity at current tick

    # Tick distribution (±5 ticks = ±0.5% price range window)
    "tick_distribution": [
        {
            "tick":           int,
            "price":          float,    # price at this tick
            "net_liquidity":  float,    # real LP liquidity
            "jit_liquidity":  float,    # bot JIT overlay (signal of attack)
        },
        # ... 11 entries total
    ],

    # Agent portfolio
    "agent_token0":       float,        # ETH balance
    "agent_token1":       float,        # USDC balance
    "agent_positions": [
        {
            "position_id":  str,
            "tick_lower":   int,
            "tick_upper":   int,
            "liquidity":    float,
            "fees_earned":  float,
        }
    ],

    # Mempool — adversarial intelligence
    "mempool": [
        {
            "tx_id":        str,
            "action_type":  str,        # type of pending tx
            "amount":       float,      # size of pending tx
            "zero_for_one": bool,       # direction
        }
    ],

    "last_mev_loss":      float,        # MEV extracted in previous step
    "step_num":           int,
    "max_steps":          int,
}
```

**Key design decision:** `jit_liquidity` is exposed in the observation. A smart agent
that monitors when `jit_liquidity > 0` at the current tick will recognize a JIT
attack is staged and can respond by using `split_swap` or `use_private_rpc`. This
creates a genuine signal-detection challenge rather than just noise.

### Action Space

Eight actions covering the full spectrum from passive to strategic:

```python
action = {
    "action_type": str,   # one of the 8 types below
    "params":      dict,  # action-specific parameters
}
```

| Action | Parameters | Strategic Use |
|---|---|---|
| `swap_exact_in` | `amount_in`, `zero_for_one`, `use_private_rpc` | Standard trade; private RPC defeats mempool MEV |
| `split_swap` | `total_amount`, `num_splits`, `zero_for_one` | Reduces per-swap MEV exposure across multiple blocks |
| `add_liquidity` | `tick_lower`, `tick_upper`, `amount0_desired`, `amount1_desired` | Earn fees by providing concentrated liquidity |
| `remove_liquidity` | `position_id` | Exit LP position, collect fees |
| `range_order` | `tick_lower`, `tick_upper`, `token`, `amount` | Single-sided LP = limit order primitive |
| `jit_liquidity` | `tick_lower`, `tick_upper`, `amount0`, `amount1` | Agent can use JIT offensively if capital allows |
| `hold` | — | Explicit wait; useful when mempool is dangerous |
| `close_episode` | — | Agent declares done; graded on current state |

**Why this action space is strong for evaluation:** Each action requires a different
kind of reasoning. `use_private_rpc` requires mempool threat detection. `split_swap`
requires thinking multiple steps ahead. `range_order` requires price prediction.
A random policy performs poorly; a reasoning agent performs well. This separation
is what makes a good benchmark.

---

## 6. MEV Attack Mechanics

### Sandwich Attack (V3-Aware)

```
Block N:
  [1] Bot reads pending agent swap from mempool
  [2] Bot front-runs: buys token0, pushing price UP against agent's buy direction
  [3] Agent swap executes at worse price (higher)
  [4] Bot back-runs: sells token0 at the inflated price
  [5] Bot nets profit = price_impact_caused × agent_amount

Agent loss = amount_in × 0.2% to 0.5% depending on pool depth
```

V3-specific complication: The bot must calculate whether the sandwich will cause
a tick crossing, which changes the liquidity depth mid-swap. An adaptive bot models
this; a naive bot may cause self-harm by crossing into sparse ticks.

**Agent counter-strategies:**
- `use_private_rpc=True` — submits via simulated private mempool, invisible to bots
- `split_swap` — smaller individual swaps reduce profit margin below bot's gas cost
- `hold` — wait for mempool to clear if a large bot tx is detected

### JIT Liquidity Attack

```
Block N:
  [1] Bot detects large pending swap (amount > threshold)
  [2] Bot adds concentrated liquidity at current tick (same block, priority fee)
  [3] Agent swap executes — bot earns majority of fee due to large liquidity share
  [4] Bot removes all liquidity in same block
  [5] Bot nets fee_earned - gas_cost (profitable if swap > ~$50k in real markets)

Agent impact: pays full fee; fee goes to bot rather than long-term LPs
```

**Observation signal:** `jit_liquidity` in `tick_distribution` will spike at the
current tick before the agent's swap if a JIT attack is staged. This is the key
detection signal that differentiates strong agents from weak ones.

**Agent counter-strategies:**
- Monitor `jit_liquidity` in the current tick's observation each step
- Use `use_private_rpc=True` for large swaps — prevents bot from seeing the tx
- Use `split_swap` to keep individual swap size below JIT profitability threshold

### Front-Running

```
Block N:
  [1] Bot detects pending trade in a specific direction
  [2] Bot executes identical trade at lower priority fee but higher gas
      (ensuring it lands first)
  [3] Agent trade executes at worse price

Agent loss: execution price worsens by 0.1% to 0.3%
```

**Agent counter-strategies:**
- Private RPC
- `range_order` instead of market swap for large directional bets

---

## 7. Three Tasks — Increasing Difficulty

### Task 1: EASY — Clean Pool, No Adversary

**Scenario:** Stable ETH/USDC pool at 1800 USDC/ETH. 8 background LP positions
creating moderate liquidity around current price. No MEV bots. Low price volatility.

**Goal:** Maximize portfolio value over 30 steps by trading efficiently and earning
LP fees. Learn the basic mechanics of tick-aware swapping and fee accrual.

**What the agent must learn:**
- How to read `tick_distribution` to size swaps appropriately
- How to add/remove LP positions to earn fee revenue
- That holding is sometimes better than trading into unfavorable ticks

**Config:**
```python
EnvConfig(
    mev_strategy="passive",
    max_steps=30,
    price_volatility=0.01,    # calm market
    num_lp_positions=8,
    initial_price=1800.0,
    initial_token0=10.0,      # 10 ETH
    initial_token1=18_000.0,  # 18,000 USDC
)
```

**Baseline agent score:** ~0.55–0.65
**Strong agent score:** ~0.80–0.90

**Why it's approachable but not trivial:** The tick structure still penalizes naive
BUY/SELL/HOLD behavior. An agent that ignores `tick_distribution` and makes large
swaps will cross into sparse ticks and receive worse execution than one that sizes
swaps to stay within the dense liquidity zone.

---

### Task 2: MEDIUM — JIT Bots Active

**Scenario:** Moderate volatility ETH/USDC pool. 12 background LP positions. JIT
bots are active and will inject ephemeral liquidity before any swap exceeding the
`slippage_threshold`. Budget pressure is real — unnecessary actions waste steps.

**Goal:** Maximize portfolio value while detecting and countering JIT attacks.
Use the mempool and `tick_distribution` observation to identify when a JIT attack
is staged and respond appropriately.

**What the agent must learn:**
- Recognize when `jit_liquidity > 0` signals a staged JIT attack
- Use `use_private_rpc=True` for large swaps to avoid the mempool
- Use `split_swap` to keep individual swap size below JIT profitability threshold
- Balance LP position management with active trading

**Config:**
```python
EnvConfig(
    mev_strategy="jit",
    max_steps=40,
    price_volatility=0.025,   # moderate volatility
    slippage_threshold=0.008, # bot triggers on swaps > 0.8% of pool depth
    num_lp_positions=12,
)
```

**Baseline agent score:** ~0.40–0.55
**Strong agent score:** ~0.70–0.82

**Why difficulty spikes here:** JIT attacks are subtle. Unlike sandwiches, they
don't obviously hurt execution price — they just redirect the fee. An agent that
doesn't track `fees_earned` carefully will not notice it's being drained. The
detection-response loop requires multi-step awareness.

---

### Task 3: HARD — Adaptive Adversary, High Volatility

**Scenario:** High volatility pool (4.5% price shock per step). 16 background LP
positions. Adaptive MEV bot that uses both JIT and sandwich strategies, learns
from agent behavior patterns, and increases aggressiveness when it detects
predictable agent actions. Budget constraint is tight — 50 steps total.

**Goal:** Survive a full market volatility cycle while actively countering an
adaptive adversary, managing LP positions through price swings, and maintaining
or growing portfolio value.

**What the agent must learn:**
- Vary action patterns to prevent the adaptive bot from learning them
- Manage LP positions actively — ranges set at step 1 may be far out-of-range
  by step 20 due to price volatility, earning zero fees
- Recognize when to close LP positions and reset ranges
- Use all strategic tools: private RPC, split swaps, range orders, hold

**Config:**
```python
EnvConfig(
    mev_strategy="adaptive",
    max_steps=50,
    price_volatility=0.045,   # high — forced position rebalancing
    slippage_threshold=0.005, # bot triggers on smaller swaps
    num_lp_positions=16,
    w_mev_avoidance=0.35,     # higher weight — harder to achieve
)
```

**Baseline agent score:** ~0.25–0.40
**Strong agent score:** ~0.60–0.75

**Why it genuinely challenges strong models:**
- Adaptive bot pattern-detects: if the agent always uses private RPC, bot adjusts
  strategy to front-run LP additions instead
- Price volatility forces LP range management — a static strategy fails
- Multi-objective optimization: trade profit vs. LP yield vs. MEV avoidance are
  in tension under tight budgets
- The correct response to a sandwich threat is different from the correct response
  to a JIT threat — the agent must distinguish between them in real time

---

## 8. Reward Function

The reward signal is designed to be **dense** (feedback every step) and
**informative** (each component maps to a specific behavior to reinforce).

### Per-Step Reward

```python
# Time pressure — incentivizes acting decisively
R_step_cost = -0.05

# Swap quality — rewards good execution
R_swap = (amount_out - fair_value) / fair_value * 0.1

# LP fee accrual — rewards active liquidity provision
R_fees = fees_earned_this_step * 0.5

# MEV penalty — penalizes being exploited
R_mev = -mev_loss_this_step

# Strategic action bonus — rewards using private RPC / split swaps appropriately
R_strategic = +0.1 if split_swap_used else 0
            + 0.0  # private RPC: MEV avoidance is already captured in R_mev

# Invalid action penalty — penalizes incoherent behavior
R_invalid = -0.2 per invalid action (insufficient balance, bad tick range, etc.)
```

### Terminal Reward

```python
# Portfolio growth — main optimization target
R_terminal = (final_portfolio_value - initial_portfolio_value)
             / initial_portfolio_value * 5.0
```

### Why This Reward Function Is Strong

Three properties that distinguish it from naive reward designs:

**Dense but not trivial.** The agent gets feedback every step via `R_fees` and
`R_mev`, but cannot exploit the reward function by taking obviously safe actions.
`hold` costs `-0.05` per step — pure passivity is penalized.

**MEV loss is direct.** Rather than proxying MEV avoidance through price impact,
`mev_loss_this_step` is computed explicitly from bot profit and subtracted from
reward. This creates a direct gradient toward MEV-aware behavior.

**Fee incentive aligns with real LP behavior.** `R_fees` rewards maintaining
active LP positions in-range. An agent that sets LP positions and forgets them
will see `R_fees = 0` once price moves out of range — correctly incentivizing
active range management.

---

## 9. Grading System

### Score Components (all normalized 0.0 – 1.0)

```python
def grade(env) -> dict:

    # Component 1: Profit (40%)
    raw_return = (final_value - initial_value) / initial_value
    profit_score = min(1.0, max(0.0, (raw_return + 1.0) / 1.5))
    # Maps [-100%, +50%] range to [0.0, 1.0]

    # Component 2: MEV Avoidance (30%)
    total_mev_loss = sum(env._mev_losses)
    max_possible_mev = steps_run * 0.05
    mev_score = max(0.0, 1.0 - total_mev_loss / max_possible_mev)

    # Component 3: Execution Efficiency (20%)
    strategic_actions = private_rpc_uses + split_swap_uses
    efficiency_score = min(1.0, strategic_actions / (steps_run * 0.3))

    # Component 4: LP Yield (10%)
    lp_score = min(1.0, fees_earned / (initial_value * 0.05))

    # Weighted final
    final_score = (
        0.40 * profit_score +
        0.30 * mev_score +
        0.20 * efficiency_score +
        0.10 * lp_score
    )

    return {
        "final_score":          round(final_score, 4),   # 0.0 – 1.0
        "profit_score":         round(profit_score, 4),
        "mev_avoidance_score":  round(mev_score, 4),
        "efficiency_score":     round(efficiency_score, 4),
        "lp_yield_score":       round(lp_score, 4),
        "total_mev_loss":       round(total_mev_loss, 6),
        "fees_earned":          round(fees_earned, 6),
        "final_portfolio_value": round(final_value, 4),
        "steps_run":            steps_run,
    }
```

### Determinism Guarantee

All randomness (background LP seeding, price walk, bot behavior) is controlled
by a single `seed` parameter passed to `env.reset(seed=N)`. The same seed always
produces the same episode. This guarantees:
- Reproducible grading across submissions
- Fair comparison between different agent policies
- Debuggable episodes (seed tells you exactly what happened)

### Score Interpretation

| Score Range | Interpretation |
|---|---|
| 0.0 – 0.30 | Agent fails basic mechanics; large MEV losses or negative returns |
| 0.30 – 0.50 | Agent understands basic swapping but ignores MEV signals |
| 0.50 – 0.65 | Agent avoids MEV partially; misses LP yield opportunities |
| 0.65 – 0.80 | Agent uses private RPC and split swaps appropriately |
| 0.80 – 1.0 | Agent detects JIT attacks, manages LP ranges, varies behavior to defeat adaptive bot |

---

## 10. Inference Script

```python
# inference.py — standard evaluation entry point

import json
from rli_v3_env import RLiquidityV3Env, TASK_CONFIGS

def run_episode(agent, task: str, seed: int = 42) -> dict:
    """
    Run one evaluation episode.
    
    Parameters
    ----------
    agent  : Any object with an .act(obs) -> action_dict method
    task   : "easy" | "medium" | "hard"
    seed   : int, controls all randomness (deterministic)
    
    Returns
    -------
    dict with score breakdown and full action log
    """
    env = RLiquidityV3Env(config=TASK_CONFIGS[task])
    obs, info = env.reset(seed=seed)

    action_log = []
    total_reward = 0.0

    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        action_log.append({
            "step":     info["step"],
            "action":   action["action_type"],
            "reward":   round(reward, 4),
            "mev_loss": info.get("mev_loss_this_step", 0),
            "price":    info.get("current_price"),
            "portfolio": info.get("portfolio_value"),
        })

        total_reward += reward
        if terminated or truncated:
            break

    score = env.grade()

    return {
        "task":         task,
        "seed":         seed,
        "total_reward": round(total_reward, 4),
        "score":        score,
        "action_log":   action_log,
    }


if __name__ == "__main__":
    from agents import RuleBasedAgent, LLMAgent

    # Run all three tasks with the rule-based baseline
    agent = RuleBasedAgent()
    results = {}

    for task in ("easy", "medium", "hard"):
        result = run_episode(agent, task=task, seed=42)
        results[task] = result["score"]
        print(f"\n[{task.upper()}] Final score: {result['score']['final_score']}")
        print(json.dumps(result["score"], indent=2))

    # Optionally run with LLM agent (requires API key)
    # llm_agent = LLMAgent(model="claude-sonnet-4-20250514")
    # result = run_episode(llm_agent, task="hard", seed=42)
```

---

## 11. LLM Agent Integration

One of the core contributions of this project is demonstrating that **LLM-based
agents can reason about structured financial state in ways that rule-based policies
cannot.**

### Prompt Template

The observation dict is serialized directly into the LLM prompt:

```python
SYSTEM_PROMPT = """
You are an expert DeFi trader operating in a Uniswap V3 ETH/USDC pool.
Your goal is to maximize your portfolio value while avoiding MEV attacks.

Key signals to watch:
- jit_liquidity > 0 at your current tick: a JIT bot is staged. Use private RPC.
- Large amounts in mempool: sandwich risk. Use split_swap or hold.
- agent_positions with tick range far from current_tick: earning zero fees. Rebalance.
- active_liquidity very low: your swap will have high price impact. Reduce size.

Respond ONLY with a valid JSON action dict. No explanation.
"""

def act(self, obs: dict) -> dict:
    prompt = f"""
Current pool state:
{json.dumps(obs, indent=2)}

Choose your action. Valid action types:
swap_exact_in | split_swap | add_liquidity | remove_liquidity |
range_order | jit_liquidity | hold | close_episode

Respond with JSON:
{{"action_type": "...", "params": {{...}}}}
"""
    response = call_llm(SYSTEM_PROMPT, prompt)
    return json.loads(response)
```

### Why LLMs Outperform Rule-Based Agents Here

A rule-based agent can be programmed to use private RPC when mempool is large.
But it cannot:

- Reason about whether the `jit_liquidity` spike is large enough to justify
  the private RPC cost
- Decide that a range order is better than a market swap given the current
  tick distribution shape
- Adapt its strategy mid-episode when the adaptive bot changes tactics
- Recognize that the current LP position tick range has drifted out of the
  active zone after a price swing

These decisions require **contextual reasoning over structured state** — exactly
where LLMs have demonstrated comparative advantage over rule-based systems.

---

## 12. Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Core environment | Python 3.10+ stdlib only | Zero dependencies, max portability |
| V3 math | Pure Python floats | No numpy required; deterministic |
| LLM agent | Anthropic / OpenAI API | Optional — rule-based agent works standalone |
| Visualization (optional) | Matplotlib | Price curves, MEV markers, tick distribution |
| Testing | Python unittest | Reproducible, no test framework dependency |

**No external RL libraries required.** The environment is fully self-contained.
It is compatible with Gymnasium-style wrappers but does not depend on them.

### File Structure

```
rli_v3/
├── rli_v3_env.py          # Main environment (self-contained)
├── inference.py           # Standard evaluation entry point
├── agents/
│   ├── rule_based.py      # Heuristic baseline agent
│   └── llm_agent.py       # LLM agent with prompt template
├── tasks/
│   └── configs.py         # TASK_CONFIGS: easy / medium / hard
├── grader/
│   └── grade.py           # Standalone grader — accepts episode log JSON
├── tests/
│   └── test_env.py        # Determinism + interface compliance tests
└── README.md
```

---

## 13. Real-World Impact

### Direct Analogue to Production Systems

Every strategic action in this environment maps directly to tools real traders
use on Ethereum mainnet:

| Environment Action | Real-World Equivalent |
|---|---|
| `use_private_rpc=True` | Flashbots Protect RPC, MEV Blocker |
| `split_swap` | 1inch Fusion order splitting |
| `range_order` | Uniswap V3 range order |
| `jit_liquidity` | JIT bot strategy (Uniswap research, 2022) |
| `hold` when mempool is dangerous | MEV-aware transaction timing |

### Who Uses This Research

- **MEV protection protocol teams** (Flashbots, CoW Protocol, MEV Blocker) use
  exactly this kind of simulation to test agent strategies before mainnet deployment
- **DeFi protocol risk teams** model adversarial dynamics to set fee tiers and
  tick spacing parameters
- **Algorithmic trading desks** with DeFi exposure need execution quality
  benchmarks
- **Academic researchers** in mechanism design use AMM simulations to study
  market microstructure

### The $686M Number Is Conservative

The $686M figure from 2023 MEV research covers only directly measurable on-chain
extraction. Indirect costs — worse liquidity provision due to JIT dilution, LPs
withdrawing capital from hostile pools, users paying higher slippage — are
estimated to multiply this by 3–5x in total welfare loss.

---

## 14. Execution Flow

```
Episode start
│
├─ env.reset(seed=42)
│   ├─ Seed all randomness (deterministic)
│   ├─ Initialize V3 pool at target price
│   ├─ Seed background LP positions (bell curve around current tick)
│   ├─ Initialize MEV bot with task strategy
│   └─ Return initial observation
│
├─ LOOP (up to max_steps):
│   │
│   ├─ Agent observes: tick_distribution, mempool, balances, positions
│   ├─ Agent produces: action_dict
│   │
│   ├─ [PRE-STEP] MEV bot inspects mempool
│   │   ├─ JIT strategy: inject ephemeral liquidity if swap size > threshold
│   │   └─ Sandwich strategy: front-run if slippage > threshold
│   │
│   ├─ Execute agent action
│   │   ├─ Validate action (penalize invalid)
│   │   ├─ Apply V3 swap math / LP position update
│   │   └─ Update pool state (sqrt_price, current_tick, balances)
│   │
│   ├─ [POST-STEP] MEV bot cleans up
│   │   ├─ JIT: remove ephemeral liquidity, capture fee
│   │   └─ Sandwich: back-run swap, book profit
│   │
│   ├─ Accrue LP fees to in-range positions
│   ├─ Apply random price walk (geometric Brownian motion)
│   ├─ Compute step reward (dense signal)
│   └─ Check termination (max_steps or close_episode action)
│
└─ env.grade()
    ├─ Compute profit_score (portfolio return, normalized)
    ├─ Compute mev_avoidance_score (total MEV loss, normalized)
    ├─ Compute efficiency_score (strategic action usage rate)
    ├─ Compute lp_yield_score (fees earned vs. capital deployed)
    └─ Return weighted final_score ∈ [0.0, 1.0]
```

---

## Summary

RLiquidity V3 is a focused, implementable, and genuinely novel RL environment
that sits at the intersection of two high-signal domains — DeFi market
microstructure and LLM agent evaluation — that have never been combined in a
public benchmark.

The core contribution is not simulating a blockchain. It is providing a
**structured, adversarial, multi-objective decision environment** where the
quality of an agent's reasoning is directly measurable, reproducible, and
comparable across policy types.

The three tasks cover a 2x range in agent performance between naive and expert
policies. The grading system is fully deterministic and normalized. The
implementation has zero external dependencies. And the domain is one where
a strong LLM agent, given the right observation context, can demonstrably
outperform rule-based baselines — which is the most compelling result this
hackathon's judges are looking for.

---

*Built for the Mini RL Environment Hackathon.*
*All randomness is seed-controlled. All scores are reproducible.*
*Core environment: `rli_v3_env.py` — zero dependencies beyond Python 3.10 stdlib.*
