# MEVerse

MEVerse is an RL-style OpenEnv environment for evaluating whether an agent can trade and manage liquidity in a Uniswap V3-like market while avoiding MEV attacks.

## In One Minute

### What is RL?

Reinforcement Learning, or RL, is a setup where an agent:

- observes the current state of an environment
- chooses an action
- receives a reward or penalty
- improves by learning which actions lead to better long-term outcomes

In our project, the environment is a simulated DeFi market. The agent must decide when to trade, when to provide liquidity, when to wait, and how to avoid being exploited by adversarial bots.

### What is MEV?

MEV stands for `Maximal Extractable Value`. In simple terms, it is the profit that bots or validators can extract by seeing a transaction before it is finalized and then reordering, inserting, or copying transactions around it.

For a normal user this means:

- you submit a trade
- a bot sees it in the public mempool
- the bot acts before or around your trade
- you get a worse price, lose fees, or both

This is not a theoretical problem.

`MEV bots extracted $686M from Ethereum users in 2023.`

### What problem are we solving?

We are building an environment where an agent learns and is evaluated on a real task:

- executing swaps in a concentrated-liquidity market
- managing liquidity positions
- recognizing mempool danger signals
- adapting to adversarial bot behavior

Instead of scoring a model on a static question-answer benchmark, MEVerse tests whether the agent can make a sequence of decisions under changing market conditions.

## Why This Matters

Most trading benchmarks are too simple:

- `BUY / SELL / HOLD`
- no adversary
- no market microstructure
- no mempool visibility

MEVerse is built to be closer to the real operational problem in DeFi:

- liquidity is concentrated by price range, like Uniswap V3
- transactions can be observed and exploited before execution
- the same action may be safe in one state and costly in another
- the agent is rewarded for both profit and defensive behavior

## What We Built

MEVerse is an OpenEnv environment with three difficulty levels:

- `easy`: passive market, lower volatility, no aggressive adversary
- `medium`: JIT-liquidity behavior appears
- `hard`: adaptive adversary and higher volatility

The environment tracks:

- pool price and active liquidity
- local tick distribution around the current price
- agent balances and LP positions
- visible mempool transactions
- recent MEV loss
- episode progress and task type

The agent can:

- `swap_exact_in`
- `split_swap`
- `add_liquidity`
- `remove_liquidity`
- `range_order`
- `jit_liquidity`
- `hold`
- `close_episode`

## How The Environment Works

At each step:

1. the agent receives the current market and portfolio state
2. it chooses an action
3. the environment simulates trade execution, LP updates, and adversarial behavior
4. the agent receives a reward and the next observation

The reward is dense, not just pass/fail. It reflects:

- execution quality
- portfolio improvement
- LP fee capture
- MEV damage avoided or suffered
- end-of-episode performance

At the end of an episode, the environment also returns a normalized deterministic grade in `[0, 1]`.

## Current Base Implementation

The current implementation already includes the base project structure and working environment logic:

- typed OpenEnv models for action and observation
- a MEVerse environment server
- a Python client for interacting with the environment
- task switching across `easy`, `medium`, and `hard`
- local validation through `openenv validate`
- a baseline inference runner using the OpenAI client
- placeholder `.env` configuration for model access

The current base logic supports:

- swap execution
- liquidity add/remove flows
- range-style LP positioning
- JIT-liquidity simulation
- MEV-aware step rewards
- bounded invalid-action penalties with surfaced error metadata
- deterministic grading output

## Repository Structure

There is now one canonical README for the whole project at the repo root.

Main files and folders:

```text
.
├── README.md
├── .env
├── app.py
├── client.py
├── inference.py
└── meverse/
    ├── __init__.py
    ├── client.py
    ├── models.py
    ├── openenv.yaml
    ├── pyproject.toml
    ├── Dockerfile
    └── server/
        ├── app.py
        └── meverse_environment.py
```

## How To Run The Current Base

Validate the environment:

```bash
cd meverse
openenv validate
```

Run the baseline inference from the repo root:

```bash
python inference.py
```

Current inference configuration is read from:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME` or `MEVERSE_BASE_URL`
- `MEVERSE_TASK`

## What This Project Is Evaluating

This project is best understood as an RL-style evaluation environment for decision-making under adversarial market conditions.

The model is not being judged on memorizing DeFi trivia. It is being judged on whether it can:

- interpret structured market state
- choose sensible multi-step actions
- avoid predictable exploitation
- perform better as task difficulty increases

## Current Scope

This README intentionally focuses on the base implementation now in the repo. It explains the problem, the environment, and the current working structure without overloading the reader with deeper protocol math.

More detailed technical documentation can be added once the next implementation layer is complete.
