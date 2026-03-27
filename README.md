# Aurum

[中文版](README.zh-CN.md)

Self-evolving quantitative investment system for individual investors.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — LLM agents autonomously iterate strategy code through a greedy hill-climbing loop, while humans steer direction via scoring functions and research prompts.

## How It Works

```
Human Layer          →  scorer.py / program.md / config.yaml
Agent Layer (LLM)    →  strategies/strategy.py (the only mutable file)
Infra Layer (locked)  →  data.py / backtest.py / sandbox.py / llm.py
```

**Core loop:** LLM proposes mutation → sandbox execution → walk-forward backtest → score → keep or discard. Fully autonomous, no human intervention needed.

## Results

**Current strategy: Multi-period momentum rotation across 7 assets**

| Asset Pool | Role |
|---|---|
| SPY, QQQ, EFA, EEM | Offensive (equities) |
| TLT, GLD, SHY | Defensive (bonds, gold, cash) |

### In-Sample (2008–2024, 17 years)

| Metric | Value |
|---|---|
| Sharpe | 1.22 |
| Annual Return | 14.3% |
| Excess vs SPY B&H | +5.26%/yr |
| Max Drawdown | 28.6% |

### Holdout (2025, unseen data)

| Metric | Strategy | SPY B&H |
|---|---|---|
| Total Return | **+20.22%** | +18.60% |
| Max Drawdown | **18.25%** | 18.76% |
| Sharpe | +1.13 | — |

## Key Innovations (discovered by LLM agent)

1. **Multi-period momentum weighting** — 1/3/6/12 month returns weighted 1/2/4/6
2. **Volatility-adjusted ranking** — momentum / volatility for offensive assets
3. **Relative SPY threshold** — only hold offensive if momentum >= SPY momentum
4. **Pure momentum for defensive** — prevents SHY score inflation from ultra-low volatility
5. **Market volatility filter** — tighter threshold in high-vol regimes

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Set API key (Bailian/DashScope)
cp .env.example .env
# Edit .env with your DASHSCOPE_API_KEY

# Run evolution (50 iterations, ~30 min)
python loop.py -n 50

# Validate on holdout data
python validate.py

# Publish monthly signal (requires Supabase credentials)
python publish_signal.py
python publish_signal.py --dry-run    # preview only
```

## Architecture

```
aurum/
├── config.yaml              # Asset universe, time windows, LLM settings
├── program.md               # Agent work manual (steer research direction)
├── loop.py                  # Autoresearch-style evolution engine
├── validate.py              # Holdout period validation
├── publish_signal.py        # Monthly signal publisher to Supabase
├── portfolio.py             # Multi-strategy portfolio combiner
│
├── infra/                   # Immutable — never modify during evolution
│   ├── data.py              # yfinance multi-asset data + parquet/pickle cache
│   ├── backtest.py          # Multi-asset rotation backtest + walk-forward
│   ├── scorer.py            # Scoring function with Deflated Sharpe Ratio
│   ├── sandbox.py           # Subprocess sandbox for safe strategy execution
│   └── llm.py               # LLM client (Bailian/DashScope, OpenAI-compatible)
│
├── strategies/
│   └── strategy.py          # Current best strategy (LLM evolves this)
│
├── experiments/
│   └── results.tsv          # Full experiment log (including failures)
│
└── .github/workflows/
    ├── publish-signal.yml   # Monthly: auto-publish rotation signal
    └── evolve.yml           # Quarterly: auto-evolve strategy (creates PR)
```

## Scoring Philosophy

The scoring function (`infra/scorer.py`) **is** your investment philosophy in code. Change its weights and the agent evolves in a completely different direction.

Current design (v5):
- **Excess return vs SPY B&H** — the primary metric (weight 3.0)
- **Absolute return** — raw performance (weight 1.0)
- **Deflated Sharpe Ratio** — anti-overfitting: adjusts confidence based on total experiments run
- **Cash penalty** — >30% cash starts deducting, >70% hard rejection
- **Consistency** — low variance of excess returns across years

Every version of the scorer was eventually "exploited" by the agent:

| Exploit | Agent Behavior | Fix |
|---|---|---|
| High drawdown penalty | Learned to hold cash (0 trades = 0 drawdown) | Added participation rate |
| Drawdown advantage bonus | 100% cash = max bonus | Removed the bonus entirely |
| Cash penalty > 60% | Kept exactly 59% cash | Progressive penalty from 30% |

## Volta Integration

Aurum integrates with [Volta](https://github.com/gxcsoccer/volta) (AI trading arena) for paper trading:

```
Aurum (offline)              Volta (online)
  │                            │
  │  publish_signal.py         │  Every 15 min
  │         │                  │       │
  ▼         ▼                  ▼       ▼
  Strategy → Supabase ──────→ Aurum Rotator agent
  evolution   signal table     reads signal, executes trades
```

- **Strategic layer (monthly):** Aurum computes which asset to hold
- **Tactical layer (15 min):** Volta checks circuit breaker (stop_loss=8%)
- **Feedback layer (monthly):** Real trade data calibrates backtest assumptions

## Anti-Overfitting

- **Walk-forward validation** — yearly sub-period evaluation (not single train/test split)
- **Deflated Sharpe Ratio** (Lopez de Prado) — penalizes strategies found after many trials
- **Holdout period** — 2025 data never touched during evolution
- **Sandbox execution** — strategies run in subprocess, can't hack the scorer

## Configuration

Key settings in `config.yaml`:

```yaml
universe: [SPY, QQQ, EFA, EEM, TLT, GLD, SHY]
eval_start: "2008-01-01"
eval_end: "2024-12-31"
holdout_start: "2025-01-01"
max_drawdown_limit: 0.30
cost_per_trade: 0.001

llm:
  base_url: "https://coding.dashscope.aliyuncs.com/v1"
  models: ["qwen3.5-plus"]
```

## License

MIT
