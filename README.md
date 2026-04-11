# Aurum

[中文版](README.zh-CN.md)

Self-evolving quantitative investment system for individual investors.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent) — LLM agents autonomously discover and accumulate alpha factors, while humans steer direction via scoring functions and research prompts.

## How It Works

Two evolution modes:

```
Mode 1: Monolithic (loop.py)          Mode 2: Factor Accumulation (factor_loop.py) ← recommended
  LLM rewrites entire strategy          LLM proposes one small factor
  → sandbox → score                     → quality check → add to library → reassemble
  → keep or discard whole file           → keep factor or discard (others unaffected)
```

```
Human Layer          →  scorer.py / program.md / config.yaml
Agent Layer (LLM)    →  factors/ (factor library, LLM's canvas)
Assembly Layer       →  factor_loop.py assembles factors into strategies/strategy.py
Infra Layer (locked)  →  data.py / backtest.py / sandbox.py / llm.py
```

**Factor accumulation** outperforms monolithic mutation: 50 rounds yielded 4 useful factors vs 200 rounds with 0 improvements in monolithic mode.

## Results

**Current strategy: Multi-period momentum rotation across 7 assets**

| Asset Pool | Role |
|---|---|
| SPY, QQQ, EFA, EEM | Offensive (equities) |
| TLT, GLD, SHY | Defensive (bonds, gold, cash) |

### Cumulative Growth of $10,000 (2008–2024)

| | Aurum | SPY B&H |
|---|---|---|
| **Final Value** | **$160,084** | $56,023 |
| Total Return | +1500% | +460% |
| Annual Return | +18.0% | +10.6% |
| Sharpe | 0.99 | ~0.55 |
| Max Drawdown | 28.6% | ~51% |

### Year-by-Year Performance

| Year | Aurum | SPY | Excess | Winner |
|---|---|---|---|---|
| 2008 | +6.43% | -36.24% | **+42.67%** | Aurum |
| 2009 | +44.67% | +22.65% | +22.01% | Aurum |
| 2010 | +17.84% | +13.14% | +4.70% | Aurum |
| 2011 | -1.64% | +0.85% | -2.49% | SPY |
| 2012 | +11.95% | +14.17% | -2.22% | SPY |
| 2013 | +25.50% | +29.00% | -3.50% | SPY |
| 2014 | +17.87% | +14.56% | +3.31% | Aurum |
| 2015 | +9.65% | +1.29% | +8.37% | Aurum |
| 2016 | -3.01% | +13.59% | -16.60% | SPY |
| 2017 | +22.98% | +20.78% | +2.20% | Aurum |
| 2018 | -0.94% | -5.25% | +4.31% | Aurum |
| 2019 | +62.04% | +31.09% | +30.96% | Aurum |
| 2020 | +41.28% | +17.24% | +24.04% | Aurum |
| 2021 | +25.57% | +30.51% | -4.94% | SPY |
| 2022 | -11.62% | -18.65% | +7.02% | Aurum |
| 2023 | +26.42% | +26.71% | -0.29% | SPY |
| 2024 | +22.29% | +26.05% | -3.76% | SPY |

**Win rate: 10/17 years (59%)** — beats SPY in down years, keeps up in bull years.

### Out-of-Sample (unseen data, never used during evolution)

| Period | Aurum | SPY | Excess |
|---|---|---|---|
| 2025 Full Year (8-factor) | **+29.08%** | +18.60% | **+12.15%** |
| 2025 Full Year (4-factor baseline) | +20.22% | +18.60% | +1.62% |

The 8-factor version (after factor evolution) significantly outperforms the 4-factor baseline on holdout data, confirming that the evolved factors generalize well.

## Key Innovations

**Base factors** (discovered by LLM monolithic evolution):
1. **Multi-period momentum weighting** — 1/3/6/12 month returns weighted 1/2/4/6
2. **Volatility-adjusted ranking** — momentum / volatility for offensive assets
3. **Relative SPY threshold** — only hold offensive if momentum >= SPY momentum
4. **Pure momentum for defensive** — prevents SHY score inflation from ultra-low volatility
5. **Market volatility filter** — tighter threshold in high-vol regimes

**Evolved factors** (discovered by LLM factor accumulation):
6. **Momentum exhaustion regime** — broad deceleration of offensive momentum precedes vol spikes
7. **Trend consistency** — prefer steady risers over volatile jumpers
8. **Correlation velocity regime** — rapid increase in cross-asset correlation signals crisis onset
9. **Defensive leadership persistence** — sustained defensive outperformance confirms risk-off regime

## Execution-Layer Cash Substitute

Strategy signals use SHY (1-3yr Treasury) for backtesting (requires 2008+ history).
At execution time, SHY is automatically mapped to **SGOV** (iShares 0-3 Month Treasury Bond ETF):

| | SGOV | SHY |
|---|---|---|
| Annual Return | **2.9%** | 1.5% |
| Volatility | **0.24%** | 1.83% |
| Max Drawdown | **-0.03%** | -5.71% |
| 2022 (rate shock) | **+1.59%** | -3.77% |

This mapping is applied in `publish_signal.py` and can be disabled with `--no-substitute`.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Set API key (Bailian/DashScope)
cp .env.example .env
# Edit .env with your DASHSCOPE_API_KEY

# Run factor evolution (recommended, 50 iterations, ~1 hour)
python factor_loop.py -n 50

# Or run monolithic evolution (legacy, 50 iterations, ~30 min)
python loop.py -n 50

# Validate on holdout data
python validate.py

# Publish monthly signal (requires Supabase credentials)
python publish_signal.py                # SHY auto-mapped to SGOV
python publish_signal.py --dry-run      # preview only
python publish_signal.py --no-substitute  # keep SHY as-is
```

## Architecture

```
aurum/
├── config.yaml              # Asset universe, time windows, LLM settings
├── factor_config.yaml       # Factor registry, weights, combiner parameters
├── program.md               # Agent work manual for monolithic evolution
├── program_factor.md        # Agent work manual for factor mining
├── loop.py                  # Monolithic evolution engine (legacy)
├── factor_loop.py           # Factor accumulation engine (recommended)
├── validate.py              # Holdout period validation
├── publish_signal.py        # Monthly signal publisher (SHY→SGOV mapping)
├── portfolio.py             # Multi-strategy portfolio combiner
│
├── factors/                 # Factor library (base + LLM-evolved)
│   ├── base_offensive_score.py       # Vol-adjusted multi-period momentum
│   ├── base_defensive_score.py       # Pure multi-period momentum
│   ├── base_market_regime.py         # Market volatility regime
│   ├── base_ma_filter.py             # 63-day MA trend filter
│   ├── evolved_015_*.py              # Momentum exhaustion regime
│   ├── evolved_020_*.py              # Trend consistency
│   ├── evolved_036_*.py              # Correlation velocity regime
│   └── evolved_053_*.py              # Defensive leadership persistence
│
├── infra/                   # Immutable — never modify during evolution
│   ├── data.py              # yfinance multi-asset data + parquet/pickle cache
│   ├── backtest.py          # Multi-asset rotation backtest + walk-forward
│   ├── scorer.py            # Scoring function with Deflated Sharpe Ratio
│   ├── sandbox.py           # Subprocess sandbox for safe strategy execution
│   └── llm.py               # LLM client (Bailian/DashScope, OpenAI-compatible)
│
├── strategies/
│   └── strategy.py          # Current best strategy (auto-assembled from factors)
│
├── experiments/
│   ├── results.tsv          # Monolithic evolution log
│   └── factor_results.tsv   # Factor evolution log
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
- **Factor count discipline** — empirically, 8 factors is the sweet spot; 12 factors caused holdout degradation from +12% to +7% excess (overfitting)

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
