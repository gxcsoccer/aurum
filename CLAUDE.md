# Aurum — Self-Evolving Quantitative Investment System

## Architecture
Three-layer separation (inspired by Karpathy's autoresearch):
- **Immutable layer** (`infra/`): data pipeline, backtester, scorer, sandbox — NEVER modify
- **Mutable layer** (`strategies/`): strategy code — the LLM's canvas
- **Control layer** (`program.md`): agent instructions — human iterates this to steer direction

## Quick Start
```bash
pip install -e .
export DASHSCOPE_API_KEY=sk-xxxxxxxx
python loop.py          # run evolution loop
python validate.py      # holdout validation
```

## Key Rules
- Files in `infra/` are immutable — do not modify during evolution
- All strategy signals must use `.shift(1)` to prevent look-ahead bias
- `config.yaml` controls evaluation parameters and LLM settings
- `program.md` controls the agent's behavior and research direction
- `experiments/results.tsv` logs all experiments (including failures)
- Git history tracks only successful strategy mutations

## File Structure
- `loop.py` — main autoresearch-style evolution engine (monolithic strategy mutation)
- `factor_loop.py` — factor-based evolution engine (RD-Agent style, factor accumulation)
- `validate.py` — holdout period validation
- `config.yaml` — system configuration
- `factor_config.yaml` — factor registry, weights, and combiner parameters
- `program.md` — agent work manual for strategy evolution
- `program_factor.md` — agent work manual for factor mining
- `publish_signal.py` — compute and publish monthly signal to Supabase
- `infra/data.py` — yfinance data fetching + parquet caching
- `infra/backtest.py` — vectorized backtesting + walk-forward evaluation
- `infra/scorer.py` — scoring function (your investment philosophy)
- `infra/sandbox.py` — subprocess sandbox for safe strategy execution
- `infra/llm.py` — LLM client (Bailian/DashScope, OpenAI-compatible)
- `strategies/strategy.py` — current best strategy (auto-assembled from factor library)
- `factors/` — factor library (base + evolved factors)

## Evolution Modes
- **`loop.py`** (monolithic): LLM rewrites entire strategy.py each iteration. Greedy hill climbing.
  200 rounds yielded 0 improvements — strategy is at a local optimum.
- **`factor_loop.py`** (factor accumulation, recommended): LLM proposes one small factor per iteration.
  Good factors accumulate, bad factors are discarded. 50 rounds yielded 4 new factors.
  Warning: more than ~8 factors causes overfitting (12 factors degraded holdout from +12% to +7%).

## Execution-Layer Cash Substitute
Strategy signals use SHY (1-3yr Treasury) for backtesting (2008+ history required).
At execution time, SHY signals are mapped to SGOV (0-3mo Treasury) — see `config.yaml`.
Reason: SGOV has higher yield (2.9% vs 1.5%), 8x lower volatility, near-zero drawdown.
This mapping is handled by `publish_signal.py` and can be disabled with `--no-substitute`.

## Milestones
- M1: Minimum viable evolution loop ✅
- M2: Anti-overfitting (Deflated Sharpe Ratio, walk-forward validation) ✅
- M3: Factor-based evolution (RD-Agent style factor accumulation) ✅
- M4: Production (signal publishing to Supabase, SGOV cash substitute) ✅
- M5: Early stopping for factor evolution (prevent overfitting past optimal factor count)
