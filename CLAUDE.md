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
- `loop.py` — main autoresearch-style evolution engine
- `validate.py` — holdout period validation
- `config.yaml` — system configuration
- `program.md` — agent work manual (iterate this to steer research)
- `infra/data.py` — yfinance data fetching + parquet caching
- `infra/backtest.py` — vectorized backtesting + walk-forward evaluation
- `infra/scorer.py` — scoring function (your investment philosophy)
- `infra/sandbox.py` — subprocess sandbox for safe strategy execution
- `infra/llm.py` — LLM client (Bailian/DashScope, OpenAI-compatible)
- `strategies/strategy.py` — current best strategy (LLM evolves this)

## Milestones
- M1: Minimum viable evolution loop (current)
- M2: Anti-overfitting (Deflated Sharpe Ratio, AST lookahead checker)
- M3: Diversity (Creative Director, multi-asset, multi-model competition)
- M4: Production (holdout validation, paper trading, monitoring)
