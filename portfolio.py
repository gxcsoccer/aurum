"""
策略组合器 — 将多个独立策略的信号合并为投资组合。

组合逻辑：
  - 等权混合：每个策略权重相等
  - 多数投票：多数策略看多才看多
  - Sharpe加权：按历史 Sharpe 比率分配权重

组合的核心价值：降低相关性 → 降低波动 → 提高 Sharpe。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

from infra.backtest import backtest, walk_forward_evaluate
from infra.data import get_prices
from infra.sandbox import run_strategy


def load_strategies() -> dict[str, str]:
    """加载所有已进化的策略文件。"""
    strategies = {}
    strategy_files = {
        "momentum": "strategies/strategy_momentum.py",
        "meanrev": "strategies/strategy_meanrev.py",
    }
    for name, path in strategy_files.items():
        if Path(path).exists():
            strategies[name] = open(path).read()
    return strategies


def get_strategy_signals(strategies: dict[str, str],
                         data_path: str) -> dict[str, pd.Series]:
    """获取每个策略的信号。"""
    signals = {}
    for name, code in strategies.items():
        try:
            sig = run_strategy(code, data_path)
            signals[name] = sig
            print(f"  ✅ {name}: {(sig == 1).sum()} 天做多 / {len(sig)} 天")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    return signals


def combine_equal_weight(signals: dict[str, pd.Series]) -> pd.Series:
    """等权混合：任一策略看多则看多（OR逻辑）。"""
    df = pd.DataFrame(signals)
    # 任一策略发出做多信号即做多
    combined = (df.max(axis=1)).astype(int)
    return combined


def combine_majority_vote(signals: dict[str, pd.Series]) -> pd.Series:
    """多数投票：过半策略看多才看多。"""
    df = pd.DataFrame(signals)
    n = len(signals)
    threshold = n / 2
    combined = (df.sum(axis=1) > threshold).astype(int)
    return combined


def combine_any(signals: dict[str, pd.Series]) -> pd.Series:
    """联合策略：任一策略看多则看多（最激进）。"""
    df = pd.DataFrame(signals)
    combined = (df.sum(axis=1) > 0).astype(int)
    return combined


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    symbol = config["symbol"]
    cost = config.get("cost_per_trade", 0.001)

    strategies = load_strategies()
    if len(strategies) < 2:
        print("❌ 需要至少 2 个策略才能组合。")
        print(f"   当前有: {list(strategies.keys())}")
        sys.exit(1)

    print(f"📊 策略组合分析 — {symbol}")
    print(f"   策略: {', '.join(strategies.keys())}")

    # ── In-Sample ──
    print(f"\n{'='*60}")
    print(f"  IN-SAMPLE (2015-2024)")
    print(f"{'='*60}")

    prices = get_prices(symbol, config["eval_start"], config["eval_end"])
    data_path = f"data_cache/{symbol}_eval.parquet"
    prices.to_parquet(data_path)

    print("\n📡 获取各策略信号:")
    all_signals = get_strategy_signals(strategies, data_path)

    # 计算相关性
    sig_df = pd.DataFrame(all_signals)
    corr = sig_df.corr()
    print(f"\n📐 策略信号相关性:")
    for i, name_i in enumerate(all_signals):
        for j, name_j in enumerate(all_signals):
            if i < j:
                print(f"   {name_i} ↔ {name_j}: {corr.loc[name_i, name_j]:.3f}")

    # 各种组合方式
    combiners = {
        "any (任一看多)": combine_any,
        "majority (多数投票)": combine_majority_vote,
    }

    # 加上单策略基准
    print(f"\n{'─'*60}")
    print(f"  单策略表现")
    print(f"{'─'*60}")
    for name, sig in all_signals.items():
        result = backtest(sig, prices, cost)
        print(f"  {name:20s}  Sharpe={result['sharpe']:+.3f}  "
              f"Return={result['total_return']:+.2%}  "
              f"MaxDD={result['max_drawdown']:.2%}  "
              f"Participation={result['participation']:.2%}")

    print(f"\n{'─'*60}")
    print(f"  组合策略表现")
    print(f"{'─'*60}")
    best_method = None
    best_sharpe = -999

    for method_name, combiner in combiners.items():
        combined = combiner(all_signals)
        result = backtest(combined, prices, cost)
        print(f"  {method_name:20s}  Sharpe={result['sharpe']:+.3f}  "
              f"Return={result['total_return']:+.2%}  "
              f"MaxDD={result['max_drawdown']:.2%}  "
              f"Participation={result['participation']:.2%}")
        if result["sharpe"] > best_sharpe:
            best_sharpe = result["sharpe"]
            best_method = method_name
            best_combiner = combiner

    # ── Holdout ──
    print(f"\n{'='*60}")
    print(f"  HOLDOUT (2025)")
    print(f"{'='*60}")

    prices_h = get_prices(symbol, config["holdout_start"], config["holdout_end"])
    data_path_h = f"data_cache/{symbol}_holdout.parquet"
    prices_h.to_parquet(data_path_h)

    print("\n📡 获取各策略信号:")
    all_signals_h = get_strategy_signals(strategies, data_path_h)

    print(f"\n{'─'*60}")
    print(f"  单策略 Holdout")
    print(f"{'─'*60}")
    for name, sig in all_signals_h.items():
        result = backtest(sig, prices_h, cost)
        print(f"  {name:20s}  Sharpe={result['sharpe']:+.3f}  "
              f"Return={result['total_return']:+.2%}  "
              f"MaxDD={result['max_drawdown']:.2%}")

    print(f"\n{'─'*60}")
    print(f"  组合策略 Holdout")
    print(f"{'─'*60}")
    for method_name, combiner in combiners.items():
        combined = combiner(all_signals_h)
        result = backtest(combined, prices_h, cost)
        marker = " ← best" if method_name == best_method else ""
        print(f"  {method_name:20s}  Sharpe={result['sharpe']:+.3f}  "
              f"Return={result['total_return']:+.2%}  "
              f"MaxDD={result['max_drawdown']:.2%}{marker}")

    # Buy-and-hold 基准
    bh_return = prices_h["close"].iloc[-1] / prices_h["close"].iloc[0] - 1
    print(f"\n  {'Buy & Hold':20s}  Return={bh_return:+.2%}")

    print(f"\n{'='*60}")
    print(f"  最佳组合方式: {best_method}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
