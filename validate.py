"""
Holdout 验证 — 多资产轮动版。
在进化循环期间从未使用的保留期数据上验证最终策略。
"""
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from infra.backtest import backtest_rotation
from infra.data import get_multi_prices, save_multi_prices
from infra.sandbox import run_strategy


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    universe = config["universe"]
    cash_asset = config.get("cash_asset", "SHY")
    benchmark = config.get("benchmark", "SPY")
    holdout_start = config["holdout_start"]
    holdout_end = config["holdout_end"]

    print(f"📋 Holdout 验证 ({holdout_start} ~ {holdout_end})")
    print(f"   资产池: {', '.join(universe)}")
    print()

    # 获取数据（多拿 18 个月历史做动量指标预热）
    from datetime import datetime, timedelta
    warmup_start = (datetime.strptime(holdout_start, "%Y-%m-%d") - timedelta(days=548)).strftime("%Y-%m-%d")
    all_prices = get_multi_prices(universe, warmup_start, holdout_end)
    data_path = "data_cache/multi_holdout.pkl"
    save_multi_prices(all_prices, data_path)

    # 执行策略
    print("🔄 运行策略...")
    strategy_code = open("strategies/strategy.py").read()
    signals_full = run_strategy(strategy_code, data_path)
    # 只取 holdout 期间的信号用于评估
    signals = signals_full[signals_full.index >= holdout_start]

    result = backtest_rotation(
        signals, all_prices, cash_asset, benchmark,
        config.get("cost_per_trade", 0.001),
    )

    # 报告
    print()
    print("=" * 58)
    print("  HOLDOUT VALIDATION RESULTS (多资产轮动)")
    print("=" * 58)
    print(f"  Period:          {holdout_start} ~ {holdout_end}")
    print(f"  Trading Days:    {result['n_days']}")
    print(f"  ────────────────────────────────────────")
    print(f"  Sharpe:          {result['sharpe']:+.4f}")
    print(f"  Ann Return:      {result['ann_return']:+.2%}")
    print(f"  Total Return:    {result['total_return']:+.2%}")
    print(f"  Max Drawdown:    {result['max_drawdown']:.2%}")
    print(f"  ────────────────────────────────────────")
    print(f"  SPY B&H Return:  {result['bh_return']:+.2%}")
    print(f"  SPY B&H MaxDD:   {result['bh_max_dd']:.2%}")
    print(f"  **Excess Return: {result['excess_return']:+.2%}**")
    print(f"  ────────────────────────────────────────")
    print(f"  Cash %:          {result['cash_pct']:.2%}")
    print(f"  Switches:        {result['n_switches']}")
    print(f"  Win Rate:        {result['win_rate']:.2%}")

    # 资产配置
    alloc = result.get("allocation", {})
    if alloc:
        print(f"  ────────────────────────────────────────")
        print(f"  Asset Allocation:")
        for name, pct in sorted(alloc.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct * 30)
            print(f"    {name:5s} {pct:5.1%} {bar}")

    print("=" * 58)

    # 判定
    print()
    beat_bh = result["excess_return"] > 0
    low_dd = result["max_drawdown"] < result["bh_max_dd"]

    if beat_bh and low_dd:
        print("✅ 完美！跑赢 SPY 且回撤更低。可以考虑纸交易。")
    elif beat_bh:
        print("✅ 跑赢 SPY！但回撤比 B&H 高，可继续优化风控。")
    elif low_dd:
        print("⚠️  没跑赢 SPY，但回撤显著更低。风险调整后可能有价值。")
    else:
        print("❌ 未跑赢 SPY 且回撤不占优。策略需继续优化。")


if __name__ == "__main__":
    main()
