"""
计算当月策略信号并发布到 Supabase，供 Volta 读取执行。

执行层现金替代：策略信号中的 SHY 会被自动映射为 SGOV（见 config.yaml 中的
execution_cash_substitute 配置）。原因是 SGOV（0-3月国债）比 SHY（1-3年国债）
收益更高、波动更低，是更优的现金停泊工具。策略计算层仍使用 SHY 价格数据，
因为 SGOV 上市于 2020-06，缺乏 2008+ 历史数据用于回测。

用法：
  python publish_signal.py                    # 计算并发布（SHY→SGOV）
  python publish_signal.py --dry-run          # 只计算，不发布
  python publish_signal.py --month 2025-08    # 指定月份
  python publish_signal.py --no-substitute    # 不做 SHY→SGOV 替换
"""
import argparse
import os
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv

from infra.data import get_multi_prices, save_multi_prices
from infra.sandbox import run_strategy

load_dotenv()


def get_current_signal(config: dict) -> dict:
    """运行策略，获取最新信号。"""
    universe = config["universe"]

    # 获取最近 18 个月数据（策略需要 12 个月回望）
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=548)).strftime("%Y-%m-%d")

    print(f"  获取数据 ({start_date} ~ {end_date})...")
    prices = get_multi_prices(universe, start_date, end_date)
    data_path = "data_cache/multi_latest.pkl"
    save_multi_prices(prices, data_path)

    print("  运行策略...")
    strategy_code = open("strategies/strategy.py").read()
    signals = run_strategy(strategy_code, data_path)

    # 取最后一个有效信号
    latest_signal = signals.iloc[-1]
    latest_date = signals.index[-1]

    # 简单 12 个月动量（仅供参考，实际策略用多期加权）
    momentum_scores = {}
    for name, df in prices.items():
        close = df["close"]
        if len(close) > 252:
            mom = close.iloc[-1] / close.iloc[-253] - 1
            momentum_scores[name] = round(float(mom), 4)

    return {
        "target_asset": str(latest_signal),
        "as_of_date": latest_date.strftime("%Y-%m-%d"),
        "momentum_scores": momentum_scores,
    }


def publish_to_supabase(signal: dict, month: str) -> None:
    """将信号写入 Supabase（upsert，同月重复运行会更新而非报错）。"""
    from supabase import create_client

    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise ValueError(
            "请设置环境变量：\n"
            "  NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co\n"
            "  SUPABASE_SERVICE_ROLE_KEY=eyJxxx"
        )

    client = create_client(url, key)

    valid_from = f"{month}-01"
    year, mon = int(month[:4]), int(month[5:7])
    if mon == 12:
        valid_until = f"{year+1}-01-01"
    else:
        valid_until = f"{year}-{mon+1:02d}-01"

    record = {
        "strategy_name": "rotation_v1",
        "target_asset": signal["target_asset"],
        "target_weight": 1.0,
        "metadata": {
            "momentum_scores": signal["momentum_scores"],
            "as_of_date": signal["as_of_date"],
            "published_at": datetime.now().isoformat(),
        },
        "valid_from": valid_from,
        "valid_until": valid_until,
    }

    client.table("aurum_signals").upsert(
        record, on_conflict="strategy_name,valid_from"
    ).execute()
    print(f"  已发布到 Supabase: {month} -> {signal['target_asset']}")


def apply_execution_substitute(signal: dict, config: dict) -> tuple[dict, str | None]:
    """
    执行层现金替代：将策略信号中的 cash_asset 替换为 execution_cash_substitute。

    策略计算使用 SHY 的历史价格数据（2008+ 回测需要），但实际交易时
    SGOV 是更优的现金停泊工具（收益更高、波动更低、回撤更小）。

    Returns:
        (modified_signal, original_asset) - original_asset 为 None 表示未替换
    """
    substitute = config.get("execution_cash_substitute")
    cash_asset = config.get("cash_asset", "SHY")

    if not substitute or signal["target_asset"] != cash_asset:
        return signal, None

    original = signal["target_asset"]
    signal = dict(signal)
    signal["target_asset"] = substitute
    return signal, original


def main():
    parser = argparse.ArgumentParser(description="Publish Aurum signal to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Only compute, don't publish")
    parser.add_argument("--month", type=str, default=None, help="Target month (YYYY-MM)")
    parser.add_argument("--no-substitute", action="store_true",
                        help="Disable SHY->SGOV execution substitution")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    month = args.month or datetime.now().strftime("%Y-%m")

    print(f"Aurum Signal Publisher - {month}")
    print()

    signal = get_current_signal(config)

    # 执行层现金替代（SHY → SGOV）
    original_asset = None
    if not args.no_substitute:
        signal, original_asset = apply_execution_substitute(signal, config)

    print(f"\n{'='*50}")
    print(f"  {month} Signal")
    print(f"{'='*50}")
    print(f"  Target: {signal['target_asset']}")
    if original_asset:
        print(f"  (策略信号: {original_asset} → 执行替代: {signal['target_asset']})")
    print(f"  As of:  {signal['as_of_date']}")
    print(f"  Momentum (12M reference):")
    for name, score in sorted(signal["momentum_scores"].items(), key=lambda x: -x[1]):
        bar = "#" * int(max(0, score) * 30)
        print(f"    {name:5s} {score:+.2%} {bar}")
    print(f"{'='*50}")

    if args.dry_run:
        print("\n  Dry run - not published")
    else:
        publish_to_supabase(signal, month)
        print("\n  Done!")


if __name__ == "__main__":
    main()
