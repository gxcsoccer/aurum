"""
计算当月策略信号并发布到 Supabase，供 Volta 读取执行。

用法：
  python publish_signal.py                    # 计算并发布
  python publish_signal.py --dry-run          # 只计算，不发布
  python publish_signal.py --month 2025-08    # 指定月份
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


def main():
    parser = argparse.ArgumentParser(description="Publish Aurum signal to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Only compute, don't publish")
    parser.add_argument("--month", type=str, default=None, help="Target month (YYYY-MM)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    month = args.month or datetime.now().strftime("%Y-%m")

    print(f"Aurum Signal Publisher - {month}")
    print()

    signal = get_current_signal(config)

    print(f"\n{'='*50}")
    print(f"  {month} Signal")
    print(f"{'='*50}")
    print(f"  Target: {signal['target_asset']}")
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
