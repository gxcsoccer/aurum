"""
Aurum 主循环 — 多资产轮动版。
autoresearch 风格的策略进化引擎。
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from infra.backtest import backtest_rotation, walk_forward_rotation
from infra.data import get_multi_prices, save_multi_prices
from infra.llm import ask_mutation, create_client
from infra.sandbox import run_strategy
from infra.scorer import score

# ── 文件路径 ──
CONFIG_PATH = "config.yaml"
STRATEGY_PATH = "strategies/strategy.py"
PROGRAM_MD_PATH = "program.md"
RESULTS_FILE = "experiments/results.tsv"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def log_experiment(**kwargs) -> None:
    """追加实验记录到 TSV。"""
    headers = [
        "timestamp", "iteration", "model", "score", "sharpe",
        "max_dd", "ann_return", "excess_return", "cash_pct",
        "n_switches", "status", "hypothesis",
    ]
    if not os.path.exists(RESULTS_FILE):
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            f.write("\t".join(headers) + "\n")

    row = [
        kwargs.get("timestamp", datetime.now().isoformat()),
        str(kwargs.get("iteration", 0)),
        kwargs.get("model", ""),
        f"{kwargs.get('score', 0):.6f}",
        f"{kwargs.get('sharpe', 0):.4f}",
        f"{kwargs.get('max_dd', 0):.4f}",
        f"{kwargs.get('ann_return', 0):.4f}",
        f"{kwargs.get('excess_return', 0):.4f}",
        f"{kwargs.get('cash_pct', 0):.4f}",
        str(kwargs.get("n_switches", 0)),
        kwargs.get("status", ""),
        kwargs.get("hypothesis", "").replace("\t", " ").replace("\n", " ")[:120],
    ]
    with open(RESULTS_FILE, "a") as f:
        f.write("\t".join(row) + "\n")


def get_history(n: int = 30) -> str:
    """读取最近 N 条实验历史。"""
    if not os.path.exists(RESULTS_FILE):
        return "（暂无历史记录）"
    with open(RESULTS_FILE) as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return "（暂无历史记录）"

    recent = lines[-n:] if len(lines) > n + 1 else lines[1:]
    formatted = []
    for line in recent:
        parts = line.strip().split("\t")
        if len(parts) >= 12:
            status = parts[10]
            icon = "✅" if status == "KEPT" else "❌" if status == "DISCARDED" else "💥"
            formatted.append(
                f"{icon} score={parts[3]} excess={parts[7]} sharpe={parts[4]} | {parts[11]}"
            )
    return "\n".join(formatted) if formatted else "（暂无历史记录）"


def evaluate_strategy(strategy_code: str, data_path: str,
                      all_prices: dict, config: dict) -> tuple[float, dict]:
    """执行策略并计算 walk-forward 评分。"""
    signals = run_strategy(strategy_code, data_path)

    cash_asset = config.get("cash_asset", "SHY")
    benchmark = config.get("benchmark", "SPY")

    sub_results = walk_forward_rotation(
        signals, all_prices,
        cash_asset=cash_asset,
        benchmark=benchmark,
        cost=config.get("cost_per_trade", 0.001),
        sub_period=config.get("sub_period", "yearly"),
    )

    s = score(
        sub_results,
        max_dd_limit=config.get("max_drawdown_limit", 0.30),
        min_sharpe=config.get("min_sharpe", 0.0),
    )

    if sub_results:
        # 合并所有子期间的 allocation
        all_alloc = {}
        for r in sub_results:
            for k, v in r.get("allocation", {}).items():
                all_alloc[k] = all_alloc.get(k, 0) + v
        total = sum(all_alloc.values()) or 1
        all_alloc = {k: v / total for k, v in all_alloc.items()}

        agg = {
            "sharpe": round(np.mean([r["sharpe"] for r in sub_results]), 4),
            "max_drawdown": round(max(r["max_drawdown"] for r in sub_results), 4),
            "ann_return": round(np.mean([r["ann_return"] for r in sub_results]), 4),
            "excess_return": round(np.mean([r["excess_return"] for r in sub_results]), 4),
            "cash_pct": round(np.mean([r["cash_pct"] for r in sub_results]), 4),
            "n_switches": round(np.mean([r["n_switches"] for r in sub_results]), 1),
            "win_rate": round(np.mean([r["win_rate"] for r in sub_results]), 4),
            "allocation": all_alloc,
            "n_periods": len(sub_results),
        }
    else:
        agg = {
            "sharpe": 0, "max_drawdown": 0, "ann_return": 0,
            "excess_return": 0, "cash_pct": 1, "n_switches": 0,
            "win_rate": 0, "allocation": {}, "n_periods": 0,
        }

    return s, agg


def git_commit(message: str) -> None:
    subprocess.run(["git", "add", STRATEGY_PATH], capture_output=True)
    subprocess.run(["git", "commit", "-m", message], capture_output=True)


def main() -> None:
    config = load_config()
    universe = config["universe"]

    # ── 数据准备 ──
    print(f"🏗️  Aurum 多资产轮动 — 加载 {len(universe)} 个资产...")
    all_prices = get_multi_prices(universe, config["eval_start"], config["eval_end"])

    data_path = "data_cache/multi_eval.pkl"
    save_multi_prices(all_prices, data_path)

    # ── 加载资源 ──
    program_md = open(PROGRAM_MD_PATH).read()
    current_code = open(STRATEGY_PATH).read()

    # ── 基线评估 ──
    print("\n📊 评估初始策略...")
    current_score, current_agg = evaluate_strategy(
        current_code, data_path, all_prices, config,
    )
    alloc_str = ", ".join(f"{k}:{v:.0%}" for k, v in
                          sorted(current_agg["allocation"].items(), key=lambda x: -x[1])[:5])
    print(f"   初始评分: {current_score:.4f}")
    print(f"   Sharpe: {current_agg['sharpe']:.4f} | "
          f"年化: {current_agg['ann_return']:.2%} | "
          f"超额: {current_agg['excess_return']:.2%} | "
          f"MaxDD: {current_agg['max_drawdown']:.2%}")
    print(f"   配置: {alloc_str}")

    # ── LLM ──
    llm_client = create_client(config)
    models = config["llm"]["models"]
    iterations = config["loop"]["iterations"]
    history_window = config["loop"]["history_window"]

    kept_count = 0
    crash_count = 0

    # ── 主循环 ──
    print(f"\n🔄 开始进化 ({iterations} 次迭代)...\n")

    for i in range(1, iterations + 1):
        model = models[(i - 1) % len(models)]

        try:
            history = get_history(history_window)

            new_code, hypothesis = ask_mutation(
                llm_client, model,
                current_code, current_score, current_agg,
                history, program_md,
            )

            new_score, new_agg = evaluate_strategy(
                new_code, data_path, all_prices, config,
            )

            kept = new_score > current_score
            status = "KEPT" if kept else "DISCARDED"

            log_experiment(
                iteration=i, model=model, score=new_score,
                sharpe=new_agg["sharpe"], max_dd=new_agg["max_drawdown"],
                ann_return=new_agg["ann_return"],
                excess_return=new_agg["excess_return"],
                cash_pct=new_agg["cash_pct"],
                n_switches=new_agg["n_switches"],
                status=status, hypothesis=hypothesis,
            )

            icon = "✅" if kept else "❌"
            print(
                f"{icon} #{i:3d} [{model:15s}] "
                f"score={new_score:+.4f} (was {current_score:+.4f}) "
                f"excess={new_agg['excess_return']:+.2%} "
                f"| {hypothesis[:50]}"
            )

            if kept:
                kept_count += 1
                current_code = new_code
                current_score = new_score
                current_agg = new_agg
                with open(STRATEGY_PATH, "w") as f:
                    f.write(current_code)
                git_commit(f"improve #{i}: score={new_score:.4f} | {hypothesis[:60]}")

        except KeyboardInterrupt:
            print("\n⏹️  用户中断")
            break
        except Exception as e:
            crash_count += 1
            error_msg = str(e)[:100]
            print(f"💥 #{i:3d} [{model:15s}] {error_msg}")
            log_experiment(
                iteration=i, model=model, score=0, sharpe=0, max_dd=0,
                ann_return=0, excess_return=0, cash_pct=0, n_switches=0,
                status="CRASH", hypothesis=f"ERROR: {error_msg}",
            )
            continue

    # ── 总结 ──
    total = i if "i" in dir() else 0
    print(f"\n{'='*60}")
    print(f"🏁 进化完成!")
    print(f"   迭代: {total} | 改进: {kept_count} ({kept_count/max(total,1):.0%}) | 崩溃: {crash_count}")
    print(f"   最终评分: {current_score:.4f}")
    print(f"   Sharpe: {current_agg['sharpe']:.4f}")
    print(f"   年化: {current_agg['ann_return']:.2%}")
    print(f"   超额(vs SPY): {current_agg['excess_return']:.2%}")
    print(f"   最大回撤: {current_agg['max_drawdown']:.2%}")
    print(f"{'='*60}")
    print(f"\n下一步: python validate.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aurum evolution loop")
    parser.add_argument("-n", "--iterations", type=int, default=None)
    args = parser.parse_args()

    _cfg = load_config()
    if args.iterations is not None:
        _cfg["loop"]["iterations"] = args.iterations
    load_config = lambda: _cfg
    main()
