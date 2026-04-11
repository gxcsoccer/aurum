"""
Aurum 因子进化引擎 — RD-Agent 风格的因子挖掘 + 积累。

核心区别于 loop.py：
- loop.py: 每次变异整个策略，只保留最优（贪心爬山）
- factor_loop.py: 每次提出一个因子，好因子积累，坏因子丢弃

工作流：
1. 加载因子库，组装策略，评估基线
2. LLM 提出新因子
3. 沙盒测试新因子（是否能运行、输出格式正确）
4. 评估因子质量（IC、与现有因子相关性）
5. 临时加入因子库，重新组装策略，评估全局得分
6. 得分提升 → 保留因子，git commit
7. 得分下降 → 禁用因子
"""
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from infra.backtest import walk_forward_rotation
from infra.data import get_multi_prices, save_multi_prices
from infra.llm import create_client
from infra.sandbox import run_strategy
from infra.scorer import score

# ── 路径 ──
MAIN_CONFIG = "config.yaml"
FACTOR_CONFIG = "factor_config.yaml"
STRATEGY_PATH = "strategies/strategy.py"
PROGRAM_MD = "program_factor.md"
RESULTS_FILE = "experiments/factor_results.tsv"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# ════════════════════════════════════════════
#  因子管理
# ════════════════════════════════════════════

def load_factor_source(filepath):
    """读取因子文件源码。"""
    with open(filepath) as f:
        return f.read()


def get_enabled_factors(factor_cfg):
    """返回所有已启用的因子配置。"""
    return {
        name: cfg for name, cfg in factor_cfg["factors"].items()
        if cfg.get("enabled", True)
    }


def get_factor_summary(factor_cfg):
    """生成因子库摘要（供 LLM 参考）。"""
    lines = []
    for name, cfg in factor_cfg["factors"].items():
        status = "✅" if cfg.get("enabled", True) else "❌"
        cat = cfg.get("category", "?")
        # 读取因子文件的 docstring 作为描述
        desc = ""
        if os.path.exists(cfg["file"]):
            src = load_factor_source(cfg["file"])
            match = re.search(r'"""(.*?)"""', src, re.DOTALL)
            if match:
                desc = match.group(1).strip().split("\n")[-1].strip()
        lines.append(f"{status} [{cat:10s}] {name}: {desc[:60]}")
    return "\n".join(lines)


# ════════════════════════════════════════════
#  策略组装器
# ════════════════════════════════════════════

def assemble_strategy(factor_cfg):
    """
    将所有启用的因子 + 组合逻辑组装成完整的 strategy.py。

    生成一个自包含的 Python 文件，可被 infra/sandbox.py 执行。
    """
    enabled = get_enabled_factors(factor_cfg)

    offensive = factor_cfg["offensive"]
    defensive = factor_cfg["defensive"]
    cash = factor_cfg["cash"]
    combiner = factor_cfg.get("combiner", {})

    # ── 1. 内联因子函数 ──
    factor_functions = []
    factor_calls = []

    for name, cfg in enabled.items():
        src = load_factor_source(cfg["file"])
        # 重命名 compute → compute_{name}
        func_name = f"compute_{name}"
        src = src.replace("def compute(", f"def {func_name}(")
        factor_functions.append(f"# ── Factor: {name} ──\n{src}")

        scope = cfg.get("scope", "offensive")
        if scope == "offensive":
            assets_var = "OFFENSIVE"
        elif scope == "defensive":
            assets_var = "DEFENSIVE"
        else:
            assets_var = "list(prices.keys())"

        category = cfg.get("category", "offensive")
        weight = cfg.get("weight", 1.0)

        factor_calls.append({
            "name": name,
            "func_name": func_name,
            "category": category,
            "scope_var": assets_var,
            "weight": weight,
        })

    # ── 2. 生成策略代码 ──
    # 分类因子调用
    off_factors = [f for f in factor_calls if f["category"] == "offensive"]
    def_factors = [f for f in factor_calls if f["category"] == "defensive"]
    regime_factors = [f for f in factor_calls if f["category"] == "regime"]
    filter_factors = [f for f in factor_calls if f["category"] == "filter"]

    # 生成因子调用代码
    factor_compute_lines = []
    for f in factor_calls:
        if f["category"] == "regime":
            factor_compute_lines.append(
                f'    regime_{f["name"]} = {f["func_name"]}(prices, all_dates, {f["scope_var"]})'
            )
        elif f["category"] == "filter":
            factor_compute_lines.append(
                f'    filter_{f["name"]} = {f["func_name"]}(prices, all_dates, {f["scope_var"]})'
            )
        elif f["category"] == "offensive":
            factor_compute_lines.append(
                f'    off_{f["name"]} = {f["func_name"]}(prices, all_dates, {f["scope_var"]})'
            )
        elif f["category"] == "defensive":
            factor_compute_lines.append(
                f'    def_{f["name"]} = {f["func_name"]}(prices, all_dates, {f["scope_var"]})'
            )

    # 生成进攻型组合评分
    if off_factors:
        total_w = sum(f["weight"] for f in off_factors)
        off_combine_parts = []
        for f in off_factors:
            w = f["weight"] / total_w
            off_combine_parts.append(f'off_{f["name"]} * {w:.4f}')
        off_combine_expr = " + ".join(off_combine_parts)
    else:
        off_combine_expr = "pd.DataFrame()"

    # 生成防御型组合评分
    if def_factors:
        total_w = sum(f["weight"] for f in def_factors)
        def_combine_parts = []
        for f in def_factors:
            w = f["weight"] / total_w
            def_combine_parts.append(f'def_{f["name"]} * {w:.4f}')
        def_combine_expr = " + ".join(def_combine_parts)
    else:
        def_combine_expr = "pd.DataFrame()"

    # 生成 regime 组合（多个 regime 取平均）
    regime_names = [f'regime_{f["name"]}' for f in regime_factors]
    if regime_names:
        regime_expr = f"({' + '.join(regime_names)}) / {len(regime_names)}"
    else:
        regime_expr = "pd.Series(0.0, index=all_dates)"

    # 生成 filter 组合（多个 filter 取乘积 = AND）
    filter_names = [f'filter_{f["name"]}' for f in filter_factors]

    # 需要原始动量（用于 SPY 对比）
    spy_margin = combiner.get("spy_outperform_margin", 0.0)
    vol_threshold = combiner.get("vol_threshold", 0.01)

    # 组装完整策略
    strategy_code = f'''"""
Aurum 多资产轮动策略 — 自动组装自因子库
因子数量: {len(enabled)}
组装时间: {datetime.now().isoformat()}
"""
import pandas as pd
import numpy as np

CASH = "{cash}"
OFFENSIVE = {offensive}
DEFENSIVE = {defensive}

# ════════════════════════════════════════════
#  内联因子函数
# ════════════════════════════════════════════

{"".join(f + chr(10) + chr(10) for f in factor_functions)}
# ════════════════════════════════════════════
#  组合器 + 信号生成
# ════════════════════════════════════════════

MOM_PERIODS = [21, 63, 126, 252]
MOM_WEIGHTS = [1, 2, 4, 6]

def generate_signals(prices):
    # 计算日期并集
    all_dates = None
    for df in prices.values():
        idx = df.index
        if all_dates is None:
            all_dates = idx
        else:
            all_dates = all_dates.union(idx)
    all_dates = all_dates.sort_values()

    # 计算所有因子
{chr(10).join(factor_compute_lines)}

    # 组合进攻型评分
    off_score = {off_combine_expr}

    # 组合防御型评分
    def_score = {def_combine_expr}

    # 市场 regime（0=正常, 1=高风险）
    regime = {regime_expr}

    # 计算原始动量（用于 SPY 对比）
    momentums = {{}}
    for name, df in prices.items():
        close = df["close"].reindex(all_dates)
        mom_values = []
        for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
            mom = close.pct_change(period).shift(1)
            mom_values.append(mom * weight)
        momentums[name] = sum(mom_values) / sum(MOM_WEIGHTS)
    mom_df = pd.DataFrame(momentums).reindex(all_dates)

    # 生成信号
    signals = pd.Series(CASH, index=all_dates)
    current_asset = CASH

    for i, date in enumerate(all_dates):
        # 月度再平衡
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        row_off = off_score.loc[date].dropna() if date in off_score.index else pd.Series(dtype=float)
        row_mom = mom_df.loc[date].dropna() if date in mom_df.index else pd.Series(dtype=float)
        row_def = def_score.loc[date].dropna() if date in def_score.index else pd.Series(dtype=float)

        if len(row_off) == 0:
            signals.iloc[i] = current_asset
            continue

        # 选进攻型最强资产
        best_off = row_off.idxmax()
        best_off_mom = row_mom.get(best_off, -1)
        spy_mom = row_mom.get("SPY", 0)

        # 波动率调整门槛
        vol_adj = {vol_threshold} if (date in regime.index and pd.notna(regime.loc[date]) and regime.loc[date] > 0.5) else 0.0

        if best_off_mom >= spy_mom + {spy_margin} + vol_adj:
            current_asset = best_off
        else:
            # 防御型选择（带 filter）
            if len(row_def) > 0:
                def_sorted = row_def.sort_values(ascending=False)
                selected = None
                for asset in def_sorted.index:
                    if asset == CASH:
                        selected = CASH
                        break
                    # 应用所有 filter
                    pass_filter = True
{chr(10).join(f"                    if date in {fn}.index and asset in {fn}.columns:" + chr(10) + f"                        if pd.notna({fn}.loc[date, asset]) and {fn}.loc[date, asset] < 0.5:" + chr(10) + f"                            pass_filter = False" for fn in filter_names) if filter_names else "                    pass"}
                    if pass_filter:
                        selected = asset
                        break
                current_asset = selected if selected else CASH
            else:
                current_asset = CASH

        signals.iloc[i] = current_asset

    return signals
'''
    return strategy_code


# ════════════════════════════════════════════
#  因子质量评估
# ════════════════════════════════════════════

def evaluate_factor_quality(factor_code, factor_category, factor_scope,
                            all_prices, factor_cfg, main_cfg):
    """
    独立评估一个因子的质量。

    返回: dict with ic_mean, ic_std, ic_ir, max_correlation
    """
    import pickle
    import tempfile

    # 准备数据
    offensive = factor_cfg["offensive"]
    defensive = factor_cfg["defensive"]
    scope_assets = offensive if factor_scope == "offensive" else defensive

    # 用沙盒执行因子代码
    runner = f'''
import pandas as pd
import numpy as np
import pickle
import json
import sys

data_path = sys.argv[1]
factor_path = sys.argv[2]
scope = json.loads(sys.argv[3])

with open(data_path, "rb") as f:
    prices = pickle.load(f)

# 计算日期并集
all_dates = None
for df in prices.values():
    idx = df.index
    if all_dates is None:
        all_dates = idx
    else:
        all_dates = all_dates.union(idx)
all_dates = all_dates.sort_values()

# 执行因子
exec(open(factor_path).read(), globals())
result = compute(prices, all_dates, scope)

# 输出
if isinstance(result, pd.DataFrame):
    output = {{"type": "dataframe", "data": result.to_json()}}
elif isinstance(result, pd.Series):
    output = {{"type": "series", "data": result.to_json()}}
print(json.dumps(output))
'''
    import json

    data_path = "data_cache/multi_eval.pkl"
    runner_fd, runner_path = tempfile.mkstemp(suffix="_frunner.py")
    factor_fd, factor_path = tempfile.mkstemp(suffix="_factor.py")

    try:
        with os.fdopen(runner_fd, "w") as f:
            f.write(runner)
        with os.fdopen(factor_fd, "w") as f:
            f.write(factor_code)

        result = subprocess.run(
            [sys.executable, runner_path, data_path, factor_path,
             json.dumps(scope_assets)],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            return {"error": result.stderr[-500:]}

        output = json.loads(result.stdout)

        import pandas as pd
        from io import StringIO

        if output["type"] == "dataframe":
            factor_scores = pd.read_json(StringIO(output["data"]))
        elif output["type"] == "series":
            factor_scores = pd.read_json(StringIO(output["data"]), typ="series")
        else:
            return {"error": "Unknown output type"}

    except Exception as e:
        return {"error": str(e)[:200]}
    finally:
        for p in (runner_path, factor_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    # 计算 IC（信息系数）
    # 仅对 asset factor（DataFrame）计算
    if not isinstance(factor_scores, type(None)) and hasattr(factor_scores, 'columns'):
        import pandas as pd
        ics = []
        for year in range(2009, 2025):
            year_mask = factor_scores.index.year == year
            if year_mask.sum() < 20:
                continue

            # 计算下个月收益
            for name in factor_scores.columns:
                if name not in all_prices:
                    continue
                close = all_prices[name]["close"].reindex(factor_scores.index)
                fwd_ret = close.pct_change(21).shift(-21)  # 未来 1 月收益

                year_scores = factor_scores.loc[year_mask, name].dropna()
                year_rets = fwd_ret.reindex(year_scores.index).dropna()

                common = year_scores.index.intersection(year_rets.index)
                if len(common) < 20:
                    continue

                from scipy import stats
                ic, _ = stats.spearmanr(
                    year_scores.loc[common], year_rets.loc[common]
                )
                if not np.isnan(ic):
                    ics.append(ic)

        ic_mean = np.mean(ics) if ics else 0
        ic_std = np.std(ics) if ics else 1
        ic_ir = ic_mean / ic_std if ic_std > 1e-9 else 0

        # 计算与现有因子的最大相关性
        max_corr = 0.0
        # 简化：跳过相关性检查（MVP）

        return {
            "ic_mean": round(float(ic_mean), 4),
            "ic_std": round(float(ic_std), 4),
            "ic_ir": round(float(ic_ir), 4),
            "max_correlation": round(float(max_corr), 4),
            "n_observations": len(ics),
        }
    else:
        # regime / filter 因子不计算 IC
        return {
            "ic_mean": 0, "ic_std": 0, "ic_ir": 0,
            "max_correlation": 0, "n_observations": 0,
            "note": "regime/filter factor, IC not applicable",
        }


# ════════════════════════════════════════════
#  策略评估（复用 infra）
# ════════════════════════════════════════════

def evaluate_strategy(strategy_code, data_path, all_prices, config):
    """执行组装后的策略并评分。"""
    signals = run_strategy(strategy_code, data_path)

    sub_results = walk_forward_rotation(
        signals, all_prices,
        cash_asset=config.get("cash_asset", "SHY"),
        benchmark=config.get("benchmark", "SPY"),
        cost=config.get("cost_per_trade", 0.001),
        sub_period=config.get("sub_period", "yearly"),
    )

    s = score(
        sub_results,
        max_dd_limit=config.get("max_drawdown_limit", 0.30),
        min_sharpe=config.get("min_sharpe", 0.0),
    )

    if sub_results:
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


# ════════════════════════════════════════════
#  LLM 交互
# ════════════════════════════════════════════

def ask_new_factor(client, model, factor_cfg, current_score, current_agg,
                   history, program_md):
    """请求 LLM 提出一个新因子。"""
    factor_summary = get_factor_summary(factor_cfg)
    alloc = current_agg.get("allocation", {})
    alloc_str = ", ".join(
        f"{k}: {v:.0%}" for k, v in sorted(alloc.items(), key=lambda x: -x[1])
    )

    prompt = f"""{program_md}

---
## 当前因子库
{factor_summary}

## 当前策略表现（多年度 walk-forward 平均）
- 综合评分: {current_score:.4f}
- Sharpe: {current_agg.get('sharpe', 0):.4f}
- 年化收益: {current_agg.get('ann_return', 0):.2%}
- 超额收益(vs SPY): {current_agg.get('excess_return', 0):.2%}
- 最大回撤: {current_agg.get('max_drawdown', 0):.2%}
- 现金占比: {current_agg.get('cash_pct', 0):.2%}
- 资产配置: {alloc_str}

## 最近实验历史
{history}

请提出一个新因子。严格按输出格式。
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.7,
    )

    text = response.choices[0].message.content
    if not text:
        raise ValueError("LLM returned empty response")

    # 解析
    hypothesis = ""
    if "HYPOTHESIS:" in text:
        hyp_part = text.split("FACTOR_NAME:")[0] if "FACTOR_NAME:" in text else text.split("CODE:")[0]
        hypothesis = hyp_part.split("HYPOTHESIS:")[-1].strip()

    factor_name = "unnamed_factor"
    if "FACTOR_NAME:" in text:
        fn_part = text.split("CATEGORY:")[0] if "CATEGORY:" in text else text.split("CODE:")[0]
        factor_name = fn_part.split("FACTOR_NAME:")[-1].strip()
        factor_name = re.sub(r'[^a-zA-Z0-9_]', '_', factor_name).strip('_').lower()

    category = "offensive"
    if "CATEGORY:" in text:
        cat_part = text.split("CODE:")[0]
        cat_raw = cat_part.split("CATEGORY:")[-1].strip().lower()
        for valid_cat in ["offensive", "defensive", "regime", "filter"]:
            if valid_cat in cat_raw:
                category = valid_cat
                break

    code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not code_blocks:
        raise ValueError("Failed to parse code block from LLM response")

    code = code_blocks[-1].strip()
    if "def compute(" not in code:
        raise ValueError("Code missing compute() function")

    return factor_name, category, code, hypothesis


# ════════════════════════════════════════════
#  日志
# ════════════════════════════════════════════

def log_experiment(**kwargs):
    headers = [
        "timestamp", "iteration", "model", "action", "factor_name",
        "category", "score", "delta", "sharpe", "excess_return",
        "ic_ir", "status", "hypothesis",
    ]
    if not os.path.exists(RESULTS_FILE):
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            f.write("\t".join(headers) + "\n")

    row = [
        kwargs.get("timestamp", datetime.now().isoformat()),
        str(kwargs.get("iteration", 0)),
        kwargs.get("model", ""),
        kwargs.get("action", "NEW_FACTOR"),
        kwargs.get("factor_name", ""),
        kwargs.get("category", ""),
        f"{kwargs.get('score', 0):.6f}",
        f"{kwargs.get('delta', 0):.6f}",
        f"{kwargs.get('sharpe', 0):.4f}",
        f"{kwargs.get('excess_return', 0):.4f}",
        f"{kwargs.get('ic_ir', 0):.4f}",
        kwargs.get("status", ""),
        kwargs.get("hypothesis", "").replace("\t", " ").replace("\n", " ")[:120],
    ]
    with open(RESULTS_FILE, "a") as f:
        f.write("\t".join(row) + "\n")


def get_history(n=30):
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
        if len(parts) >= 13:
            status = parts[11]
            icon = "✅" if status == "KEPT" else "❌" if status == "DISCARDED" else "💥"
            formatted.append(
                f"{icon} [{parts[4]:20s}] score_delta={parts[7]} ic_ir={parts[10]} | {parts[12]}"
            )
    return "\n".join(formatted) if formatted else "（暂无历史记录）"


# ════════════════════════════════════════════
#  Git
# ════════════════════════════════════════════

def git_commit(message):
    subprocess.run(["git", "add", "factors/", FACTOR_CONFIG, STRATEGY_PATH],
                   capture_output=True)
    subprocess.run(["git", "commit", "-m", message], capture_output=True)


# ════════════════════════════════════════════
#  主循环
# ════════════════════════════════════════════

def main():
    main_cfg = load_yaml(MAIN_CONFIG)
    factor_cfg = load_yaml(FACTOR_CONFIG)
    universe = main_cfg["universe"]

    # ── 数据准备 ──
    print(f"🏗️  Aurum 因子进化引擎 — 加载 {len(universe)} 个资产...")
    all_prices = get_multi_prices(universe, main_cfg["eval_start"], main_cfg["eval_end"])
    data_path = "data_cache/multi_eval.pkl"
    save_multi_prices(all_prices, data_path)

    # ── 组装并评估基线 ──
    print("\n📊 组装策略并评估基线...")
    baseline_code = assemble_strategy(factor_cfg)
    with open(STRATEGY_PATH, "w") as f:
        f.write(baseline_code)

    current_score, current_agg = evaluate_strategy(
        baseline_code, data_path, all_prices, main_cfg
    )

    alloc_str = ", ".join(
        f"{k}:{v:.0%}" for k, v in
        sorted(current_agg["allocation"].items(), key=lambda x: -x[1])[:5]
    )
    print(f"   基线评分: {current_score:.4f}")
    print(f"   Sharpe: {current_agg['sharpe']:.4f} | "
          f"年化: {current_agg['ann_return']:.2%} | "
          f"超额: {current_agg['excess_return']:.2%} | "
          f"MaxDD: {current_agg['max_drawdown']:.2%}")
    print(f"   配置: {alloc_str}")
    print(f"   因子数: {len(get_enabled_factors(factor_cfg))}")

    # ── LLM ──
    llm_client = create_client(main_cfg)
    models = main_cfg["llm"]["models"]
    program_md = open(PROGRAM_MD).read()

    evo_cfg = factor_cfg.get("evolution", {})
    iterations = evo_cfg.get("iterations", 100)
    history_window = evo_cfg.get("history_window", 30)

    kept_count = 0
    crash_count = 0
    factor_counter = len(factor_cfg["factors"])

    # ── 主循环 ──
    print(f"\n🔄 开始因子进化 ({iterations} 次迭代)...\n")

    for i in range(1, iterations + 1):
        model = models[(i - 1) % len(models)]

        try:
            history = get_history(history_window)

            # 1. LLM 提出新因子
            factor_name, category, factor_code, hypothesis = ask_new_factor(
                llm_client, model, factor_cfg,
                current_score, current_agg, history, program_md,
            )

            # 去重命名
            factor_counter += 1
            safe_name = f"evolved_{factor_counter:03d}_{factor_name[:30]}"
            factor_file = f"factors/{safe_name}.py"

            # 2. 保存因子文件
            with open(factor_file, "w") as f:
                f.write(factor_code)

            # 3. 因子质量评估
            scope = "offensive" if category in ("offensive", "regime") else "defensive"
            quality = evaluate_factor_quality(
                factor_code, category, scope,
                all_prices, factor_cfg, main_cfg,
            )

            if "error" in quality:
                raise RuntimeError(f"Factor failed: {quality['error']}")

            ic_ir = quality.get("ic_ir", 0)

            # 4. 临时添加到配置
            factor_cfg["factors"][safe_name] = {
                "file": factor_file,
                "category": category,
                "scope": scope,
                "weight": 1.0,
                "enabled": True,
            }

            # 5. 重新组装策略并评估
            new_code = assemble_strategy(factor_cfg)
            new_score, new_agg = evaluate_strategy(
                new_code, data_path, all_prices, main_cfg,
            )

            delta = new_score - current_score
            kept = delta > 0

            status = "KEPT" if kept else "DISCARDED"
            icon = "✅" if kept else "❌"

            log_experiment(
                iteration=i, model=model, action="NEW_FACTOR",
                factor_name=safe_name, category=category,
                score=new_score, delta=delta,
                sharpe=new_agg["sharpe"],
                excess_return=new_agg["excess_return"],
                ic_ir=ic_ir, status=status, hypothesis=hypothesis,
            )

            print(
                f"{icon} #{i:3d} [{safe_name:35s}] "
                f"score={new_score:+.4f} (Δ{delta:+.4f}) "
                f"IC_IR={ic_ir:+.3f} "
                f"| {hypothesis[:45]}"
            )

            if kept:
                kept_count += 1
                current_score = new_score
                current_agg = new_agg

                # 保存更新后的配置和策略
                with open(STRATEGY_PATH, "w") as f:
                    f.write(new_code)
                save_yaml(factor_cfg, FACTOR_CONFIG)
                git_commit(
                    f"factor #{i}: +{safe_name} score={new_score:.4f} | {hypothesis[:50]}"
                )
            else:
                # 回滚：禁用因子
                factor_cfg["factors"][safe_name]["enabled"] = False
                # 删除因子文件
                try:
                    os.unlink(factor_file)
                except OSError:
                    pass
                del factor_cfg["factors"][safe_name]

        except KeyboardInterrupt:
            print("\n⏹️  用户中断")
            break
        except Exception as e:
            crash_count += 1
            error_msg = str(e)[:100]
            print(f"💥 #{i:3d} [{model:15s}] {error_msg}")
            log_experiment(
                iteration=i, model=model, action="NEW_FACTOR",
                factor_name="CRASH", category="",
                score=0, delta=0, sharpe=0, excess_return=0,
                ic_ir=0, status="CRASH",
                hypothesis=f"ERROR: {error_msg}",
            )
            # 清理可能半写入的因子
            if 'safe_name' in dir() and safe_name in factor_cfg.get("factors", {}):
                del factor_cfg["factors"][safe_name]
            continue

    # ── 总结 ──
    total = i if "i" in dir() else 0
    enabled = get_enabled_factors(factor_cfg)
    print(f"\n{'='*60}")
    print(f"🏁 因子进化完成!")
    print(f"   迭代: {total} | 新增因子: {kept_count} | 崩溃: {crash_count}")
    print(f"   最终因子数: {len(enabled)}")
    print(f"   最终评分: {current_score:.4f}")
    print(f"   Sharpe: {current_agg['sharpe']:.4f}")
    print(f"   年化: {current_agg['ann_return']:.2%}")
    print(f"   超额(vs SPY): {current_agg['excess_return']:.2%}")
    print(f"   最大回撤: {current_agg['max_drawdown']:.2%}")
    print(f"\n   启用的因子:")
    for name, cfg in enabled.items():
        print(f"     [{cfg['category']:10s}] {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aurum factor evolution loop")
    parser.add_argument("-n", "--iterations", type=int, default=None)
    args = parser.parse_args()

    if args.iterations is not None:
        _fcfg = load_yaml(FACTOR_CONFIG)
        _fcfg.setdefault("evolution", {})["iterations"] = args.iterations
        save_yaml(_fcfg, FACTOR_CONFIG)

    main()
