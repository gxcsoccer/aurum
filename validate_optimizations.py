"""
验证三种优化是否带来正向改进：
  1. 因子相关性剪枝（移除冗余因子）
  2. Ridge 权重优化（学习最优线性组合）
  3. ML 组合器（梯度提升非线性组合）

对比维度：
  - 训练期 (2008-2024 walk-forward)
  - Holdout 期 (2025)
  - 训练期内部分割：2008-2017 训练，2018-2024 测试

"正向优化" = 同时改善 holdout 表现 + 训练期 out-of-sample 表现。
"""
import importlib.util
import os
import sys
from io import StringIO

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

from infra.backtest import backtest_rotation, walk_forward_rotation
from infra.data import get_multi_prices
from infra.scorer import score

# ── 配置 ──
MAIN_CONFIG = "config.yaml"
FACTOR_CONFIG = "factor_config.yaml"


def load_yaml_(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_factor_module(filepath):
    """动态加载因子模块。"""
    spec = importlib.util.spec_from_file_location("factor_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute


def compute_all_factor_scores(prices, all_dates, factor_cfg):
    """一次性计算所有因子的输出。"""
    offensive = factor_cfg["offensive"]
    defensive = factor_cfg["defensive"]
    scores = {}
    for name, cfg in factor_cfg["factors"].items():
        if not cfg.get("enabled", True):
            continue
        compute_fn = load_factor_module(cfg["file"])
        scope = cfg.get("scope", "offensive")
        scope_assets = offensive if scope == "offensive" else defensive
        try:
            result = compute_fn(prices, all_dates, scope_assets)
        except Exception as e:
            print(f"  ⚠️  factor {name} failed: {e}")
            continue
        scores[name] = {
            "category": cfg.get("category", "offensive"),
            "data": result,
        }
    return scores


# ════════════════════════════════════════════
#  工具：将因子展平为 (date, asset) → 特征向量
# ════════════════════════════════════════════

def build_feature_matrix(factor_scores, prices, target_assets, all_dates):
    """
    构造训练矩阵（向量化）：
      X: shape (n_samples, n_features)
      y: shape (n_samples,)，未来 21 日收益
    缺失值用 0 填充而非丢弃。仅丢弃 target 缺失或全特征缺失的行。
    """
    feature_names = sorted(factor_scores.keys())
    target_dates = pd.DatetimeIndex(all_dates)

    # 为每个资产构造一个 (n_dates, n_features) 矩阵
    asset_matrices = {}
    for asset in target_assets:
        if asset not in prices:
            continue
        cols = []
        for fn in feature_names:
            fdata = factor_scores[fn]["data"]
            cat = factor_scores[fn]["category"]
            if cat == "regime" and isinstance(fdata, pd.Series):
                col = fdata.reindex(target_dates).values
            elif isinstance(fdata, pd.DataFrame):
                if asset in fdata.columns:
                    col = fdata[asset].reindex(target_dates).values
                else:
                    col = np.full(len(target_dates), np.nan)
            else:
                col = np.full(len(target_dates), np.nan)
            cols.append(col)
        mat = np.column_stack(cols)  # (n_dates, n_features)
        asset_matrices[asset] = mat

    # 构造 forward returns
    rows_X = []
    rows_y = []
    meta = []
    for asset, mat in asset_matrices.items():
        close = prices[asset]["close"].reindex(target_dates)
        fwd = close.pct_change(21).shift(-21).values
        for i in range(len(target_dates)):
            if np.isnan(fwd[i]):
                continue
            features = mat[i].copy()
            # 全 NaN 行跳过
            if np.all(np.isnan(features)):
                continue
            features = np.nan_to_num(features, nan=0.0)
            rows_X.append(features)
            rows_y.append(float(fwd[i]))
            meta.append((target_dates[i], asset))

    return np.array(rows_X), np.array(rows_y), meta, feature_names


# ════════════════════════════════════════════
#  组合器：用模型预测得分
# ════════════════════════════════════════════

def predict_scores_with_model(model, factor_scores, prices, target_assets,
                               all_dates, feature_names):
    """用训练好的模型为每个 (date, asset) 计算预测得分（向量化）。"""
    target_dates = pd.DatetimeIndex(all_dates)
    score_dict = {}
    for asset in target_assets:
        if asset not in prices:
            continue
        cols = []
        for fn in feature_names:
            fdata = factor_scores[fn]["data"]
            cat = factor_scores[fn]["category"]
            if cat == "regime" and isinstance(fdata, pd.Series):
                col = fdata.reindex(target_dates).values
            elif isinstance(fdata, pd.DataFrame) and asset in fdata.columns:
                col = fdata[asset].reindex(target_dates).values
            else:
                col = np.zeros(len(target_dates))
            cols.append(col)
        X = np.column_stack(cols)
        X = np.nan_to_num(X, nan=0.0)
        preds = model.predict(X)
        score_dict[asset] = pd.Series(preds, index=target_dates)
    return pd.DataFrame(score_dict).reindex(target_dates)


# ════════════════════════════════════════════
#  组合器：等权 / 剪枝后等权
# ════════════════════════════════════════════

def equal_weight_scores(factor_scores, target_assets, all_dates, category):
    """简单加权平均（所有 asset 因子等权）。"""
    matched = [
        scores["data"] for scores in factor_scores.values()
        if scores["category"] == category and isinstance(scores["data"], pd.DataFrame)
    ]
    if not matched:
        return pd.DataFrame()
    # 对齐
    asset_cols = sorted(set().union(*[df.columns for df in matched]))
    asset_cols = [a for a in asset_cols if a in target_assets]
    summed = None
    count = 0
    for df in matched:
        df_aligned = df.reindex(all_dates).reindex(columns=asset_cols)
        if summed is None:
            summed = df_aligned.fillna(0)
        else:
            summed = summed + df_aligned.fillna(0)
        count += 1
    return summed / count if count > 0 else pd.DataFrame()


# ════════════════════════════════════════════
#  信号生成（与 assemble_strategy 等效）
# ════════════════════════════════════════════

def generate_signals(off_score_df, def_score_df, factor_scores, prices,
                     all_dates, offensive, defensive, cash):
    """根据组合好的得分生成持仓信号。"""
    # 计算原始动量供 SPY 比较
    MOM_PERIODS = [21, 63, 126, 252]
    MOM_WEIGHTS = [1, 2, 4, 6]
    momentums = {}
    for name, df in prices.items():
        close = df["close"].reindex(all_dates)
        mom_values = []
        for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
            mom = close.pct_change(period).shift(1)
            mom_values.append(mom * weight)
        momentums[name] = sum(mom_values) / sum(MOM_WEIGHTS)
    mom_df = pd.DataFrame(momentums).reindex(all_dates)

    # 提取 regime 因子（多个取平均）
    regime_series = []
    for name, info in factor_scores.items():
        if info["category"] == "regime" and isinstance(info["data"], pd.Series):
            regime_series.append(info["data"].reindex(all_dates).fillna(0))
    regime = sum(regime_series) / len(regime_series) if regime_series else pd.Series(0.0, index=all_dates)

    # 提取 filter 因子（多个取乘积）
    filters = []
    for name, info in factor_scores.items():
        if info["category"] == "filter" and isinstance(info["data"], pd.DataFrame):
            filters.append(info["data"].reindex(all_dates))

    signals = pd.Series(cash, index=all_dates)
    current_asset = cash
    VOL_THRESHOLD = 0.01

    for i, date in enumerate(all_dates):
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        row_off = off_score_df.loc[date].dropna() if date in off_score_df.index else pd.Series(dtype=float)
        row_mom = mom_df.loc[date].dropna() if date in mom_df.index else pd.Series(dtype=float)
        row_def = def_score_df.loc[date].dropna() if date in def_score_df.index else pd.Series(dtype=float)

        if len(row_off) == 0:
            signals.iloc[i] = current_asset
            continue

        best_off = row_off.idxmax()
        best_off_mom = row_mom.get(best_off, -1)
        spy_mom = row_mom.get("SPY", 0)

        vol_adj = VOL_THRESHOLD if (date in regime.index and pd.notna(regime.loc[date]) and regime.loc[date] > 0.5) else 0.0

        if best_off_mom >= spy_mom + vol_adj:
            current_asset = best_off
        else:
            if len(row_def) > 0:
                def_sorted = row_def.sort_values(ascending=False)
                selected = None
                for asset in def_sorted.index:
                    if asset == cash:
                        selected = cash
                        break
                    pass_filter = True
                    for filt_df in filters:
                        if date in filt_df.index and asset in filt_df.columns:
                            v = filt_df.loc[date, asset]
                            if pd.notna(v) and v < 0.5:
                                pass_filter = False
                                break
                    if pass_filter:
                        selected = asset
                        break
                current_asset = selected if selected else cash
            else:
                current_asset = cash

        signals.iloc[i] = current_asset

    return signals


# ════════════════════════════════════════════
#  评估
# ════════════════════════════════════════════

def evaluate(signals, all_prices, period_label, eval_start=None, eval_end=None):
    """评估信号，返回 dict 包含关键指标。"""
    if eval_start and eval_end:
        mask = (signals.index >= eval_start) & (signals.index <= eval_end)
        signals = signals[mask]
        sub_prices = {}
        for name, df in all_prices.items():
            sub_df = df.loc[df.index.isin(signals.index)]
            if len(sub_df) >= 20:
                sub_prices[name] = sub_df
    else:
        sub_prices = all_prices

    result = backtest_rotation(signals, sub_prices, cash_asset="SHY", benchmark="SPY", cost=0.001)

    # walk-forward 评分
    sub_results = walk_forward_rotation(
        signals, sub_prices, cash_asset="SHY", benchmark="SPY", cost=0.001, sub_period="yearly"
    )
    s = score(sub_results, max_dd_limit=0.30, min_sharpe=0.0) if sub_results else -999.0

    return {
        "label": period_label,
        "score": s,
        "sharpe": result["sharpe"],
        "ann_return": result["ann_return"],
        "max_dd": result["max_drawdown"],
        "excess": result["excess_return"],
        "n_switches": result["n_switches"],
    }


# ════════════════════════════════════════════
#  相关性剪枝
# ════════════════════════════════════════════

def compute_factor_correlations(factor_scores):
    """计算所有 asset 因子之间的相关性矩阵（展平为长向量后算）。"""
    flat = {}
    for name, info in factor_scores.items():
        d = info["data"]
        if isinstance(d, pd.DataFrame):
            arr = d.values.flatten()
        elif isinstance(d, pd.Series):
            arr = d.values
        else:
            continue
        # 去掉 NaN
        flat[name] = arr

    names = list(flat.keys())
    n = len(names)
    corr = pd.DataFrame(np.eye(n), index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            x = flat[a]
            y = flat[b]
            # 长度对齐
            min_len = min(len(x), len(y))
            x = x[-min_len:]
            y = y[-min_len:]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 100:
                continue
            try:
                c = np.corrcoef(x[mask], y[mask])[0, 1]
                if np.isnan(c):
                    c = 0.0
            except Exception:
                c = 0.0
            corr.loc[a, b] = c
            corr.loc[b, a] = c
    return corr


def prune_correlated_factors(factor_scores, threshold=0.85):
    """贪心移除高相关因子（保留先加入的）。"""
    corr = compute_factor_correlations(factor_scores)
    names = list(factor_scores.keys())
    to_drop = set()
    for i, a in enumerate(names):
        if a in to_drop:
            continue
        for j in range(i + 1, len(names)):
            b = names[j]
            if b in to_drop:
                continue
            if abs(corr.loc[a, b]) > threshold:
                to_drop.add(b)
    pruned = {k: v for k, v in factor_scores.items() if k not in to_drop}
    return pruned, list(to_drop), corr


# ════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════

def main():
    main_cfg = load_yaml_(MAIN_CONFIG)
    factor_cfg = load_yaml_(FACTOR_CONFIG)
    universe = main_cfg["universe"]
    offensive = factor_cfg["offensive"]
    defensive = factor_cfg["defensive"]
    cash = factor_cfg["cash"]

    print("=" * 70)
    print("🔬 因子优化验证实验")
    print("=" * 70)

    # ── 加载训练 + holdout 数据 ──
    print("\n📥 加载数据...")
    train_prices = get_multi_prices(universe, "2008-01-01", "2024-12-31")
    full_prices = get_multi_prices(universe, "2008-01-01", "2025-12-31")

    # 公共日期
    train_dates = None
    for df in train_prices.values():
        if train_dates is None:
            train_dates = df.index
        else:
            train_dates = train_dates.union(df.index)
    train_dates = train_dates.sort_values()

    full_dates = None
    for df in full_prices.values():
        if full_dates is None:
            full_dates = df.index
        else:
            full_dates = full_dates.union(df.index)
    full_dates = full_dates.sort_values()

    # ── 计算所有因子 ──
    print("\n🧮 计算所有因子...")
    factor_scores_full = compute_all_factor_scores(full_prices, full_dates, factor_cfg)
    print(f"   已计算 {len(factor_scores_full)} 个因子")

    # ── 划分内部训练 / 内部测试 ──
    train_inner_end = pd.Timestamp("2017-12-31")
    test_inner_start = pd.Timestamp("2018-01-01")
    test_inner_end = pd.Timestamp("2024-12-31")
    holdout_start = pd.Timestamp("2025-01-01")
    holdout_end = pd.Timestamp("2025-12-31")

    print(f"\n📅 时间窗口划分:")
    print(f"   组合器训练: 2008-01-01 ~ 2017-12-31")
    print(f"   组合器测试: 2018-01-01 ~ 2024-12-31")
    print(f"   Holdout:    2025-01-01 ~ 2025-12-31")

    results = {}

    # ════════════════════════════════════════════
    # 变体 A: 基线（等权）
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【变体 A】基线 - 等权组合")
    print("─" * 70)

    off_A = equal_weight_scores(factor_scores_full, offensive, full_dates, "offensive")
    def_A = equal_weight_scores(factor_scores_full, defensive, full_dates, "defensive")
    sig_A = generate_signals(off_A, def_A, factor_scores_full, full_prices,
                              full_dates, offensive, defensive, cash)
    r_train_A = evaluate(sig_A, full_prices, "Train(2008-2024)", "2008-01-01", "2024-12-31")
    r_test_A = evaluate(sig_A, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
    r_hold_A = evaluate(sig_A, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
    results["A_baseline"] = (r_train_A, r_test_A, r_hold_A)
    for r in (r_train_A, r_test_A, r_hold_A):
        print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}  MaxDD={r['max_dd']:.2%}")

    # ════════════════════════════════════════════
    # 变体 B: 相关性剪枝
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【变体 B】相关性剪枝 (阈值 0.85)")
    print("─" * 70)

    pruned_factors, dropped, corr_matrix = prune_correlated_factors(factor_scores_full, threshold=0.85)
    print(f"   原始因子: {len(factor_scores_full)}")
    print(f"   保留因子: {len(pruned_factors)}")
    print(f"   移除因子: {dropped if dropped else '（无高相关因子）'}")

    # 打印相关性矩阵摘要
    print("\n   因子相关性矩阵 (绝对值 > 0.5 的对):")
    names = list(factor_scores_full.keys())
    high_corr_pairs = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            c = corr_matrix.loc[a, b]
            if abs(c) > 0.5:
                high_corr_pairs.append((a, b, c))
    if high_corr_pairs:
        for a, b, c in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            print(f"     {a[:40]:40s} ↔ {b[:40]:40s}  corr={c:+.3f}")
    else:
        print("     （无高相关因子对）")

    if not dropped:
        print("\n   ⏭️  无因子被移除，跳过 B 变体（与 A 等同）")
        results["B_pruned"] = results["A_baseline"]
    else:
        off_B = equal_weight_scores(pruned_factors, offensive, full_dates, "offensive")
        def_B = equal_weight_scores(pruned_factors, defensive, full_dates, "defensive")
        sig_B = generate_signals(off_B, def_B, pruned_factors, full_prices,
                                  full_dates, offensive, defensive, cash)
        r_train_B = evaluate(sig_B, full_prices, "Train(2008-2024)", "2008-01-01", "2024-12-31")
        r_test_B = evaluate(sig_B, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
        r_hold_B = evaluate(sig_B, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
        results["B_pruned"] = (r_train_B, r_test_B, r_hold_B)
        for r in (r_train_B, r_test_B, r_hold_B):
            print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}  MaxDD={r['max_dd']:.2%}")

    # ════════════════════════════════════════════
    # 变体 C: Ridge 权重优化
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【变体 C】Ridge 线性权重优化")
    print("─" * 70)

    from sklearn.linear_model import Ridge

    # 构造训练集（仅用 2008-2017 数据训练组合器）
    inner_train_dates = [d for d in full_dates if d <= train_inner_end]

    # 训练 offensive 模型
    X_off, y_off, meta_off, feat_names_off = build_feature_matrix(
        factor_scores_full, full_prices, offensive, inner_train_dates
    )
    print(f"   Offensive 训练样本: {len(X_off)}")

    if len(X_off) > 100:
        ridge_off = Ridge(alpha=1.0)
        ridge_off.fit(X_off, y_off)

        # 预测全期 offensive 得分
        off_C = predict_scores_with_model(ridge_off, factor_scores_full, full_prices,
                                          offensive, full_dates, feat_names_off)

        print(f"   Ridge offensive 权重: {dict(zip(feat_names_off, ridge_off.coef_.round(4)))}")
    else:
        print("   ⚠️  训练样本不足，回退到等权")
        off_C = off_A

    # 训练 defensive 模型
    X_def, y_def, meta_def, feat_names_def = build_feature_matrix(
        factor_scores_full, full_prices, defensive, inner_train_dates
    )
    print(f"   Defensive 训练样本: {len(X_def)}")

    if len(X_def) > 100:
        ridge_def = Ridge(alpha=1.0)
        ridge_def.fit(X_def, y_def)
        def_C = predict_scores_with_model(ridge_def, factor_scores_full, full_prices,
                                          defensive, full_dates, feat_names_def)
        print(f"   Ridge defensive 权重: {dict(zip(feat_names_def, ridge_def.coef_.round(4)))}")
    else:
        print("   ⚠️  训练样本不足，回退到等权")
        def_C = def_A

    sig_C = generate_signals(off_C, def_C, factor_scores_full, full_prices,
                              full_dates, offensive, defensive, cash)
    r_train_C = evaluate(sig_C, full_prices, "Train(2008-2024)", "2008-01-01", "2024-12-31")
    r_test_C = evaluate(sig_C, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
    r_hold_C = evaluate(sig_C, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
    results["C_ridge"] = (r_train_C, r_test_C, r_hold_C)
    for r in (r_train_C, r_test_C, r_hold_C):
        print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}  MaxDD={r['max_dd']:.2%}")

    # ════════════════════════════════════════════
    # 变体 D: 梯度提升 ML 组合器
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【变体 D】梯度提升 ML 组合器")
    print("─" * 70)

    from sklearn.ensemble import GradientBoostingRegressor

    if len(X_off) > 100:
        gbm_off = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        gbm_off.fit(X_off, y_off)
        off_D = predict_scores_with_model(gbm_off, factor_scores_full, full_prices,
                                          offensive, full_dates, feat_names_off)
        print(f"   GBM offensive 训练完成 (n_est=100, depth=3)")
        print(f"   特征重要性: {dict(zip(feat_names_off, gbm_off.feature_importances_.round(3)))}")
    else:
        off_D = off_A

    if len(X_def) > 100:
        gbm_def = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        gbm_def.fit(X_def, y_def)
        def_D = predict_scores_with_model(gbm_def, factor_scores_full, full_prices,
                                          defensive, full_dates, feat_names_def)
        print(f"   GBM defensive 训练完成")
    else:
        def_D = def_A

    sig_D = generate_signals(off_D, def_D, factor_scores_full, full_prices,
                              full_dates, offensive, defensive, cash)
    r_train_D = evaluate(sig_D, full_prices, "Train(2008-2024)", "2008-01-01", "2024-12-31")
    r_test_D = evaluate(sig_D, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
    r_hold_D = evaluate(sig_D, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
    results["D_gbm"] = (r_train_D, r_test_D, r_hold_D)
    for r in (r_train_D, r_test_D, r_hold_D):
        print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}  MaxDD={r['max_dd']:.2%}")

    # ════════════════════════════════════════════
    # 汇总
    # ════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 汇总对比")
    print("=" * 70)

    variants = [
        ("A 基线 (等权)", "A_baseline"),
        ("B 剪枝", "B_pruned"),
        ("C Ridge 权重", "C_ridge"),
        ("D GBM ML", "D_gbm"),
    ]

    print(f"\n{'变体':<20s} {'阶段':<18s} {'Score':>10s} {'Sharpe':>8s} {'年化':>8s} {'超额':>8s} {'MaxDD':>8s}")
    print("─" * 80)
    for label, key in variants:
        for r in results[key]:
            print(f"{label:<20s} {r['label']:<18s} {r['score']:>+10.4f} {r['sharpe']:>+8.3f} "
                  f"{r['ann_return']:>+8.2%} {r['excess']:>+8.2%} {r['max_dd']:>8.2%}")
        print()

    # ── 判断正向优化 ──
    print("\n🎯 是否为正向优化（同时改善 Test 和 Holdout）?")
    print("─" * 70)
    baseline_test = results["A_baseline"][1]
    baseline_hold = results["A_baseline"][2]

    for label, key in variants[1:]:
        test = results[key][1]
        hold = results[key][2]
        test_better = test["score"] > baseline_test["score"]
        hold_better = hold["score"] > baseline_hold["score"]
        excess_test_better = test["excess"] > baseline_test["excess"]
        excess_hold_better = hold["excess"] > baseline_hold["excess"]

        verdict = "✅ 正向" if (hold_better and test_better) else \
                  "⚠️  仅训练改善（疑似过拟合）" if test_better else \
                  "⚠️  仅 Holdout 改善（运气？）" if hold_better else \
                  "❌ 负向"
        print(f"  {label:<20s}  Test Δscore={test['score']-baseline_test['score']:+.4f}  "
              f"Hold Δscore={hold['score']-baseline_hold['score']:+.4f}  "
              f"Hold Δexcess={hold['excess']-baseline_hold['excess']:+.2%}  →  {verdict}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
