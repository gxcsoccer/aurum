"""
验证 v2：测试三种新方向
  E. Walk-forward Ridge（每年重训，避免过拟合训练期）
  F. Walk-forward GBM（同上，非线性版）
  G. Rank-target Ridge（target 改为收益排名，匹配选择语义）

依赖前置脚本 validate_optimizations.py 的工具函数。
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

from infra.backtest import backtest_rotation, walk_forward_rotation
from infra.data import get_multi_prices
from infra.scorer import score
from validate_optimizations import (
    load_yaml_, compute_all_factor_scores, equal_weight_scores,
    generate_signals, evaluate, MAIN_CONFIG, FACTOR_CONFIG,
    build_feature_matrix, predict_scores_with_model,
)


# ════════════════════════════════════════════
#  Walk-forward 重训
# ════════════════════════════════════════════

def walk_forward_predict(model_class, model_kwargs, factor_scores, prices,
                         target_assets, full_dates, train_start_year=2008,
                         eval_start_year=2018):
    """
    走查窗口重训：每年开始时，用所有过去数据重新训练模型。
    返回拼接后的 score DataFrame。
    """
    feature_names = sorted(factor_scores.keys())
    out = pd.DataFrame(index=full_dates, columns=target_assets, dtype=float)

    for year in range(eval_start_year, 2026):
        train_end = pd.Timestamp(f"{year-1}-12-31")
        pred_start = pd.Timestamp(f"{year}-01-01")
        pred_end = pd.Timestamp(f"{year}-12-31")

        train_dates = [d for d in full_dates if d <= train_end]
        pred_dates = [d for d in full_dates if pred_start <= d <= pred_end]

        if len(train_dates) < 252 or len(pred_dates) == 0:
            continue

        X_train, y_train, _, _ = build_feature_matrix(
            factor_scores, prices, target_assets, train_dates
        )
        if len(X_train) < 100:
            continue

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

        # 预测 pred_dates 的得分
        pred_scores = predict_scores_with_model(
            model, factor_scores, prices, target_assets, pred_dates, feature_names
        )
        for col in pred_scores.columns:
            out.loc[pred_scores.index, col] = pred_scores[col].values

    # 2018 之前的日期：用基线（防止 NaN 干扰）
    early_dates = [d for d in full_dates if d.year < eval_start_year]
    if early_dates:
        baseline = equal_weight_scores(factor_scores, target_assets, full_dates,
                                        "offensive" if "SPY" in target_assets else "defensive")
        for col in baseline.columns:
            if col in out.columns:
                out.loc[early_dates, col] = baseline.loc[early_dates, col].values

    return out


# ════════════════════════════════════════════
#  Rank-target 优化
# ════════════════════════════════════════════

def build_rank_feature_matrix(factor_scores, prices, target_assets, all_dates):
    """
    构造 rank-target 训练矩阵：
      每个日期内，对所有 target_assets 的 forward return 做 cross-sectional 排名
      target = 该资产在当日的 forward return 排名（标准化为 [0, 1]）
    """
    feature_names = sorted(factor_scores.keys())
    target_dates = pd.DatetimeIndex(all_dates)

    # 构造 forward returns DataFrame
    fwd_dict = {}
    for asset in target_assets:
        if asset not in prices:
            continue
        close = prices[asset]["close"].reindex(target_dates)
        fwd_dict[asset] = close.pct_change(21).shift(-21)
    fwd_df = pd.DataFrame(fwd_dict)

    # Cross-sectional 排名（每日内排名）
    rank_df = fwd_df.rank(axis=1, pct=True)  # [0, 1]

    # 构造 (date, asset) 特征
    rows_X = []
    rows_y = []
    for asset in target_assets:
        if asset not in fwd_dict:
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
                col = np.full(len(target_dates), np.nan)
            cols.append(col)
        mat = np.column_stack(cols)
        rank_target = rank_df[asset].values

        for i in range(len(target_dates)):
            if np.isnan(rank_target[i]):
                continue
            features = mat[i].copy()
            if np.all(np.isnan(features)):
                continue
            features = np.nan_to_num(features, nan=0.0)
            rows_X.append(features)
            rows_y.append(float(rank_target[i]))

    return np.array(rows_X), np.array(rows_y), feature_names


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
    print("🔬 因子优化验证 v2: walk-forward + rank loss")
    print("=" * 70)

    print("\n📥 加载数据...")
    full_prices = get_multi_prices(universe, "2008-01-01", "2025-12-31")
    full_dates = None
    for df in full_prices.values():
        if full_dates is None:
            full_dates = df.index
        else:
            full_dates = full_dates.union(df.index)
    full_dates = full_dates.sort_values()

    print("\n🧮 计算因子...")
    factor_scores = compute_all_factor_scores(full_prices, full_dates, factor_cfg)
    print(f"   {len(factor_scores)} 个因子")

    results = {}

    # ════════════════════════════════════════════
    # 基线 A
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【基线 A】等权组合")
    print("─" * 70)
    off_A = equal_weight_scores(factor_scores, offensive, full_dates, "offensive")
    def_A = equal_weight_scores(factor_scores, defensive, full_dates, "defensive")
    sig_A = generate_signals(off_A, def_A, factor_scores, full_prices,
                              full_dates, offensive, defensive, cash)
    r_test_A = evaluate(sig_A, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
    r_hold_A = evaluate(sig_A, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
    results["A"] = (r_test_A, r_hold_A)
    for r in (r_test_A, r_hold_A):
        print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}")

    # ════════════════════════════════════════════
    # 变体 E: Walk-forward Ridge
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【变体 E】Walk-forward Ridge（每年重训）")
    print("─" * 70)
    from sklearn.linear_model import Ridge

    print("   训练 offensive...")
    off_E = walk_forward_predict(Ridge, {"alpha": 1.0}, factor_scores, full_prices,
                                  offensive, full_dates)
    print("   训练 defensive...")
    def_E = walk_forward_predict(Ridge, {"alpha": 1.0}, factor_scores, full_prices,
                                  defensive, full_dates)
    sig_E = generate_signals(off_E, def_E, factor_scores, full_prices,
                              full_dates, offensive, defensive, cash)
    r_test_E = evaluate(sig_E, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
    r_hold_E = evaluate(sig_E, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
    results["E"] = (r_test_E, r_hold_E)
    for r in (r_test_E, r_hold_E):
        print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}")

    # ════════════════════════════════════════════
    # 变体 F: Walk-forward GBM
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【变体 F】Walk-forward GBM（每年重训）")
    print("─" * 70)
    from sklearn.ensemble import GradientBoostingRegressor

    gbm_kwargs = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05, "random_state": 42}
    print("   训练 offensive...")
    off_F = walk_forward_predict(GradientBoostingRegressor, gbm_kwargs,
                                  factor_scores, full_prices, offensive, full_dates)
    print("   训练 defensive...")
    def_F = walk_forward_predict(GradientBoostingRegressor, gbm_kwargs,
                                  factor_scores, full_prices, defensive, full_dates)
    sig_F = generate_signals(off_F, def_F, factor_scores, full_prices,
                              full_dates, offensive, defensive, cash)
    r_test_F = evaluate(sig_F, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
    r_hold_F = evaluate(sig_F, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
    results["F"] = (r_test_F, r_hold_F)
    for r in (r_test_F, r_hold_F):
        print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}")

    # ════════════════════════════════════════════
    # 变体 G: Rank-target Ridge (no walk-forward)
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【变体 G】Rank-target Ridge（target=收益排名）")
    print("─" * 70)

    inner_train_dates = [d for d in full_dates if d <= pd.Timestamp("2017-12-31")]
    X_off_r, y_off_r, feat_names = build_rank_feature_matrix(
        factor_scores, full_prices, offensive, inner_train_dates
    )
    X_def_r, y_def_r, _ = build_rank_feature_matrix(
        factor_scores, full_prices, defensive, inner_train_dates
    )
    print(f"   Offensive 样本: {len(X_off_r)}, Defensive 样本: {len(X_def_r)}")

    if len(X_off_r) > 100 and len(X_def_r) > 100:
        ridge_off_r = Ridge(alpha=1.0)
        ridge_off_r.fit(X_off_r, y_off_r)
        ridge_def_r = Ridge(alpha=1.0)
        ridge_def_r.fit(X_def_r, y_def_r)

        print(f"   Ridge offensive 权重: {dict(zip(feat_names, ridge_off_r.coef_.round(3)))}")

        off_G = predict_scores_with_model(ridge_off_r, factor_scores, full_prices,
                                          offensive, full_dates, feat_names)
        def_G = predict_scores_with_model(ridge_def_r, factor_scores, full_prices,
                                          defensive, full_dates, feat_names)
        sig_G = generate_signals(off_G, def_G, factor_scores, full_prices,
                                  full_dates, offensive, defensive, cash)
        r_test_G = evaluate(sig_G, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
        r_hold_G = evaluate(sig_G, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
        results["G"] = (r_test_G, r_hold_G)
        for r in (r_test_G, r_hold_G):
            print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}")
    else:
        print("   ⚠️ 样本不足，跳过")
        results["G"] = (r_test_A, r_hold_A)

    # ════════════════════════════════════════════
    # 变体 H: Rank-target Walk-forward Ridge（组合 E + G 思路）
    # ════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("【变体 H】Rank-target Walk-forward Ridge（E + G 组合）")
    print("─" * 70)

    def walk_forward_rank_predict(factor_scores, prices, target_assets, full_dates):
        feature_names = sorted(factor_scores.keys())
        out = pd.DataFrame(index=full_dates, columns=target_assets, dtype=float)
        for year in range(2018, 2026):
            train_end = pd.Timestamp(f"{year-1}-12-31")
            pred_start = pd.Timestamp(f"{year}-01-01")
            pred_end = pd.Timestamp(f"{year}-12-31")
            train_dates = [d for d in full_dates if d <= train_end]
            pred_dates = [d for d in full_dates if pred_start <= d <= pred_end]
            if len(train_dates) < 252 or len(pred_dates) == 0:
                continue
            X_train, y_train, _ = build_rank_feature_matrix(
                factor_scores, prices, target_assets, train_dates
            )
            if len(X_train) < 100:
                continue
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            pred_scores = predict_scores_with_model(
                model, factor_scores, prices, target_assets, pred_dates, feature_names
            )
            for col in pred_scores.columns:
                out.loc[pred_scores.index, col] = pred_scores[col].values
        # fill 2008-2017 with baseline
        early = [d for d in full_dates if d.year < 2018]
        baseline = equal_weight_scores(factor_scores, target_assets, full_dates,
                                        "offensive" if "SPY" in target_assets else "defensive")
        for col in baseline.columns:
            if col in out.columns:
                out.loc[early, col] = baseline.loc[early, col].values
        return out

    print("   训练 offensive...")
    off_H = walk_forward_rank_predict(factor_scores, full_prices, offensive, full_dates)
    print("   训练 defensive...")
    def_H = walk_forward_rank_predict(factor_scores, full_prices, defensive, full_dates)
    sig_H = generate_signals(off_H, def_H, factor_scores, full_prices,
                              full_dates, offensive, defensive, cash)
    r_test_H = evaluate(sig_H, full_prices, "Test(2018-2024)", "2018-01-01", "2024-12-31")
    r_hold_H = evaluate(sig_H, full_prices, "Holdout(2025)", "2025-01-01", "2025-12-31")
    results["H"] = (r_test_H, r_hold_H)
    for r in (r_test_H, r_hold_H):
        print(f"   {r['label']:20s}  score={r['score']:+.4f}  Sharpe={r['sharpe']:+.3f}  超额={r['excess']:+.2%}")

    # ════════════════════════════════════════════
    # 汇总
    # ════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 汇总")
    print("=" * 70)

    labels = {
        "A": "A 基线 (等权)",
        "E": "E WF Ridge",
        "F": "F WF GBM",
        "G": "G Rank Ridge",
        "H": "H WF Rank Ridge",
    }
    print(f"\n{'变体':<22s} {'阶段':<18s} {'Score':>10s} {'Sharpe':>8s} {'超额':>10s} {'MaxDD':>8s}")
    print("─" * 80)
    for key in ("A", "E", "F", "G", "H"):
        if key not in results:
            continue
        for r in results[key]:
            print(f"{labels[key]:<22s} {r['label']:<18s} {r['score']:>+10.4f} {r['sharpe']:>+8.3f} "
                  f"{r['excess']:>+10.2%} {r['max_dd']:>8.2%}")
        print()

    base_test = results["A"][0]
    base_hold = results["A"][1]
    print("\n🎯 是否为正向优化（Test 和 Holdout 都改善）?")
    print("─" * 70)
    for key in ("E", "F", "G", "H"):
        if key not in results:
            continue
        test = results[key][0]
        hold = results[key][1]
        test_better = test["score"] > base_test["score"]
        hold_better = hold["score"] > base_hold["score"]
        verdict = "✅ 正向" if (hold_better and test_better) else \
                  "⚠️  仅 Test 改善" if test_better else \
                  "⚠️  仅 Holdout 改善" if hold_better else \
                  "❌ 负向"
        print(f"  {labels[key]:<22s}  Test Δ={test['score']-base_test['score']:+.4f}  "
              f"Hold Δ={hold['score']-base_hold['score']:+.4f}  →  {verdict}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
