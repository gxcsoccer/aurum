"""
Aurum 多资产轮动策略 — agent 可以修改此文件的所有内容
当前策略：波动率调整动量 + 动态防御资产选择 + 相对 SPY 动量过滤
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 252          # 动量回望期（~12 个月）
VOL_LOOKBACK = 180      # 波动率计算期（~9 个月，从 126 天延长）
SPY_OUTPERFORM_MARGIN = 0.0  # 进攻型资产需要超过 SPY 动量的幅度（0% 表示只需超过或等于 SPY）
CASH = "SHY"            # 现金等价资产
OFFENSIVE = ["SPY", "QQQ", "EFA", "EEM"]   # 进攻型资产
DEFENSIVE = ["TLT", "GLD", "SHY"]          # 防御型资产

# ============ 信号逻辑区 ============
def generate_signals(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """
    输入：prices dict，key=资产名，value=OHLCV DataFrame
    输出：Series，index=日期，values=持有的资产名 (str)

    波动率调整动量策略 + 动态防御资产选择 + 相对 SPY 动量过滤：
    1. 计算每个资产的动量（12 个月）
    2. 计算每个资产的波动率（9 个月）
    3. 使用动量/波动率作为评分指标
    4. 如果最强进攻型资产动量 >= SPY 动量 → 持有它
    5. 否则 → 切换到防御型资产中波动率调整动量最强的
    6. 每月初再平衡一次
    """
    # 获取公共日期
    all_dates = None
    for df in prices.values():
        idx = df.index
        if all_dates is None:
            all_dates = idx
        else:
            all_dates = all_dates.intersection(idx)
    all_dates = all_dates.sort_values()

    # 计算每个资产的动量和波动率（shift(1) 防止前视偏差）
    momentums = {}
    volatilities = {}
    for name, df in prices.items():
        close = df["close"].reindex(all_dates)
        # 动量：12 个月收益率
        momentums[name] = close.pct_change(LOOKBACK).shift(1)
        # 波动率：9 个月年化波动率（从 6 个月延长到 9 个月）
        returns = close.pct_change().shift(1)
        volatilities[name] = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)

    mom_df = pd.DataFrame(momentums).reindex(all_dates)
    vol_df = pd.DataFrame(volatilities).reindex(all_dates)

    # 计算波动率调整动量评分
    # 避免除以零，添加小值
    score_df = mom_df / (vol_df + 0.0001)

    # 生成信号
    signals = pd.Series(CASH, index=all_dates)
    current_asset = CASH

    for i, date in enumerate(all_dates):
        # 月度再平衡：仅在每月第一个交易日调仓
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        row_score = score_df.loc[date].dropna()
        row_mom = mom_df.loc[date].dropna()
        
        if len(row_score) == 0:
            signals.iloc[i] = current_asset
            continue

        # 选进攻型资产中波动率调整动量最强的
        off_score = {k: row_score[k] for k in OFFENSIVE if k in row_score}
        off_mom = {k: row_mom[k] for k in OFFENSIVE if k in row_mom}
        
        if off_score and off_mom:
            # 按波动率调整评分排序
            best_off = max(off_score, key=off_score.get)
            best_off_mom = off_mom[best_off]
        else:
            best_off = CASH
            best_off_mom = -1

        # 相对动量过滤：进攻型资产动量需超过或等于 SPY 动量
        spy_mom = row_mom.get("SPY", -1) if "SPY" in row_mom else 0
        if best_off_mom >= spy_mom + SPY_OUTPERFORM_MARGIN:
            current_asset = best_off
        else:
            # 切换到防御型资产中波动率调整动量最强的
            def_score = {k: row_score[k] for k in DEFENSIVE if k in row_score}
            if def_score:
                current_asset = max(def_score, key=def_score.get)
            else:
                current_asset = CASH

        signals.iloc[i] = current_asset

    return signals