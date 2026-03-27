"""
Aurum 多资产轮动策略 — agent 可以修改此文件的所有内容
当前策略：改进的 Dual Momentum（带正动量阈值）
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 252          # 动量回望期（~12 个月）
MOM_THRESHOLD = 0.01    # 进攻型资产需要的最小正动量阈值（1%）
CASH = "SHY"            # 现金等价资产
OFFENSIVE = ["SPY", "QQQ", "EFA", "EEM"]   # 进攻型资产
DEFENSIVE = ["TLT", "GLD", "SHY"]          # 防御型资产

# ============ 信号逻辑区 ============
def generate_signals(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """
    输入：prices dict，key=资产名，value=OHLCV DataFrame
    输出：Series，index=日期，values=持有的资产名 (str)

    改进的 Dual Momentum：
    1. 在进攻型资产中选动量最强的
    2. 如果最强的动量 > 阈值 (1%) → 持有它
    3. 否则 → 切换到现金 (SHY)
    4. 每月初再平衡一次
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

    # 计算每个资产的动量（shift(1) 防止前视偏差）
    momentums = {}
    for name, df in prices.items():
        close = df["close"].reindex(all_dates)
        momentums[name] = close.pct_change(LOOKBACK).shift(1)

    mom_df = pd.DataFrame(momentums).reindex(all_dates)

    # 生成信号
    signals = pd.Series(CASH, index=all_dates)
    current_asset = CASH

    for i, date in enumerate(all_dates):
        # 月度再平衡：仅在每月第一个交易日调仓
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        row = mom_df.loc[date].dropna()
        if len(row) == 0:
            signals.iloc[i] = current_asset
            continue

        # 选进攻型资产中动量最强的
        off_mom = {k: row[k] for k in OFFENSIVE if k in row}
        if off_mom:
            best_off = max(off_mom, key=off_mom.get)
            best_off_mom = off_mom[best_off]
        else:
            best_off = CASH
            best_off_mom = -1

        # 绝对动量过滤：动量超过阈值才持有进攻型
        if best_off_mom > MOM_THRESHOLD:
            current_asset = best_off
        else:
            # 直接切换到现金，不尝试选择防御型资产
            current_asset = CASH

        signals.iloc[i] = current_asset

    return signals