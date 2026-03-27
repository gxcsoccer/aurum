"""
Aurum 多资产轮动策略 — agent 可以修改此文件的所有内容
当前策略：Dual Momentum + 短期动量过滤 (放宽阈值) + 波动率调整动量排名 + 退出滞后效应
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK_LONG = 252       # 长期动量回望期（~12 个月）
LOOKBACK_SHORT = 63       # 短期动量回望期（~3 个月）
VOL_LOOKBACK = 63         # 波动率计算回望期（~3 个月）
SHORT_MOM_THRESHOLD = -0.10  # 短期动量阈值，低于此值则避险（从 -5% 放宽到 -10%）
CASH = "SHY"              # 现金等价资产
OFFENSIVE = ["SPY", "QQQ", "EFA", "EEM"]  # 进攻型资产
DEFENSIVE = ["TLT", "GLD", "SHY"]  # 防御型资产

# ============ 信号逻辑区 ============
def generate_signals(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """
    输入：prices dict，key=资产名，value=OHLCV DataFrame
    输出：Series，index=日期，values=持有的资产名 (str)
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

    # 计算每个资产的长期动量、短期动量、波动率（shift(1) 防止前视偏差）
    mom_long = {}
    mom_short = {}
    volatility = {}
    for name, df in prices.items():
        close = df["close"].reindex(all_dates)
        returns = close.pct_change()
        mom_long[name] = close.pct_change(LOOKBACK_LONG).shift(1)
        mom_short[name] = close.pct_change(LOOKBACK_SHORT).shift(1)
        volatility[name] = returns.rolling(VOL_LOOKBACK).std().shift(1)

    mom_long_df = pd.DataFrame(mom_long).reindex(all_dates)
    mom_short_df = pd.DataFrame(mom_short).reindex(all_dates)
    vol_df = pd.DataFrame(volatility).reindex(all_dates)

    # 计算波动率调整动量（风险调整后的动量得分）
    vol_adj_mom = {}
    for name in mom_long_df.columns:
        vol_adj_mom[name] = mom_long_df[name] / (vol_df[name] + 1e-6)  # 避免除零
    
    vol_adj_mom_df = pd.DataFrame(vol_adj_mom).reindex(all_dates)

    # 生成信号
    signals = pd.Series(CASH, index=all_dates)
    current_asset = CASH
    in_offensive = False  # 追踪当前是否在进攻型资产中

    for i, date in enumerate(all_dates):
        # 月度再平衡
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        row_long = mom_long_df.loc[date].dropna()
        row_short = mom_short_df.loc[date].dropna()
        row_vol_adj = vol_adj_mom_df.loc[date].dropna()
        
        if len(row_long) == 0:
            signals.iloc[i] = current_asset
            continue

        # 选进攻型中波动率调整动量最强的
        off_mom = {k: row_vol_adj[k] for k in OFFENSIVE if k in row_vol_adj}
        if off_mom:
            best_off = max(off_mom, key=off_mom.get)
            best_off_mom_long = row_long.get(best_off, -1) if best_off in row_long else -1
            best_off_mom_short = row_short.get(best_off, 0) if best_off in row_short else 0
        else:
            best_off = CASH
            best_off_mom_long = -1
            best_off_mom_short = 0

        # 滞后效应逻辑：退出条件比进入条件更严格
        if in_offensive:
            # 已经在进攻型资产中，只有当长期和短期动量同时转负时才退出
            if best_off_mom_long > 0 or best_off_mom_short > SHORT_MOM_THRESHOLD:
                # 至少一个条件满足，继续保持进攻
                current_asset = best_off
            else:
                # 两个条件都失败，切换到防御
                def_mom = {k: row_long[k] for k in DEFENSIVE if k in row_long}
                if def_mom:
                    current_asset = max(def_mom, key=def_mom.get)
                else:
                    current_asset = CASH
                in_offensive = False
        else:
            # 当前在防御中，需要两个条件都满足才能进入进攻
            if best_off_mom_long > 0 and best_off_mom_short > SHORT_MOM_THRESHOLD:
                current_asset = best_off
                in_offensive = True
            else:
                # 切换到防御型资产中选长期动量最强的
                def_mom = {k: row_long[k] for k in DEFENSIVE if k in row_long}
                if def_mom:
                    current_asset = max(def_mom, key=def_mom.get)
                else:
                    current_asset = CASH
                in_offensive = False

        signals.iloc[i] = current_asset

    return signals