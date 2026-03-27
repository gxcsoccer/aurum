"""
Aurum 多资产轮动策略 — agent 可以修改此文件的所有内容
当前策略：Dual Momentum + 短期动量过滤 (放宽阈值) + 波动率调整动量排名 + 退出滞后效应 + 防御资产基于进攻动量强度选择 + 自适应防御阈值
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

# 自适应防御阈值参数
VOL_PERCENTILE_HIGH = 0.70  # 波动率高分位阈值
VOL_PERCENTILE_LOW = 0.30   # 波动率低分位阈值
DEF_THRESHOLD_HIGH_VOL = -0.03  # 高波动时防御切换阈值（更宽松，-3% 就切防御）
DEF_THRESHOLD_LOW_VOL = -0.08   # 低波动时防御切换阈值（更严格，-8% 才切防御）
DEF_THRESHOLD_NORMAL = -0.05    # 正常波动时防御切换阈值（-5%）

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

    # 计算 SPY 波动率的历史分位数（用于自适应阈值）
    spy_vol = vol_df["SPY"].dropna()
    spy_vol_quantiles = {
        'high': spy_vol.quantile(VOL_PERCENTILE_HIGH),
        'low': spy_vol.quantile(VOL_PERCENTILE_LOW)
    }

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

        # 根据当前 SPY 波动率状态确定防御切换阈值
        current_spy_vol = vol_df.loc[date, "SPY"] if "SPY" in vol_df.columns else 0
        if current_spy_vol > spy_vol_quantiles['high']:
            def_threshold = DEF_THRESHOLD_HIGH_VOL  # 高波动，更保守
        elif current_spy_vol < spy_vol_quantiles['low']:
            def_threshold = DEF_THRESHOLD_LOW_VOL   # 低波动，更激进
        else:
            def_threshold = DEF_THRESHOLD_NORMAL    # 正常波动

        # 防御资产选择逻辑：基于进攻型资产动量强度选择不同防御资产
        def select_defensive(row_vol_adj, row_long, offensive_mom_long, threshold):
            """
            根据进攻型资产动量强度选择防御资产：
            - 进攻动量严重下跌 (mom <= threshold): 选 GLD (危机对冲)
            - 进攻动量温和下跌 (threshold < mom < 0): 选 TLT (通常在温和下跌中表现好)
            - 否则：选波动率调整动量最强的防御资产
            """
            def_vol_adj = {k: row_vol_adj[k] for k in DEFENSIVE if k in row_vol_adj}
            
            if not def_vol_adj:
                return CASH
            
            # 基于进攻型资产长期动量强度选择防御资产
            if offensive_mom_long <= threshold:
                # 严重下跌，优先 GLD 作为危机对冲
                if "GLD" in def_vol_adj:
                    return "GLD"
                elif "TLT" in def_vol_adj:
                    return "TLT"
                else:
                    return CASH
            elif offensive_mom_long < 0:
                # 温和下跌，优先 TLT (通常与股市负相关)
                if "TLT" in def_vol_adj:
                    return "TLT"
                elif "GLD" in def_vol_adj:
                    return "GLD"
                else:
                    return CASH
            else:
                # 进攻动量为正但短期动量触发避险，选波动率调整动量最强的防御资产
                return max(def_vol_adj, key=def_vol_adj.get)

        # 滞后效应逻辑：退出条件比进入条件更严格
        if in_offensive:
            # 已经在进攻型资产中，只有当长期和短期动量同时转负时才退出
            if best_off_mom_long > 0 or best_off_mom_short > SHORT_MOM_THRESHOLD:
                # 至少一个条件满足，继续保持进攻
                current_asset = best_off
            else:
                # 两个条件都失败，切换到防御
                current_asset = select_defensive(row_vol_adj, row_long, best_off_mom_long, def_threshold)
                in_offensive = False
        else:
            # 当前在防御中，需要两个条件都满足才能进入进攻
            if best_off_mom_long > 0 and best_off_mom_short > SHORT_MOM_THRESHOLD:
                current_asset = best_off
                in_offensive = True
            else:
                # 切换到防御型资产
                current_asset = select_defensive(row_vol_adj, row_long, best_off_mom_long, def_threshold)
                in_offensive = False

        signals.iloc[i] = current_asset

    return signals