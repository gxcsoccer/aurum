"""
Aurum 多资产轮动策略 — agent 可以修改此文件的所有内容
当前策略：多期动量加权 + 进攻型波动率调整 + 防御型纯动量 + 防御型 63 日均线趋势过滤
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
MOM_PERIODS = [21, 63, 126, 252]  # 多期动量：1/3/6/12 个月
MOM_WEIGHTS = [1, 2, 4, 6]        # 权重分配（短期到长期递增）
VOL_LOOKBACK = 210                # 波动率计算期（~10 个月）
SPY_OUTPERFORM_MARGIN = 0.0       # 进攻型资产需要超过 SPY 动量的幅度
CASH = "SHY"                      # 现金等价资产
OFFENSIVE = ["SPY", "QQQ", "EFA", "EEM"]   # 进攻型资产
DEFENSIVE = ["TLT", "GLD", "SHY"]          # 防御型资产
DEFENSIVE_MA_PERIOD = 63          # 防御型资产趋势确认均线周期

# ============ 信号逻辑区 ============
def generate_signals(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """
    输入：prices dict，key=资产名，value=OHLCV DataFrame
    输出：Series，index=日期，values=持有的资产名 (str)

    多期动量加权策略 + 进攻型波动率调整 + 防御型纯动量 + 防御型趋势过滤：
    1. 计算每个资产的多期动量（1/3/6/12 个月加权）
    2. 计算每个资产的波动率（10 个月）
    3. 进攻型资产使用动量/波动率作为评分指标
    4. 防御型资产使用纯动量作为评分指标
    5. 防御型资产需价格高于 63 日均线才考虑持有（否则切到 SHY）
    6. 如果最强进攻型资产动量 >= SPY 动量 → 持有它
    7. 否则 → 切换到防御型资产中纯动量最强的（需通过趋势过滤）
    8. 每月初再平衡一次
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

    # 计算每个资产的多期加权动量和波动率（shift(1) 防止前视偏差）
    momentums = {}
    volatilities = {}
    ma_63 = {}  # 63 日均线用于防御型资产趋势过滤
    
    for name, df in prices.items():
        close = df["close"].reindex(all_dates)
        
        # 多期动量加权
        mom_values = []
        for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
            mom = close.pct_change(period).shift(1)
            mom_values.append(mom * weight)
        # 加权平均动量
        momentums[name] = sum(mom_values) / sum(MOM_WEIGHTS)
        
        # 波动率：10 个月年化波动率
        returns = close.pct_change().shift(1)
        volatilities[name] = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
        
        # 63 日均线（用于防御型资产趋势过滤）
        ma_63[name] = close.rolling(DEFENSIVE_MA_PERIOD).mean().shift(1)

    mom_df = pd.DataFrame(momentums).reindex(all_dates)
    vol_df = pd.DataFrame(volatilities).reindex(all_dates)
    ma_df = pd.DataFrame(ma_63).reindex(all_dates)

    # 计算进攻型资产的波动率调整动量评分
    off_score_df = mom_df[OFFENSIVE] / (vol_df[OFFENSIVE] + 0.0001)

    # 防御型资产使用纯动量评分（不除以波动率）
    def_score_df = mom_df[DEFENSIVE]

    # 生成信号
    signals = pd.Series(CASH, index=all_dates)
    current_asset = CASH

    for i, date in enumerate(all_dates):
        # 月度再平衡：仅在每月第一个交易日调仓
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        row_off_score = off_score_df.loc[date].dropna()
        row_mom = mom_df.loc[date].dropna()
        row_def_score = def_score_df.loc[date].dropna()
        
        if len(row_off_score) == 0:
            signals.iloc[i] = current_asset
            continue

        # 选进攻型资产中波动率调整动量最强的
        if len(row_off_score) > 0:
            best_off = row_off_score.idxmax()
            best_off_mom = row_mom.get(best_off, -1)
        else:
            best_off = CASH
            best_off_mom = -1

        # 相对动量过滤：进攻型资产动量需超过或等于 SPY 动量
        spy_mom = row_mom.get("SPY", -1) if "SPY" in row_mom else 0
        if best_off_mom >= spy_mom + SPY_OUTPERFORM_MARGIN:
            current_asset = best_off
        else:
            # 切换到防御型资产中纯动量最强的（需通过 63 日均线趋势过滤）
            if len(row_def_score) > 0:
                # 按动量排序防御型资产
                def_sorted = row_def_score.sort_values(ascending=False)
                selected_def = None
                
                for def_asset in def_sorted.index:
                    if def_asset == CASH:
                        selected_def = CASH
                        break
                    
                    # 检查防御型资产是否高于 63 日均线
                    asset_price = mom_df.loc[date, def_asset]  # 这里用 momentum 近似代表价格趋势
                    asset_ma = ma_df.loc[date, def_asset]
                    close_price = prices[def_asset]["close"].loc[date] if date in prices[def_asset]["close"].index else None
                    
                    if close_price is not None and close_price > asset_ma:
                        selected_def = def_asset
                        break
                
                if selected_def is None:
                    # 所有防御资产都在 63 日均线下方，切换到 SHY
                    current_asset = CASH
                else:
                    current_asset = selected_def
            else:
                current_asset = CASH

        signals.iloc[i] = current_asset

    return signals