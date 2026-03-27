import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 252          # 动量回望期（~12 个月）
VOL_LOOKBACK = 60       # 波动率回望期（~3 个月）
CASH = "SHY"            # 现金等价资产
OFFENSIVE = ["SPY", "QQQ", "EFA", "EEM"]   # 进攻型资产
DEFENSIVE = ["TLT", "GLD", "SHY"]          # 防御型资产

# ============ 信号逻辑区 ============
def generate_signals(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """
    输入：prices dict，key=资产名，value=OHLCV DataFrame
    输出：Series，index=日期，values=持有的资产名 (str)

    变异：波动率调整动量 (Volatility-Adjusted Momentum)
    1. 计算每个资产的动量和波动率
    2. 评分 = 动量 / 波动率
    3. 在进攻型资产中选评分最高的
    4. 如果最强资产的原始动量为正 → 持有它
    5. 否则 → 切换到防御型资产中评分最高的
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
        # 波动率：60 日收益率标准差
        volatilities[name] = close.pct_change().rolling(VOL_LOOKBACK).std().shift(1)

    mom_df = pd.DataFrame(momentums).reindex(all_dates)
    vol_df = pd.DataFrame(volatilities).reindex(all_dates)
    
    # 计算波动率调整动量评分 (避免除以 0)
    score_df = mom_df / (vol_df + 1e-6)

    # 生成信号
    signals = pd.Series(CASH, index=all_dates)
    current_asset = CASH

    for i, date in enumerate(all_dates):
        # 月度再平衡：仅在每月第一个交易日调仓
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        # 获取当日数据
        row_score = score_df.loc[date].dropna()
        row_mom = mom_df.loc[date].dropna()
        
        if len(row_score) == 0:
            signals.iloc[i] = current_asset
            continue

        # 选进攻型资产中评分最高的
        off_score = {k: row_score[k] for k in OFFENSIVE if k in row_score}
        if off_score:
            best_off = max(off_score, key=off_score.get)
            best_off_mom = row_mom.get(best_off, -1)
        else:
            best_off = CASH
            best_off_mom = -1

        # 绝对动量过滤：正动量才持有进攻型 (使用原始动量确保收益为正)
        if best_off_mom > 0:
            current_asset = best_off
        else:
            # 选防御型中评分最高的
            def_score = {k: row_score[k] for k in DEFENSIVE if k in row_score}
            if def_score:
                current_asset = max(def_score, key=def_score.get)
            else:
                current_asset = CASH

        signals.iloc[i] = current_asset

    return signals