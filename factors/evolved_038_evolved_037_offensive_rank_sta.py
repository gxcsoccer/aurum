"""
Factor: evolved_037_offensive_rank_stability
Category: offensive
Description: 衡量进攻型资产风险调整动量排名的稳定性——排名波动小的资产表明趋势质量更高、资金流入更持续，与趋势一致性和动量耗尽信号正交。
"""
import pandas as pd
import numpy as np

def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame (columns: open/high/low/close/volume)
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        DataFrame(index=all_dates, columns=assets, values=float scores)
    """
    # 只处理进攻型资产
    offensive_assets = [a for a in assets if a in ['SPY', 'QQQ', 'EFA', 'EEM']]
    
    if len(offensive_assets) == 0:
        return pd.DataFrame(index=all_dates, columns=assets, values=0.0)
    
    # 初始化结果
    scores = pd.DataFrame(index=all_dates, columns=assets, values=0.0)
    
    # 计算每个进攻型资产的风险调整动量
    vol_adj_mom = {}
    for asset in offensive_assets:
        if asset not in prices:
            continue
        
        close = prices[asset]['close'].reindex(all_dates)
        
        # 21 日动量
        mom_21 = close.pct_change(21)
        
        # 21 日波动率
        vol_21 = close.pct_change().rolling(21).std()
        
        # 风险调整动量
        vol_adj = mom_21 / (vol_21 + 1e-8)
        
        vol_adj_mom[asset] = vol_adj
    
    # 构建面板数据
    vol_adj_df = pd.DataFrame(vol_adj_mom)
    
    # 计算每个资产的滚动排名稳定性（过去 63 天内排名的标准差）
    lookback = 63
    rank_stability = pd.DataFrame(index=all_dates, columns=offensive_assets)
    
    for date in all_dates:
        idx = all_dates.get_loc(date)
        if idx < lookback:
            rank_stability.loc[date] = np.nan
            continue
        
        window_start = idx - lookback
        window_end = idx
        
        # 获取窗口内的风险调整动量
        window_data = vol_adj_df.iloc[window_start:window_end]
        
        # 计算每日排名
        daily_ranks = window_data.rank(axis=1, ascending=False)
        
        # 对每个资产计算排名标准差（越低越稳定）
        for asset in offensive_assets:
            if asset in daily_ranks.columns:
                rank_std = daily_ranks[asset].std()
                # 转换为稳定性分数（标准差越小，分数越高）
                rank_stability.loc[date, asset] = 1.0 / (rank_std + 0.5)
            else:
                rank_stability.loc[date, asset] = np.nan
    
    # 前视偏差修复：shift(1)
    rank_stability = rank_stability.shift(1)
    
    # 填充进攻型资产分数
    for asset in offensive_assets:
        if asset in rank_stability.columns:
            scores[asset] = rank_stability[asset].fillna(0.0)
    
    return scores