"""
Factor: defensive_relative_strength_rank
Category: defensive
Description: 衡量防御型资产（TLT/GLD/SHY）之间的相对强度排名——识别当前环境下最优防御资产。不同宏观风险下最优防御资产不同，动态排名能捕捉防御资产内部轮动，优化防御配置。
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
    # 防御型资产列表
    defensive_assets = ['TLT', 'GLD', 'SHY']
    
    # 初始化结果 DataFrame
    scores = pd.DataFrame(index=all_dates, columns=assets, values=0.0, dtype=float)
    
    # 获取防御资产的收盘价并重新索引到所有日期
    close_dict = {}
    for asset in defensive_assets:
        if asset in prices and asset in assets:
            close = prices[asset]['close'].reindex(all_dates)
            close_dict[asset] = close
    
    # 需要至少 2 个防御资产有数据才能计算相对强度
    valid_assets = [a for a in defensive_assets if a in close_dict]
    
    if len(valid_assets) < 2:
        return scores
    
    # 计算多个时间窗口的动量（20 日、60 日、120 日）
    momentum_scores = {}
    for window in [20, 60, 120]:
        for asset in valid_assets:
            close = close_dict[asset]
            # 计算动量（收益率）
            momentum = close.pct_change(window)
            # 确保无前视偏差
            momentum = momentum.shift(1)
            
            if asset not in momentum_scores:
                momentum_scores[asset] = []
            momentum_scores[asset].append(momentum)
    
    # 综合多个时间窗口的动量（等权平均）
    composite_momentum = {}
    for asset in valid_assets:
        # 将多个时间窗口的动量取平均
        momentum_df = pd.DataFrame(momentum_scores[asset]).T
        composite_momentum[asset] = momentum_df.mean(axis=1)
    
    # 计算相对强度排名分数
    # 对于每个日期，将防御资产按动量排序，最高得 1.0，最低得 0.0
    for date in all_dates:
        # 获取该日期各防御资产的动量值
        mom_values = {}
        for asset in valid_assets:
            val = composite_momentum[asset].loc[date]
            if pd.notna(val):
                mom_values[asset] = val
        
        if len(mom_values) < 2:
            continue
        
        # 排序并分配排名分数（0 到 1 之间）
        sorted_assets = sorted(mom_values.keys(), key=lambda x: mom_values[x])
        
        for rank, asset in enumerate(sorted_assets):
            # 归一化排名分数：最低=0.0，最高=1.0，中间线性插值
            if len(sorted_assets) == 1:
                score = 0.5
            else:
                score = rank / (len(sorted_assets) - 1)
            
            scores.loc[date, asset] = score
    
    # 对于非防御资产，分数为 0
    for asset in assets:
        if asset not in defensive_assets:
            scores[asset] = 0.0
    
    return scores