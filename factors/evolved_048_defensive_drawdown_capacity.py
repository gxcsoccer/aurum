"""
Factor: defensive_drawdown_capacity
Category: defensive
Description: Measures remaining protective capacity of defensive assets based on drawdown depth - shallow drawdowns indicate more dry powder for hedging during market stress.
"""
import pandas as pd
import numpy as np

def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        DataFrame(index=all_dates, columns=assets, values=float scores)
    """
    scores = pd.DataFrame(index=all_dates, columns=assets, values=0.0)
    
    for asset in assets:
        if asset not in prices:
            continue
        
        close = prices[asset]['close'].reindex(all_dates)
        
        # 计算 60 日滚动最高价
        rolling_max = close.rolling(window=60, min_periods=20).max()
        
        # 当前回撤深度（负值）
        drawdown = (close - rolling_max) / (rolling_max + 1e-10)
        
        # 反转：回撤越浅（越接近 0），保护能力越强
        # shift(1) 避免前视偏差
        capacity = -drawdown.shift(1)
        
        # 用滚动分位数归一化，避免极端值
        rolling_min = capacity.rolling(window=252, min_periods=60).min()
        rolling_max_cap = capacity.rolling(window=252, min_periods=60).max()
        rolling_range = rolling_max_cap - rolling_min + 1e-10
        
        capacity_normalized = (capacity - rolling_min) / rolling_range
        
        scores[asset] = capacity_normalized.fillna(0.0)
    
    return scores