"""
Factor: offensive_volume_confirmation
Category: offensive
Description: 衡量进攻型资产价格上涨时的成交量确认度。计算上涨日成交量与下跌日成交量的比率，并乘以价格动量，确保只有放量上涨的资产获得高分。
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
    scores = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    for asset in assets:
        if asset not in prices:
            scores[asset] = np.nan
            continue
        
        df = prices[asset].copy()
        df = df.reindex(all_dates)
        
        # 计算日收益率
        close = df['close'].fillna(method='ffill')
        returns = close.pct_change()
        
        # 计算成交量
        volume = df['volume'].fillna(method='ffill')
        
        # 上涨日成交量（当日收涨）
        up_volume = volume.where(returns > 0, 0)
        
        # 下跌日成交量（当日收跌）
        down_volume = volume.where(returns < 0, 0)
        
        # 滚动 20 日上涨/下跌成交量比（加 1 避免除零）
        up_vol_sum = up_volume.rolling(window=20, min_periods=5).sum()
        down_vol_sum = down_volume.rolling(window=20, min_periods=5).sum()
        vol_ratio = (up_vol_sum / (down_vol_sum + 1)).shift(1)
        
        # 20 日价格动量
        momentum = (close / close.shift(20) - 1).shift(1)
        
        # 成交量确认分数：动量 * 成交量比
        # 只有放量上涨（vol_ratio > 1）且动量为正时得高分
        volume_confirmation = vol_ratio * momentum
        
        # 标准化到 0-1 范围（使用滚动分位数）
        rolling_median = volume_confirmation.rolling(window=60, min_periods=20).median()
        rolling_std = volume_confirmation.rolling(window=60, min_periods=20).std()
        
        score = (volume_confirmation - rolling_median) / (rolling_std + 1e-8)
        
        # 限制极端值
        score = score.clip(-3, 3)
        
        # 归一化到 0-1
        score = (score + 3) / 6
        
        scores[asset] = score
    
    return scores