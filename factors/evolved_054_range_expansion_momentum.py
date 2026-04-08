"""
Factor: range_expansion_momentum
Category: offensive
Description: 捕捉交易区间压缩后的扩张信号——区间扩张表明机构活动增加，往往先于持续性方向突破，与基于收益率的波动率因子正交（区间可以宽但收盘接近，或区间窄但跳空），提供"能量积蓄"的独立信息。
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
    scores = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    for asset in assets:
        if asset not in prices:
            scores[asset] = np.nan
            continue
        
        df = prices[asset].reindex(all_dates)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 交易区间（高 - 低）
        tr = high - low
        
        # 区间相对扩张度：当前区间 / 20 日均区间
        # 使用 min_periods=10 确保有足够数据
        tr_ma = tr.rolling(20, min_periods=10).mean()
        tr_ratio = tr / tr_ma
        
        # 动量成分（20 日收益率）
        momentum = close.pct_change(20)
        
        # 组合：区间扩张 × 动量方向
        # 区间扩张 + 正动量 = 高进攻分数（突破向上）
        # 区间扩张 + 负动量 = 低进攻分数（突破向下）
        # 区间压缩 = 低分数（方向不明）
        score = momentum.shift(1) * tr_ratio.shift(1)
        
        scores[asset] = score
    
    return scores