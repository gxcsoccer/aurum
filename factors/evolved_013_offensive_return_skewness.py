"""
Factor: offensive_return_skewness
Category: offensive
Description: 衡量进攻型资产收益分布的偏度——正偏度表明上涨日幅度大于下跌日（健康趋势），负偏度表明下跌日幅度更大（脆弱趋势）。与动量、波动率、趋势一致性正交。
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
            continue
        
        df = prices[asset].copy()
        df = df.reindex(all_dates)
        
        # 计算日收益率
        close = df['close'].fillna(method='ffill')
        returns = close.pct_change()
        
        # 滚动偏度（60 天窗口）- 三阶矩衡量分布不对称性
        # 正偏度：大涨日多于大跌日 = 健康上涨
        # 负偏度：大跌日多于大涨日 = 脆弱上涨（即使总价涨）
        skew_window = 60
        skewness = returns.rolling(window=skew_window, min_periods=20).apply(
            lambda x: x.skew() if len(x) >= 20 else np.nan,
            raw=False
        )
        
        # shift(1) 确保无前视偏差
        skewness = skewness.shift(1)
        
        # 标准化到 0-1 范围（基于历史分位数）
        # 偏度通常在 -3 到 +3 之间
        skewness_normalized = (skewness + 2) / 4  # 大致映射到 0-1
        skewness_normalized = skewness_normalized.clip(0, 1)
        
        scores[asset] = skewness_normalized
    
    return scores.fillna(0.5)  # 中性分数用于缺失值