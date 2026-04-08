"""
Factor: offensive_return_autocorrelation
Category: offensive
Description: 衡量进攻型资产日收益的一阶自相关性——正自相关表明动量持续性高、趋势可信；负自相关表明均值回归、趋势脆弱。与趋势一致性（上涨日比例）正交，捕捉收益的时间序列结构而非方向频率。
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
        
        df = prices[asset].copy()
        df = df.reindex(all_dates)
        close = df['close']
        
        # 计算日收益
        returns = close.pct_change()
        
        # 滚动自相关（滞后 1 期）- 使用 60 天窗口捕捉中期动量结构
        window = 60
        autocorr = returns.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 10 else np.nan,
            raw=False
        )
        
        # 前视偏差修正：shift(1) 确保信号基于昨日及之前数据
        autocorr = autocorr.shift(1)
        
        # 归一化到 0-1 范围（自相关理论范围 -1 到 1）
        # 加 1 后除以 2，使 -1->0, 0->0.5, 1->1
        normalized = (autocorr + 1) / 2
        
        # 处理边界值
        normalized = normalized.clip(0, 1)
        
        scores[asset] = normalized
    
    # 填充前向数据（初期无数据时）
    scores = scores.fillna(method='ffill').fillna(0.5)
    
    return scores