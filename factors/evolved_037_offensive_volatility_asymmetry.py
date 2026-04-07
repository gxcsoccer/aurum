"""
Factor: offensive_volatility_asymmetry
Category: offensive
Description: 衡量进攻型资产上行波动率与下行波动率的比率。上行波动主导表明健康趋势，下行波动主导表明脆弱趋势，与总波动率惩罚正交。
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
    
    window = 60  # 60 日滚动窗口
    
    for asset in assets:
        if asset not in prices:
            scores[asset] = np.nan
            continue
        
        df = prices[asset].reindex(all_dates)
        close = df['close']
        
        # 计算日收益率
        returns = close.pct_change()
        
        # 分离上行和下行收益率
        upside_returns = returns.clip(lower=0)  # 只保留正收益
        downside_returns = returns.clip(upper=0)  # 只保留负收益
        
        # 计算上行和下行半方差（semi-variance）
        upside_var = upside_returns.rolling(window=window).var()
        downside_var = downside_returns.rolling(window=window).var()
        
        # 计算波动不对称比率（上行波动/下行波动）
        # 避免除以零，添加小常数
        vol_asymmetry = np.sqrt(upside_var) / (np.sqrt(downside_var) + 1e-8)
        
        # 前视偏差修正：使用 shift(1)
        signal = vol_asymmetry.shift(1)
        
        # 标准化到 0-1 范围便于解释
        # 比率>1 表示上行波动主导（好），比率<1 表示下行波动主导（差）
        scores[asset] = signal
    
    # 填充 NaN
    scores = scores.fillna(method='ffill').fillna(1.0)
    
    return scores