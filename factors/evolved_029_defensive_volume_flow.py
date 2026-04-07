"""
Factor: defensive_volume_flow
Category: defensive
Description: 捕捉防御型资产的成交量流动信号——当价格上涨伴随异常成交量时，表明机构资金积极建立对冲头寸，给予更高评分。
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
    result_dict = {}
    
    for asset in assets:
        if asset not in prices:
            result_dict[asset] = pd.Series(0.0, index=all_dates)
            continue
            
        df = prices[asset].copy()
        df = df.reindex(all_dates)
        
        close = df['close'].fillna(method='ffill').fillna(method='bfill')
        volume = df['volume'].fillna(method='ffill').fillna(method='bfill')
        
        # 计算价格变化率
        returns = close.pct_change()
        
        # 计算成交量相对基准的比率（20 日滚动均值）
        vol_ma = volume.rolling(window=20, min_periods=5).mean()
        vol_ratio = volume / vol_ma
        
        # 成交量异常度：超过均值的程度
        vol_anomaly = vol_ratio - 1.0
        
        # 价格方向信号
        price_direction = returns
        
        # 量价配合信号：价格上涨且成交量放大时得正分
        # 价格下跌且成交量放大时得负分（恐慌性抛售）
        volume_flow_signal = price_direction * vol_anomaly
        
        # 平滑信号以减少噪音
        volume_flow_smooth = volume_flow_signal.rolling(window=5, min_periods=3).mean()
        
        # 标准化到 0-1 范围
        rolling_mean = volume_flow_smooth.rolling(window=60, min_periods=20).mean()
        rolling_std = volume_flow_smooth.rolling(window=60, min_periods=20).std()
        
        z_score = (volume_flow_smooth - rolling_mean) / (rolling_std + 1e-8)
        
        # 转换为 0-1 评分
        score = 0.5 + 0.5 * np.tanh(z_score / 2.0)
        
        # 关键：shift(1) 避免前视偏差
        score = score.shift(1)
        
        result_dict[asset] = score
    
    result = pd.DataFrame(result_dict)
    result = result.reindex(all_dates)
    result = result.fillna(0.5)  # 中性评分
    
    return result