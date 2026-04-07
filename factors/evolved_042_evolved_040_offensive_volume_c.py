"""
Factor: evolved_040_offensive_volume_confirmed_momentum
Category: offensive
Description: 衡量进攻型资产价格动量与成交量趋势的确认程度——价格上涨且成交量放大时给予更高评分，价格下跌且成交量放大时给予更低评分。成交量确认的动量比纯价格动量更具持续性。
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
    score_df = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    for asset in assets:
        if asset not in prices:
            score_df[asset] = 0.0
            continue
            
        df = prices[asset].copy()
        
        # 重索引到所有交易日，处理缺失日期
        df = df.reindex(all_dates)
        
        # 前向填充缺失值（但信号要 shift）
        df = df.fillna(method='ffill')
        
        close = df['close']
        volume = df['volume']
        
        # 计算价格动量（20 日收益率）
        price_mom = close.pct_change(20).shift(1)
        
        # 计算成交量趋势（20 日平均成交量相对于 60 日平均的变化）
        vol_20 = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_60 = volume.rolling(60, min_periods=30).mean().shift(1)
        vol_trend = (vol_20 / vol_60 - 1).shift(1)
        
        # 计算量价确认信号
        # 当价格和成交量同向时确认度高，反向时确认度低
        # 标准化两个信号
        price_mom_z = (price_mom - price_mom.rolling(252, min_periods=60).mean()) / (price_mom.rolling(252, min_periods=60).std() + 1e-8)
        vol_trend_z = (vol_trend - vol_trend.rolling(252, min_periods=60).mean()) / (vol_trend.rolling(252, min_periods=60).std() + 1e-8)
        
        # 量价确认分数：价格动量 * 成交量趋势符号
        # 正动量 + 正成交量趋势 = 高确认
        # 正动量 + 负成交量趋势 = 低确认（背离）
        # 负动量 + 正成交量趋势 = 更低（放量下跌）
        # 负动量 + 负成交量趋势 = 中等（缩量下跌）
        volume_confirmation = price_mom_z * np.sign(vol_trend_z + 1e-8)
        
        # 限制极端值
        volume_confirmation = volume_confirmation.clip(-3, 3)
        
        # 填充 NaN
        volume_confirmation = volume_confirmation.fillna(0)
        
        score_df[asset] = volume_confirmation
    
    return score_df