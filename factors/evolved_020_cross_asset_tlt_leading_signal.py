"""
Factor: cross_asset_tlt_leading_signal
Category: regime
Description: 使用 TLT 动量变化作为进攻型资产的领先指标——TLT 动量下降预示风险偏好上升（应进攻），TLT 动量上升预示风险规避（应保守）。捕捉跨资产领先关系而非简单相关性。
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
        regime 因子：Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    # 确保所有价格数据对齐到 all_dates
    if 'TLT' not in prices:
        return pd.Series(index=all_dates, data=0.0)
    
    tlt_close = prices['TLT']['close'].reindex(all_dates)
    
    # 计算 TLT 的多时间窗口动量
    # 短期动量（5 日）- 捕捉近期变化
    tlt_mom_short = tlt_close.pct_change(5).shift(1)
    
    # 中期动量（20 日）- 捕捉趋势方向
    tlt_mom_mid = tlt_close.pct_change(20).shift(1)
    
    # 动量加速度 - 动量的变化率
    tlt_mom_accel = tlt_mom_short.diff(5).shift(1)
    
    # 标准化各信号（使用滚动标准化避免前视）
    def rolling_zscore(series, window=60):
        rolling_mean = series.rolling(window=window, min_periods=20).mean()
        rolling_std = series.rolling(window=window, min_periods=20).std()
        return (series - rolling_mean) / (rolling_std + 1e-8)
    
    tlt_mom_short_z = rolling_zscore(tlt_mom_short)
    tlt_mom_mid_z = rolling_zscore(tlt_mom_mid)
    tlt_mom_accel_z = rolling_zscore(tlt_mom_accel)
    
    # 组合信号：TLT 走强（正动量）= 风险规避信号 = 高 regime 值
    # 权重分配：中期动量为主，短期动量和加速度为确认
    regime_signal = (
        0.5 * tlt_mom_mid_z + 
        0.3 * tlt_mom_short_z + 
        0.2 * tlt_mom_accel_z
    )
    
    # 平滑信号以减少噪音
    regime_signal_smooth = regime_signal.rolling(window=5, min_periods=1).mean()
    
    # 确保输出对齐所有日期
    result = regime_signal_smooth.reindex(all_dates)
    result = result.fillna(0.0)
    
    return result