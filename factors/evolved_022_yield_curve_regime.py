"""
Factor: yield_curve_regime
Category: regime
Description: 使用国债曲线动态（SHY vs TLT 相对表现）作为宏观制度信号。曲线趋平（SHY 跑赢 TLT）通常预示经济放缓/流动性收紧，曲线趋陡预示增长乐观。曲线动量变化比绝对曲线水平更能提前捕捉 regime 转换。
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
        Series(index=all_dates, values=float) - 高值=风险高/应保守
    """
    # Check if SHY and TLT are available
    if 'SHY' not in prices or 'TLT' not in prices:
        return pd.Series(0.0, index=all_dates)
    
    # Reindex to ensure consistent dates
    shy_close = prices['SHY']['close'].reindex(all_dates)
    tlt_close = prices['TLT']['close'].reindex(all_dates)
    
    # Calculate returns over medium-term window (1 month ~ 21 days)
    shy_ret = shy_close.pct_change(21)
    tlt_ret = tlt_close.pct_change(21)
    
    # Curve spread: SHY outperformance vs TLT (positive = flattening)
    # SHY up / TLT down = flight to safety / liquidity preference = risk signal
    curve_spread = shy_ret - tlt_ret
    
    # Momentum of curve spread (acceleration of flattening/steepening)
    # Accelerating flattening = increasing risk concern
    curve_momentum = curve_spread - curve_spread.shift(21)
    
    # Normalize by rolling std for stability
    curve_std = curve_momentum.rolling(63).std()
    curve_signal = curve_momentum / curve_std.replace(0, np.nan)
    
    # Shift to avoid look-ahead bias
    signal = curve_signal.shift(1)
    
    # Fill NaN with 0 (neutral regime)
    signal = signal.fillna(0)
    
    return signal