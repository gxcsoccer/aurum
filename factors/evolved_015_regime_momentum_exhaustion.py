"""
Factor: regime_momentum_exhaustion
Category: regime
Description: Aggregates momentum acceleration across offensive assets to detect systemic trend exhaustion. High value indicates broad deceleration (high risk).
"""
import pandas as pd
import numpy as np

def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日
        assets: list[str] - 资产列表 (ignored for regime, uses internal pool)
    Returns:
        Series(index=all_dates, values=float) - High value = High Risk
    """
    # Define offensive assets for regime calculation
    offensive_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    
    # Store acceleration signals
    accel_series = []
    
    for asset in offensive_assets:
        if asset not in prices:
            continue
            
        # Get close prices and align to all_dates
        close = prices[asset]['close'].reindex(all_dates)
        
        # Calculate returns
        # 3-month (approx 63 days) and 6-month (approx 126 days) momentum
        ret_3m = close.pct_change(63)
        ret_6m = close.pct_change(126)
        
        # Momentum Acceleration = Short Mom - Long Mom
        # Positive = Accelerating, Negative = Decelerating
        accel = ret_3m - ret_6m
        
        accel_series.append(accel)
    
    if not accel_series:
        return pd.Series(0.0, index=all_dates)
    
    # Combine into DataFrame and take mean across assets
    accel_df = pd.concat(accel_series, axis=1)
    mean_accel = accel_df.mean(axis=1)
    
    # Normalize using rolling Z-score (252 days) to make signal stationary
    rolling_mean = mean_accel.rolling(252, min_periods=63).mean()
    rolling_std = mean_accel.rolling(252, min_periods=63).std()
    
    z_score = (mean_accel - rolling_mean) / (rolling_std + 1e-9)
    
    # Invert signal: Low/Negative Acceleration = High Risk = High Regime Score
    # We want high regime score to trigger conservative behavior
    regime_score = -z_score
    
    # Ensure no look-ahead bias
    regime_score = regime_score.shift(1)
    
    # Reindex and fill NaNs
    regime_score = regime_score.reindex(all_dates)
    regime_score = regime_score.ffill().fillna(0.0)
    
    return regime_score