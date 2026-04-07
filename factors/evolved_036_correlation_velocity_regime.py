"""
Factor: correlation_velocity_regime
Category: regime
Description: 衡量进攻型与防御型资产间相关性的变化速度（相关性动量）。相关性快速上升表明多元化失效风险累积，往往领先于波动率爆发，提供早期风险预警。
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
        regime 因子：Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    # 定义进攻型和防御型资产
    offensive_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    defensive_assets = ['TLT', 'GLD', 'SHY']
    
    # 获取收盘价并对齐到所有日期
    close_dict = {}
    for asset in offensive_assets + defensive_assets:
        if asset in prices:
            close = prices[asset]['close'].reindex(all_dates)
            close_dict[asset] = close
    
    # 计算收益率（使用 shift(1) 确保无前视）
    ret_dict = {}
    for asset, close in close_dict.items():
        ret = close.pct_change().shift(1)
        ret_dict[asset] = ret
    
    # 计算进攻型与防御型资产间的相关性
    # 使用 60 日滚动窗口计算每对资产的相关性
    corr_window = 60
    
    correlation_series_list = []
    
    for off_asset in offensive_assets:
        for def_asset in defensive_assets:
            if off_asset in ret_dict and def_asset in ret_dict:
                off_ret = ret_dict[off_asset]
                def_ret = ret_dict[def_asset]
                
                # 计算滚动相关性
                rolling_corr = off_ret.rolling(window=corr_window, min_periods=20).corr(def_ret)
                
                # 计算相关性变化速度（相关性的动量）
                # 使用 20 日窗口衡量相关性变化的加速度
                corr_velocity = rolling_corr.diff(20).shift(1)
                
                correlation_series_list.append(corr_velocity)
    
    if len(correlation_series_list) == 0:
        return pd.Series(0.0, index=all_dates)
    
    # 将所有相关性速度信号取平均，得到综合的相关性变化速度指标
    corr_velocity_df = pd.concat(correlation_series_list, axis=1)
    avg_corr_velocity = corr_velocity_df.mean(axis=1)
    
    # 标准化信号（使用滚动标准化，确保可比性）
    rolling_mean = avg_corr_velocity.rolling(window=120, min_periods=60).mean()
    rolling_std = avg_corr_velocity.rolling(window=120, min_periods=60).std()
    
    # 计算 z-score，正值表示相关性正在快速上升（风险信号）
    regime_signal = (avg_corr_velocity - rolling_mean) / (rolling_std + 1e-8)
    
    # 确保信号在合理范围内
    regime_signal = regime_signal.clip(-3, 3)
    
    # 填充缺失值
    regime_signal = regime_signal.fillna(0)
    
    # 确保索引对齐
    regime_signal = regime_signal.reindex(all_dates).fillna(0)
    
    return regime_signal