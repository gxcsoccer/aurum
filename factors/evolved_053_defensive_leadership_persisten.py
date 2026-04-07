"""
Factor: defensive_leadership_persistence
Category: regime
Description: 衡量防御型资产相对于进攻型资产的超额收益持续性——持续多日的防御领先表明系统性风险规避，短暂出超可能是噪音。高值时策略应更保守。
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
    # 定义资产类别
    offensive_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    defensive_assets = ['TLT', 'GLD', 'SHY']
    
    # 获取可用的资产（处理可能缺失的资产）
    available_offensive = [a for a in offensive_assets if a in prices]
    available_defensive = [a for a in defensive_assets if a in prices]
    
    if not available_offensive or not available_defensive:
        return pd.Series(index=all_dates, data=0.0)
    
    # 获取收盘价并对齐到所有日期
    def get_close_aligned(asset_name):
        close = prices[asset_name]['close'].reindex(all_dates)
        return close
    
    # 计算进攻型和防御型的等权组合收益
    offensive_returns = []
    for asset in available_offensive:
        close = get_close_aligned(asset)
        ret = close.pct_change()
        offensive_returns.append(ret)
    offensive_avg_ret = pd.DataFrame(offensive_returns).mean(axis=0)
    
    defensive_returns = []
    for asset in available_defensive:
        close = get_close_aligned(asset)
        ret = close.pct_change()
        defensive_returns.append(ret)
    defensive_avg_ret = pd.DataFrame(defensive_returns).mean(axis=0)
    
    # 防御相对进攻的超额收益
    excess_return = defensive_avg_ret - offensive_avg_ret
    
    # 计算超额收益的符号（1=防御领先，-1=进攻领先，0=平手）
    excess_sign = np.sign(excess_return)
    
    # 计算连续防御领先的天数（持续性）
    # 使用滚动窗口计算防御领先的频率
    window = 20  # 20 交易日约 1 个月
    defensive_leadership_freq = excess_sign.rolling(window=window, min_periods=5).mean()
    
    # 防御领导频率越高，regime 值越高（应更保守）
    # 将频率从 [-1, 1] 映射到 [0, 1]
    regime_signal = (defensive_leadership_freq + 1) / 2
    
    # 确保无前视偏差
    regime_signal = regime_signal.shift(1)
    
    # 填充 NaN
    regime_signal = regime_signal.fillna(0)
    
    # 确保索引对齐
    regime_signal = regime_signal.reindex(all_dates).fillna(0)
    
    return regime_signal