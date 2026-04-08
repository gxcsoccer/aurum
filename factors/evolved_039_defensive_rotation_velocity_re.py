"""
Factor: defensive_rotation_velocity_regime
Category: regime
Description: 衡量防御型资产（TLT/GLD/SHY）之间的资金轮动速度——通过计算防御资产相对强度的变化率和离散度，快速轮动表明宏观不确定性高、风险类型模糊，应提高保守程度。
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
    defensive_assets = ['TLT', 'GLD', 'SHY']
    
    # 获取防御型资产收盘价并对齐日期
    close_dict = {}
    for asset in defensive_assets:
        if asset in prices:
            close = prices[asset]['close'].reindex(all_dates, method='ffill')
            close_dict[asset] = close
    
    if len(close_dict) < 2:
        return pd.Series(0.0, index=all_dates)
    
    # 计算防御资产的动量（21 日）
    momentum_dict = {}
    for asset, close in close_dict.items():
        ret = close.pct_change(21)
        momentum_dict[asset] = ret.shift(1)  # 确保无前视
    
    # 构建动量矩阵
    momentum_df = pd.DataFrame(momentum_dict)
    
    # 计算防御资产间的相对强度差异（横截面离散度）
    # 方法 1: 动量截面的标准差
    momentum_std = momentum_df.std(axis=1)
    
    # 方法 2: 动量变化率（轮动速度）
    momentum_change = momentum_df.diff()
    momentum_change_std = momentum_change.std(axis=1)
    
    # 方法 3: 防御资产相对排名的变化频率
    # 每天对防御资产动量排名，计算排名变化的幅度
    def rank_change(row):
        if row.isna().all():
            return np.nan
        current_rank = row.rank().values
        return current_rank
    
    rank_df = momentum_df.apply(rank_change, axis=1)
    rank_change_magnitude = rank_df.diff().abs().sum(axis=1)
    
    # 组合三个信号：离散度 + 变化率 + 排名变化
    # 标准化后加权平均
    def normalize(series):
        return (series - series.rolling(252, min_periods=63).mean()) / (series.rolling(252, min_periods=63).std() + 1e-8)
    
    signal1 = normalize(momentum_std)
    signal2 = normalize(momentum_change_std.shift(1))
    signal3 = normalize(rank_change_magnitude.shift(1))
    
    # 等权重组合
    regime_signal = (signal1 + signal2 + signal3) / 3.0
    
    # 填充缺失值
    regime_signal = regime_signal.fillna(0.0)
    
    # 确保输出与 all_dates 对齐
    regime_signal = regime_signal.reindex(all_dates).fillna(0.0)
    
    return regime_signal