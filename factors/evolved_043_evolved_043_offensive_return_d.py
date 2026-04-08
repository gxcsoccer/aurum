"""
Factor: evolved_043_offensive_return_dispersion_regime
Category: regime
Description: 衡量进攻型资产收益的离散程度（标准差）。高离散度表明市场领导力狭窄、趋势基础脆弱，应转向防御；低离散度表明市场健康、趋势广泛，适合进攻。
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
        regime factor: Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    offensive_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    
    # 获取收盘价并对齐到所有日期
    close_dict = {}
    for asset in offensive_assets:
        if asset in prices:
            close = prices[asset]['close'].reindex(all_dates)
            close_dict[asset] = close
    
    # 构建收盘价 DataFrame
    close_df = pd.DataFrame(close_dict)
    
    # 计算日收益率
    returns = close_df.pct_change()
    
    # 计算滚动收益离散度（横截面标准差）
    # 使用 21 天窗口衡量近期收益分化程度
    window = 21
    dispersion = returns.std(axis=1).rolling(window=window).mean()
    
    # 标准化到 0-1 范围，便于解释
    # 使用滚动分位数避免前视偏差
    dispersion_rank = dispersion.rolling(window=252).rank(pct=True)
    
    # 向前移 1 天确保无前视偏差
    signal = dispersion_rank.shift(1)
    
    # 填充前 252 天的缺失值
    signal = signal.fillna(method='bfill').fillna(0.5)
    
    return signal