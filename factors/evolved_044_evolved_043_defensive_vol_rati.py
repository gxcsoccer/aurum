"""
Factor: evolved_043_defensive_vol_ratio_regime
Category: regime
Description: 衡量防御型资产内部波动率比率（TLT vol / GLD vol）作为宏观风险类型信号。比率上升表明通缩/增长担忧主导，比率下降表明通胀/货币担忧主导。
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
    # 获取防御型资产价格
    tlt_prices = prices.get('TLT', None)
    gld_prices = prices.get('GLD', None)
    
    # 如果缺少任一防御资产，返回中性信号
    if tlt_prices is None or gld_prices is None:
        return pd.Series(0.0, index=all_dates)
    
    # 对齐日期
    tlt_close = tlt_prices['close'].reindex(all_dates)
    gld_close = gld_prices['close'].reindex(all_dates)
    
    # 计算对数收益率
    tlt_ret = np.log(tlt_close / tlt_close.shift(1))
    gld_ret = np.log(gld_close / gld_close.shift(1))
    
    # 计算滚动波动率（21 日，约 1 个月）
    tlt_vol = tlt_ret.rolling(window=21, min_periods=10).std() * np.sqrt(252)
    gld_vol = gld_ret.rolling(window=21, min_periods=10).std() * np.sqrt(252)
    
    # 计算波动率比率（TLT / GLD）
    vol_ratio = tlt_vol / gld_vol.replace(0, np.nan)
    
    # 处理极端值和缺失值
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)
    vol_ratio = vol_ratio.fillna(vol_ratio.expanding().mean())
    
    # 标准化为 z-score（使用滚动窗口避免前视）
    vol_ratio_mean = vol_ratio.rolling(window=63, min_periods=21).mean()
    vol_ratio_std = vol_ratio.rolling(window=63, min_periods=21).std()
    vol_ratio_zscore = (vol_ratio - vol_ratio_mean) / vol_ratio_std.replace(0, np.nan)
    
    # 前视偏差修复：所有指标必须 shift(1)
    signal = vol_ratio_zscore.shift(1)
    
    # 填充早期缺失值
    signal = signal.fillna(0.0)
    
    # 限制极端值（避免单个异常值主导）
    signal = signal.clip(-3, 3)
    
    return signal