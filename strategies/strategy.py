"""
Aurum Strategy — agent 可以修改此文件的所有内容
当前策略：动量策略 + 长期趋势过滤 + 趋势斜率确认
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 20        # 动量回望期（天）
ENTRY_THRESH = 0.02  # 入场阈值（20 日收益率 > 2%）
MA_SLOPE_PERIOD = 20 # 均线斜率确认周期（20 日）

# ============ 信号逻辑区 ============
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    输入：OHLCV DataFrame (columns: open, high, low, close, volume)
    输出：signal Series, 值为 1（做多）或 0（空仓）
    """
    # 过去 LOOKBACK 天的收益率（shift(1) 确保无前视偏差）
    momentum = df['close'].pct_change(LOOKBACK).shift(1)

    # 计算 200 日均线作为长期趋势过滤（shift(1) 确保无前视偏差）
    trend_ma = df['close'].rolling(200).mean().shift(1)
    
    # 新增：计算均线斜率，确保长期趋势不仅向上且正在增强
    # 比较昨日均线与 20 日前的昨日均线，避免使用当日数据
    trend_slope = trend_ma > trend_ma.shift(MA_SLOPE_PERIOD)
    
    # 趋势过滤：当前收盘价高于 200 日均线 且 均线处于上升趋势
    trend_filter = (df['close'] > trend_ma) & trend_slope

    # 动量为正且超过阈值 且 处于长期上涨趋势 - 做多
    # 使用 fillna(0) 处理初始数据不足产生的 NaN
    signal = ((momentum > ENTRY_THRESH) & trend_filter).fillna(0).astype(int)

    return signal