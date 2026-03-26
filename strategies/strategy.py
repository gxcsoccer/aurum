"""
Aurum Strategy — agent 可以修改此文件的所有内容
当前策略：动量策略 + 长期趋势过滤
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 20        # 动量回望期（天）
ENTRY_THRESH = 0.02  # 入场阈值（20 日收益率 > 2%）

# ============ 信号逻辑区 ============
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    输入：OHLCV DataFrame (columns: open, high, low, close, volume)
    输出：signal Series, 值为 1（做多）或 0（空仓）
    """
    # 过去 LOOKBACK 天的收益率（shift(1) 确保无前视偏差）
    momentum = df['close'].pct_change(LOOKBACK).shift(1)

    # 计算 200 日均线作为长期趋势过滤（shift(1) 确保无前视偏差）
    # 使用昨日及之前的数据计算均线，避免使用当日收盘价计算当日均线带来的潜在偏差
    trend_ma = df['close'].rolling(200).mean().shift(1)
    
    # 趋势过滤：当前收盘价高于 200 日均线（视为长期上涨趋势）
    # 注意：df['close'] 是当日收盘价，信号在当日收盘后生成，可以使用当日收盘价判断趋势状态
    trend_filter = df['close'] > trend_ma

    # 动量为正且超过阈值 且 处于长期上涨趋势 - 做多
    # 使用 fillna(0) 处理初始数据不足产生的 NaN
    signal = ((momentum > ENTRY_THRESH) & trend_filter).fillna(0).astype(int)

    return signal