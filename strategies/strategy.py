"""
Aurum Strategy — agent 可以修改此文件的所有内容
当前策略：简单动量策略（基线 v4 - 短期动量）
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 10        # 动量回望期（天）
ENTRY_THRESH = 0.0   # 入场阈值（10 日收益率 > 0%）

# ============ 信号逻辑区 ============
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    输入：OHLCV DataFrame (columns: open, high, low, close, volume)
    输出：signal Series, 值为 1（做多）或 0（空仓）
    """
    # 过去 LOOKBACK 天的收益率（shift(1) 确保无前视偏差）
    momentum = df['close'].pct_change(LOOKBACK).shift(1)

    # 动量为正且超过阈值 - 做多
    signal = (momentum > ENTRY_THRESH).astype(int)

    return signal