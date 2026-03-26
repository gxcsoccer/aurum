"""
Aurum Strategy — agent 可以修改此文件的所有内容
当前策略：简单动量策略（基线 v5 - 进出场阈值不对称）
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 10        # 动量回望期（天）
ENTRY_THRESH = 0.0   # 入场阈值（10 日收益率 > 0%）
EXIT_THRESH = -0.01  # 出场阈值（10 日收益率 < -1% 才退出，增加持仓粘性）

# ============ 信号逻辑区 ============
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    输入：OHLCV DataFrame (columns: open, high, low, close, volume)
    输出：signal Series, 值为 1（做多）或 0（空仓）
    """
    # 过去 LOOKBACK 天的收益率（shift(1) 确保无前视偏差）
    momentum = df['close'].pct_change(LOOKBACK).shift(1)

    # 初始化信号
    signal = pd.Series(0, index=df.index)
    
    # 当前持仓状态（用于实现进出场不对称）
    position = 0
    
    for i in range(len(df)):
        if position == 0:
            # 空仓时：动量超过入场阈值则入场
            if momentum.iloc[i] > ENTRY_THRESH:
                position = 1
        else:
            # 持仓时：动量跌破出场阈值才退出（更严格的退出条件）
            if momentum.iloc[i] < EXIT_THRESH:
                position = 0
        
        signal.iloc[i] = position

    return signal