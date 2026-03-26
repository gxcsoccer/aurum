"""
Aurum Strategy — agent 可以修改此文件的所有内容
当前策略：双动量确认退出策略（基线 v8 - 入场阈值微调至 0.1%）
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK_LONG = 10       # 长期动量回望期（天）
LOOKBACK_SHORT = 5       # 短期动量回望期（天）
ENTRY_THRESH = 0.001     # 入场阈值（10 日收益率 > 0.1%，原为 0%）
EXIT_THRESH_LONG = -0.01 # 长期动量退出阈值（10 日收益率 < -1%）
EXIT_THRESH_SHORT = -0.005  # 短期动量退出阈值（5 日收益率 < -0.5%）

# ============ 信号逻辑区 ============
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    输入：OHLCV DataFrame (columns: open, high, low, close, volume)
    输出：signal Series, 值为 1（做多）或 0（空仓）
    """
    # 长期动量（10 日收益率，shift(1) 确保无前视偏差）
    momentum_long = df['close'].pct_change(LOOKBACK_LONG).shift(1)
    
    # 短期动量（5 日收益率，shift(1) 确保无前视偏差）
    momentum_short = df['close'].pct_change(LOOKBACK_SHORT).shift(1)

    # 初始化信号
    signal = pd.Series(0, index=df.index)
    
    # 当前持仓状态（用于实现进出场不对称）
    position = 0
    
    for i in range(len(df)):
        if position == 0:
            # 空仓时：长期动量超过入场阈值则入场
            if momentum_long.iloc[i] > ENTRY_THRESH:
                position = 1
        else:
            # 持仓时：需长期动量跌破阈值 且 短期动量为负 才退出（双重确认）
            if momentum_long.iloc[i] < EXIT_THRESH_LONG and momentum_short.iloc[i] < EXIT_THRESH_SHORT:
                position = 0
        
        signal.iloc[i] = position

    return signal