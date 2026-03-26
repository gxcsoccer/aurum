"""
Aurum Strategy B — 均值回归策略（基线）
核心思路：短期超跌后反弹，与动量策略形成互补
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
RSI_PERIOD = 14       # RSI 计算周期
RSI_OVERSOLD = 30     # 超卖阈值
RSI_EXIT = 50         # 退出阈值（回归到均值）
BB_PERIOD = 20        # 布林带周期
BB_STD = 2.0          # 布林带标准差倍数
MAX_HOLD_DAYS = 10    # 最大持仓天数

# ============ 信号逻辑区 ============
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    输入：OHLCV DataFrame (columns: open, high, low, close, volume)
    输出：signal Series, 值为 1（做多）或 0（空仓）
    """
    n = len(df)
    signal = pd.Series(0, index=df.index)
    
    # 计算 RSI（shift(1) 确保无前视偏差）
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).shift(1)
    
    # 计算布林带（shift(1) 确保无前视偏差）
    bb_mid = df['close'].rolling(BB_PERIOD).mean().shift(1)
    bb_std = df['close'].rolling(BB_PERIOD).std().shift(1)
    bb_lower = bb_mid - BB_STD * bb_std
    
    # 入场条件：RSI 超卖 OR 价格跌破布林带下轨（均值回归信号）
    entry_condition = (rsi < RSI_OVERSOLD) | (df['close'] < bb_lower)
    
    # 退出条件：RSI 回归到均值以上
    exit_condition = rsi > RSI_EXIT
    
    # 状态机逻辑：持仓状态保持
    in_position = False
    hold_days = 0
    
    for i in range(n):
        if not in_position:
            # 空仓时可以入场
            if entry_condition.iloc[i]:
                in_position = True
                hold_days = 0
                signal.iloc[i] = 1
        else:
            # 持仓中
            hold_days += 1
            # 退出条件：RSI 回归 或 达到最大持仓天数
            if exit_condition.iloc[i] or (hold_days >= MAX_HOLD_DAYS):
                in_position = False
                hold_days = 0
                signal.iloc[i] = 0
            else:
                signal.iloc[i] = 1
    
    return signal