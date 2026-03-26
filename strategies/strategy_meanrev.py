"""
Aurum Strategy B — 均值回归策略（基线）
核心思路：短期超跌后反弹，与动量策略形成互补
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
RSI_PERIOD = 14       # RSI 计算周期
RSI_OVERSOLD = 35     # 超卖阈值（从 30 提高到 35，增加入场信号）
RSI_EXIT = 45         # 退出阈值（从 50 降低到 45，允许部分回归即退出）
BB_PERIOD = 20        # 布林带周期
BB_STD = 2.0          # 布林带标准差倍数
MAX_HOLD_DAYS = 15    # 最大持仓天数
MIN_HOLD_DAYS = 3     # 最小持仓天数（从 2 延长至 3，让回归更充分）
PROFIT_TARGET = 0.035  # 止盈目标（从 3% 提高到 3.5%，让盈利更充分）
DROP_PERIOD = 3       # 累计跌幅计算周期
DROP_THRESHOLD = 0.05 # 累计跌幅阈值（5%）

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
    
    # 计算 3 日累计跌幅（shift(1) 确保无前视偏差）
    price_change = df['close'].pct_change()
    cumulative_drop = price_change.rolling(DROP_PERIOD).sum().shift(1)
    drop_condition = cumulative_drop < -DROP_THRESHOLD
    
    # 当日收盘价低于前一日收盘价（确认下跌趋势）
    price_decline = df['close'] < df['close'].shift(1)
    
    # 入场条件：RSI 超卖 OR 价格跌破布林带下轨 OR 3 日累计跌幅>5%（均值回归信号）
    # 必须同时满足当日价格下跌确认
    entry_condition = ((rsi < RSI_OVERSOLD) | (df['close'] < bb_lower) | drop_condition) & price_decline
    
    # 退出条件：RSI 回归到均值以上（从 50 降低到 45）
    exit_condition = rsi > RSI_EXIT
    
    # 状态机逻辑：持仓状态保持
    in_position = False
    hold_days = 0
    entry_price = 0.0
    
    for i in range(n):
        if not in_position:
            # 空仓时可以入场
            if entry_condition.iloc[i]:
                in_position = True
                hold_days = 0
                entry_price = df['close'].iloc[i]
                signal.iloc[i] = 1
        else:
            # 持仓中
            hold_days += 1
            current_price = df['close'].iloc[i]
            
            # 计算当前盈利比例
            profit_pct = (current_price - entry_price) / entry_price
            
            # 退出条件：RSI 回归 或 达到最大持仓天数 或 达到止盈目标
            # 必须满足最小持仓天数才能退出（避免过早退出）
            can_exit = hold_days >= MIN_HOLD_DAYS
            if can_exit and (exit_condition.iloc[i] or (hold_days >= MAX_HOLD_DAYS) or (profit_pct >= PROFIT_TARGET)):
                in_position = False
                hold_days = 0
                entry_price = 0.0
                signal.iloc[i] = 0
            else:
                signal.iloc[i] = 1
    
    return signal