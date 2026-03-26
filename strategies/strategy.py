"""
Aurum Strategy — agent 可以修改此文件的所有内容
当前策略：动量策略 + 长期趋势过滤 + 趋势斜率确认 + 波动率自适应阈值 + 波动率上限过滤 + 最小绝对动量阈值
"""
import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 20           # 动量回望期（天）
ATR_PERIOD = 20         # ATR 计算周期
ATR_MULTIPLIER = 1.5    # ATR 阈值乘数（动态入场阈值 = ATR * multiplier / close）
MA_SLOPE_PERIOD = 20    # 均线斜率确认周期（20 日）
VOLATILITY_CAP = 1.5    # 波动率比率上限（当前 ATR / 60 日平均 ATR）
MIN_MOMENTUM = 0.01     # 最小绝对动量阈值（1%），确保信号有足够的价格运动强度

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

    # 计算 ATR 作为波动率指标（使用 shift(1) 确保无前视偏差）
    high = df['high'].shift(1)
    low = df['low'].shift(1)
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean().shift(1)
    
    # 波动率自适应入场阈值：ATR * multiplier / 昨日收盘价
    # 这样阈值会随市场波动率动态调整
    dynamic_threshold = (atr * ATR_MULTIPLIER) / close_prev
    
    # 新增：波动率上限过滤
    # 计算当前 ATR 相对于 60 日平均 ATR 的比率
    atr_mean_60 = atr.rolling(60).mean()
    volatility_ratio = atr / atr_mean_60
    # 只在波动率不过度极端时交易（避免高波动反转风险）
    volatility_filter = (volatility_ratio < VOLATILITY_CAP) | volatility_ratio.isna()

    # 新增：最小绝对动量过滤
    # 确保动量信号有足够的绝对强度，避免低波动环境下阈值过低产生低质量信号
    min_momentum_filter = momentum > MIN_MOMENTUM

    # 动量为正且超过动态阈值 且 超过最小绝对阈值 且 处于长期上涨趋势 且 波动率不过度极端 - 做多
    # 使用 fillna(0) 处理初始数据不足产生的 NaN
    signal = ((momentum > dynamic_threshold) & trend_filter & volatility_filter & min_momentum_filter).fillna(0).astype(int)

    return signal