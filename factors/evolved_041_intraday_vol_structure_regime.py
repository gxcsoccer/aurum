"""
Factor: intraday_vol_structure_regime
Category: regime
Description: 衡量日内波动相对于方向性变化的比例——高值表示市场犹豫/多空分歧（应防御），低值表示趋势清晰/共识强（可进攻）。
"""
import pandas as pd
import numpy as np

def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame (columns: open/high/low/close/volume)
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        regime 因子：Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    # 收集所有资产的信号后取平均（跨资产共识）
    regime_signals = []
    
    for asset in assets:
        if asset not in prices:
            continue
        
        df = prices[asset].reindex(all_dates)
        
        # 确保有必要的列
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            continue
        
        # 日内范围 (intraday range)
        intraday_range = df['high'] - df['low']
        
        # 收盘价绝对变化 (directional move)
        close_change = df['close'].diff().abs()
        
        # 避免除以零：当收盘变化极小时，用一个小值替代
        close_change_safe = close_change.replace(0, np.nan).fillna(1e-8)
        
        # 日内波动结构比率 = 日内范围 / 收盘变化
        # 高比率 = 日内波动大但方向性弱 = 市场犹豫/分歧
        # 低比率 = 日内波动小但方向性强 = 趋势清晰/共识
        vol_structure_ratio = intraday_range / close_change_safe
        
        # 标准化：使用滚动中位数避免极端值影响
        # 使用 20 日滚动窗口捕捉中期结构变化
        rolling_median = vol_structure_ratio.rolling(window=20, min_periods=5).median()
        
        # 前移一天避免前视偏差
        signal = rolling_median.shift(1)
        
        regime_signals.append(signal)
    
    if len(regime_signals) == 0:
        return pd.Series(0.0, index=all_dates)
    
    # 跨资产平均（捕捉整体市场结构）
    regime_df = pd.concat(regime_signals, axis=1)
    regime_signal = regime_df.mean(axis=1)
    
    # 填充可能的 NaN
    regime_signal = regime_signal.fillna(method='ffill').fillna(0.0)
    
    # 确保索引对齐
    regime_signal = regime_signal.reindex(all_dates).fillna(0.0)
    
    return regime_signal