"""
Factor: offensive_vol_term_structure
Category: offensive
Description: 使用短期/长期波动率比率衡量趋势质量。低比率表示平滑趋势（短期波动<长期波动），高比率表示嘈杂市场。对进攻型资产，平滑趋势中的动量更可靠。
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
        DataFrame(index=all_dates, columns=assets, values=float scores)
    """
    scores = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    for asset in assets:
        if asset not in prices:
            scores[asset] = np.nan
            continue
        
        df = prices[asset].reindex(all_dates)
        close = df['close']
        
        # 计算短期波动率 (5 日) 和长期波动率 (60 日)
        vol_short = close.pct_change().rolling(window=5).std()
        vol_long = close.pct_change().rolling(window=60).std()
        
        # 波动率期限结构比率：短期/长期
        # 比率<1 表示短期更平静（趋势平滑），比率>1 表示短期更动荡（趋势嘈杂）
        vol_ratio = vol_short / vol_long
        
        # 反转信号：低比率=高质量趋势=高分
        # 使用分位数标准化，使分数在 0-1 之间
        vol_ratio_shifted = vol_ratio.shift(1)  # 关键：避免前视偏差
        
        # 滚动分位数标准化（252 日窗口）
        def quantile_norm(series):
            rolling_min = series.rolling(window=252, min_periods=60).min()
            rolling_max = series.rolling(window=252, min_periods=60).max()
            rolling_range = rolling_max - rolling_min
            # 避免除零
            normalized = (series - rolling_min) / (rolling_range.replace(0, np.nan))
            # 反转：低比率=高分
            return 1.0 - normalized
        
        score = vol_ratio_shifted.groupby(pd.Grouper(freq='Y')).transform(quantile_norm)
        
        # 处理边界情况
        score = score.clip(0, 1)
        scores[asset] = score
    
    return scores.fillna(0.0)