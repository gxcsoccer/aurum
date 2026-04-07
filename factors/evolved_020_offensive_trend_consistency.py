"""
Factor: offensive_trend_consistency
Category: offensive
Description: 衡量进攻型资产价格路径的持续性——上涨日的比例、连续上涨日数、以及上涨日幅度的稳定性。稳步上涨（高一致性）的资产比大起大落的资产更可能延续趋势，补充现有动量因子对价格路径质量的盲区。
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
        
        # 计算日收益率（使用前一日收盘价）
        returns = close.pct_change()
        
        # 信号 1: 上涨日比例（20 日窗口）
        up_days = (returns > 0).astype(float)
        up_ratio = up_days.rolling(window=20, min_periods=10).mean()
        
        # 信号 2: 连续上涨日数（指数衰减加权）
        consecutive_up = up_days.rolling(window=10, min_periods=5).sum()
        
        # 信号 3: 上涨日幅度稳定性（上涨日收益率的标准差，越低越稳定）
        up_returns = returns.where(returns > 0, np.nan)
        up_std = up_returns.rolling(window=20, min_periods=10).std()
        # 稳定性 = 1 / (1 + std)，标准差越低分数越高
        up_stability = 1.0 / (1.0 + up_std.fillna(1.0))
        
        # 信号 4: 下跌日平均幅度 vs 上涨日平均幅度（盈亏比）
        down_returns = returns.where(returns < 0, np.nan)
        up_mean = up_returns.rolling(window=20, min_periods=10).mean()
        down_mean = down_returns.rolling(window=20, min_periods=10).mean()
        # 盈亏比 = 平均上涨 / |平均下跌 |，避免除零
        profit_loss_ratio = up_mean / (down_mean.abs() + 1e-6)
        profit_loss_ratio = profit_loss_ratio.clip(0, 5)  # 限制极端值
        
        # 组合信号：等权重标准化后加总
        # 所有信号都需要 shift(1) 避免前视
        s1 = up_ratio.shift(1)
        s2 = (consecutive_up / 10.0).shift(1)  # 归一化到 0-1
        s3 = up_stability.shift(1)
        s4 = (profit_loss_ratio / 5.0).shift(1)  # 归一化到 0-1
        
        # 等权重组合
        combined = (s1 + s2 + s3 + s4) / 4.0
        
        # 横截面标准化（z-score）
        combined = combined.reindex(all_dates)
        scores[asset] = combined
    
    # 横截面标准化：每个交易日对所有资产进行 z-score
    for date in all_dates:
        row = scores.loc[date]
        mean_val = row.mean()
        std_val = row.std()
        if std_val > 1e-6:
            scores.loc[date] = (row - mean_val) / std_val
        else:
            scores.loc[date] = 0.0
    
    return scores