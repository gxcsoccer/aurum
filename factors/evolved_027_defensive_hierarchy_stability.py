"""
Factor: defensive_hierarchy_stability
Category: regime
Description: 衡量防御型资产（TLT/GLD/SHY）之间相对强度排名的稳定性——稳定排名表明清晰的风险叙事，频繁切换表明风险类型不确定、防御信号可靠性低。
"""
import pandas as pd
import numpy as np

def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        regime 因子：Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    defensive_assets = ['TLT', 'GLD', 'SHY']
    
    # 获取防御型资产收盘价并对齐日期
    close_dict = {}
    for asset in defensive_assets:
        if asset in prices:
            close = prices[asset]['close'].reindex(all_dates, method='ffill')
            close_dict[asset] = close
    
    if len(close_dict) < 2:
        return pd.Series(0.0, index=all_dates)
    
    # 计算防御资产间的相对动量（21 日）
    momentum_dict = {}
    for asset, close in close_dict.items():
        ret = close.pct_change(21).shift(1)  # 21 日动量，前移避免前视
        momentum_dict[asset] = ret
    
    # 构建动量 DataFrame
    mom_df = pd.DataFrame(momentum_dict)
    
    # 计算每日防御资产排名（1=最强，3=最弱）
    def get_rank_stability(row):
        if row.isnull().any():
            return np.nan
        # 获取当前排名
        ranks = row.rank(ascending=False)  # 高动量=高排名
        return ranks
    
    rank_df = mom_df.apply(get_rank_stability, axis=1)
    
    # 计算排名变化率（排名切换频率）
    rank_change = rank_df.diff().abs().sum(axis=1)  # 总排名变化幅度
    
    # 滚动计算排名稳定性（低变化=高稳定性）
    # 使用 21 日窗口观察排名切换频率
    rank_change_smooth = rank_change.rolling(21, min_periods=10).mean().shift(1)
    
    # 反转：高稳定性=低排名变化=低 regime 分数（应进攻）
    # 低稳定性=高排名变化=高 regime 分数（应保守，风险叙事混乱）
    # 标准化到 0-1 范围
    rolling_max = rank_change_smooth.rolling(252, min_periods=63).max()
    rolling_min = rank_change_smooth.rolling(252, min_periods=63).min()
    
    regime_score = (rank_change_smooth - rolling_min) / (rolling_max - rolling_min + 1e-8)
    regime_score = regime_score.clip(0, 1)
    
    # 填充 NaN
    regime_score = regime_score.fillna(0.5)
    
    return regime_score