"""
Factor: defensive_stress_performance
Category: defensive
Description: 衡量防御型资产在进攻型资产压力时期（下跌日）的相对表现——真正的对冲资产应在股市下跌时表现强势，该信号捕捉防御资产的对冲有效性差异。
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
        DataFrame(index=all_dates, columns=assets, values=float scores)
    """
    defensive_assets = ['TLT', 'GLD', 'SHY']
    offensive_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    
    # 初始化结果 DataFrame
    scores = pd.DataFrame(index=all_dates, columns=assets, values=0.0)
    
    # 获取进攻型资产的收盘价并重新索引到所有日期
    offensive_returns = {}
    for asset in offensive_assets:
        if asset in prices:
            close = prices[asset]['close'].reindex(all_dates)
            ret = close.pct_change()
            offensive_returns[asset] = ret
    
    # 计算进攻型资产的综合压力信号（平均收益率）
    offensive_avg_ret = pd.DataFrame(offensive_returns).mean(axis=1)
    
    # 识别压力日：进攻型资产平均收益为负
    stress_days = (offensive_avg_ret < 0).shift(1)  # 使用前一日信号避免前视
    
    # 为每个防御型资产计算压力期表现评分
    for asset in defensive_assets:
        if asset not in assets:
            continue
        if asset not in prices:
            continue
            
        close = prices[asset]['close'].reindex(all_dates)
        ret = close.pct_change()
        
        # 计算压力期的平均收益（仅在压力日）
        stress_ret = ret.where(stress_days, np.nan)
        
        # 使用滚动窗口计算压力期表现（21 天窗口）
        stress_performance = stress_ret.rolling(window=21, min_periods=10).mean()
        
        # 计算非压力期的平均收益作为对比
        non_stress_ret = ret.where(~stress_days, np.nan)
        non_stress_performance = non_stress_ret.rolling(window=21, min_periods=10).mean()
        
        # 评分 = 压力期表现 - 非压力期表现（真正的对冲资产在压力期表现更好）
        # 这样能区分"一直涨"和"跌时涨"的资产
        score = stress_performance - non_stress_performance
        
        # 标准化到 0-1 范围
        score = (score - score.rolling(window=252, min_periods=60).min()) / \
                (score.rolling(window=252, min_periods=60).max() - score.rolling(window=252, min_periods=60).min() + 1e-8)
        
        # 填充到结果
        scores.loc[:, asset] = score.fillna(0.5)
    
    # 非防御型资产评分为 0
    for asset in assets:
        if asset not in defensive_assets:
            scores.loc[:, asset] = 0.0
    
    return scores