"""
Factor: offensive_momentum_rank_stability
Category: offensive
Description: 衡量进攻型资产动量排名的稳定性——排名越稳定，趋势质量越高。通过计算资产在滚动窗口内动量排名的标准差，惩罚排名频繁切换的资产，捕捉领导权稳定性信号。
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
    # 初始化结果 DataFrame
    scores = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    scores[:] = np.nan
    
    # 只处理进攻型资产
    offensive_assets = [a for a in assets if a in ['SPY', 'QQQ', 'EFA', 'EEM']]
    
    if len(offensive_assets) < 2:
        return scores
    
    # 计算各资产收盘价（对齐到所有日期）
    closes = {}
    for asset in offensive_assets:
        if asset in prices:
            close = prices[asset]['close'].reindex(all_dates)
            closes[asset] = close
    
    # 检查是否有足够数据
    valid_assets = [a for a in offensive_assets if a in closes and closes[a].notna().sum() > 50]
    
    if len(valid_assets) < 2:
        return scores
    
    # 计算 20 日动量（收益率）
    momentum = {}
    for asset in valid_assets:
        mom = closes[asset].pct_change(20).shift(1)  # shift(1) 避免前视
        momentum[asset] = mom
    
    # 计算滚动窗口内动量排名稳定性（窗口 60 天）
    window = 60
    min_periods = 30
    
    for date in all_dates:
        idx = all_dates.get_loc(date)
        if idx < window:
            continue
        
        # 获取窗口内的动量数据
        window_dates = all_dates[idx - window:idx]
        
        # 计算每个资产在窗口内每天的动量排名
        ranks_history = {asset: [] for asset in valid_assets}
        
        for w_date in window_dates:
            # 获取该日各资产的动量值
            mom_values = []
            for asset in valid_assets:
                if w_date in momentum[asset].index:
                    mom_val = momentum[asset].loc[w_date]
                    if pd.notna(mom_val):
                        mom_values.append((asset, mom_val))
            
            if len(mom_values) >= 2:
                # 计算排名（动量越高排名越高）
                mom_values.sort(key=lambda x: x[1], reverse=True)
                for rank, (asset, _) in enumerate(mom_values):
                    ranks_history[asset].append(rank + 1)  # 排名从 1 开始
        
        # 计算每个资产排名序列的标准差（稳定性）
        for asset in valid_assets:
            if len(ranks_history[asset]) >= min_periods:
                rank_std = np.std(ranks_history[asset], ddof=1)
                # 排名标准差越小越稳定，转换为分数（反向）
                # 标准化：假设最大可能标准差约为资产数量/2
                max_std = len(valid_assets) / 2
                stability_score = 1.0 - (rank_std / max_std) if max_std > 0 else 0.5
                stability_score = max(0.0, min(1.0, stability_score))
                scores.loc[date, asset] = stability_score
    
    # 向前填充少量缺失值（不超过 5 天）
    scores = scores.fillna(method='ffill', limit=5)
    
    # 剩余缺失值填 0.5（中性）
    scores = scores.fillna(0.5)
    
    return scores