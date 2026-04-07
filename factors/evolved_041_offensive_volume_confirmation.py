"""
Factor: offensive_volume_confirmation
Category: offensive
Description: 衡量进攻型资产价格上涨是否得到成交量确认。放量上涨表明机构参与度高、趋势可持续；缩量上涨表明动能不足。与纯价格动量因子正交。
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
            scores[asset] = 0.0
            continue
        
        df = prices[asset].reindex(all_dates)
        close = df['close']
        volume = df['volume']
        
        # 计算价格动量（20 日收益）
        mom_20 = close.pct_change(20).shift(1)
        
        # 计算成交量相对水平（20 日平均）
        vol_ma = volume.rolling(20).mean().shift(1)
        vol_ratio = volume.shift(1) / vol_ma
        
        # 量价确认信号：正动量且放量 = 强确认；正动量但缩量 = 弱确认
        # 负动量时，放量下跌 = 强负面；缩量下跌 = 弱负面
        price_change = close.pct_change(1).shift(1)
        
        # 量价配合度：价格变化方向与成交量异常的乘积
        # 上涨 + 放量 = 正分；上涨 + 缩量 = 负分；下跌 + 放量 = 负分；下跌 + 缩量 = 较小负分
        vol_signal = (vol_ratio - 1.0)  # 高于均量为正，低于为负
        price_direction = np.sign(price_change)
        
        # 量价确认分数
        volume_confirmation = vol_signal * price_direction
        
        # 结合动量方向加权：只在有明确动量方向时给予量价信号权重
        momentum_weight = np.abs(mom_20)
        score = volume_confirmation * momentum_weight
        
        # 标准化到合理范围
        score = score.rolling(60).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8) if len(x) > 10 else 0.0,
            raw=False
        ).shift(1)
        
        scores[asset] = score.fillna(0.0)
    
    return scores