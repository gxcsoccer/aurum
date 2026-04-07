"""
Factor: Vol-Adjusted Multi-Period Momentum (Offensive)
Category: offensive
Description: 多期加权动量除以波动率，用于进攻型资产评分。
  动量周期 1/3/6/12 个月，权重 1/2/4/6，波动率 10 个月年化。
  捕捉趋势强度的同时惩罚高波动资产。
"""
import pandas as pd
import numpy as np

MOM_PERIODS = [21, 63, 126, 252]
MOM_WEIGHTS = [1, 2, 4, 6]
VOL_LOOKBACK = 210


def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - 资产价格数据
        all_dates: DatetimeIndex - 所有交易日
        assets: list[str] - 要计算的资产列表
    Returns:
        DataFrame(index=all_dates, columns=assets, values=scores)
    """
    scores = {}
    for name in assets:
        if name not in prices:
            continue
        close = prices[name]["close"].reindex(all_dates)

        # 多期加权动量
        mom_values = []
        for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
            mom = close.pct_change(period).shift(1)
            mom_values.append(mom * weight)
        momentum = sum(mom_values) / sum(MOM_WEIGHTS)

        # 年化波动率
        returns = close.pct_change().shift(1)
        vol = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)

        # 动量 / 波动率
        scores[name] = momentum / (vol + 0.0001)

    return pd.DataFrame(scores).reindex(all_dates)
