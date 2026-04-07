"""
Factor: Pure Multi-Period Momentum (Defensive)
Category: defensive
Description: 多期加权纯动量，用于防御型资产评分。
  不除以波动率，因为防御型资产波动率本身较低，除以后会放大噪音。
"""
import pandas as pd
import numpy as np

MOM_PERIODS = [21, 63, 126, 252]
MOM_WEIGHTS = [1, 2, 4, 6]


def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame]
        all_dates: DatetimeIndex
        assets: list[str]
    Returns:
        DataFrame(index=all_dates, columns=assets, values=scores)
    """
    scores = {}
    for name in assets:
        if name not in prices:
            continue
        close = prices[name]["close"].reindex(all_dates)

        mom_values = []
        for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
            mom = close.pct_change(period).shift(1)
            mom_values.append(mom * weight)
        scores[name] = sum(mom_values) / sum(MOM_WEIGHTS)

    return pd.DataFrame(scores).reindex(all_dates)
