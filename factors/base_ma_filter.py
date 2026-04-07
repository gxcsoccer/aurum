"""
Factor: MA Trend Filter (Defensive)
Category: filter
Description: 防御型资产 63 日均线趋势过滤。
  价格高于 MA 输出 1，低于输出 0。
  用于过滤趋势向下的防御资产，避免在下跌趋势中持有 TLT/GLD。
"""
import pandas as pd
import numpy as np

MA_PERIOD = 63


def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame]
        all_dates: DatetimeIndex
        assets: list[str] - 防御型资产列表
    Returns:
        DataFrame(index=all_dates, columns=assets, values=0.0 or 1.0)
    """
    filters = {}
    for name in assets:
        if name not in prices:
            continue
        close = prices[name]["close"].reindex(all_dates)
        ma = close.rolling(MA_PERIOD).mean().shift(1)
        close_shifted = close.shift(1)
        filters[name] = (close_shifted > ma).astype(float)

    return pd.DataFrame(filters).reindex(all_dates)
