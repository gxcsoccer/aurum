"""
Factor: Market Volatility Regime
Category: regime
Description: 用进攻型资产平均波动率判断市场环境。
  波动率高于 1 年滚动中位数时输出 1（高波动），否则 0。
  高波动期提高进攻门槛，促使策略更早切换到防御。
"""
import pandas as pd
import numpy as np

VOL_LOOKBACK = 210


def compute(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame]
        all_dates: DatetimeIndex
        assets: list[str] - 进攻型资产列表（用于计算市场波动率）
    Returns:
        Series(index=all_dates, values=0.0 or 1.0)
    """
    vols = []
    for name in assets:
        if name not in prices:
            continue
        close = prices[name]["close"].reindex(all_dates)
        returns = close.pct_change().shift(1)
        vol = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
        vols.append(vol)

    if not vols:
        return pd.Series(0.0, index=all_dates)

    market_vol = pd.concat(vols, axis=1).mean(axis=1)
    median_vol = market_vol.rolling(252).median().shift(1)
    regime = (market_vol > median_vol).astype(float)
    regime = regime.fillna(0.0)
    return regime
