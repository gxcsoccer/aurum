"""
回测引擎 — 不可变层，agent 不能修改此文件。
支持多资产轮动回测和 walk-forward 多期验证。
"""
import pandas as pd
import numpy as np


def backtest_rotation(signals: pd.Series,
                      all_prices: dict[str, pd.DataFrame],
                      cash_asset: str = "SHY",
                      benchmark: str = "SPY",
                      cost: float = 0.001) -> dict:
    """
    多资产轮动回测。

    参数：
      signals:    Series, index=日期, values=资产名(str)
      all_prices: dict, key=资产名, value=OHLCV DataFrame
      cash_asset: 默认现金资产名
      benchmark:  基准资产名（用于计算超额收益）
      cost:       每次切换资产的交易成本

    返回：
      dict 包含 sharpe, total_return, excess_return, max_drawdown 等指标
    """
    # 构建每个资产的日收益率
    asset_returns = {}
    for name, df in all_prices.items():
        asset_returns[name] = df["close"].pct_change().fillna(0)

    # 公共日期
    common_idx = signals.index
    for name in asset_returns:
        common_idx = common_idx.intersection(asset_returns[name].index)
    common_idx = common_idx.sort_values()

    signals = signals.reindex(common_idx).fillna(cash_asset)

    # 执行延迟：今天决策，明天执行
    held_asset = signals.shift(1).fillna(cash_asset)

    # 每日组合收益
    port_return = pd.Series(0.0, index=common_idx)
    for i, date in enumerate(common_idx):
        asset = held_asset.iloc[i]
        if asset in asset_returns:
            port_return.iloc[i] = asset_returns[asset].get(date, 0.0)

    # 切换成本
    switches = (held_asset != held_asset.shift(1))
    switches.iloc[0] = False
    n_switches = int(switches.sum())
    port_return = port_return - cost * switches.astype(float)

    # 累计净值
    cumulative = (1 + port_return).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0

    # 年化
    n_days = len(port_return)
    ann_return = port_return.mean() * 252
    ann_vol = port_return.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 1e-9 else 0.0

    # 最大回撤
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # 基准 buy-and-hold
    if benchmark in asset_returns:
        bh_ret = asset_returns[benchmark].reindex(common_idx).fillna(0)
        bh_total = (1 + bh_ret).cumprod().iloc[-1] - 1
        excess_return = total_return - bh_total

        bh_cumulative = (1 + bh_ret).cumprod()
        bh_rolling_max = bh_cumulative.cummax()
        bh_dd = (bh_cumulative - bh_rolling_max) / bh_rolling_max
        bh_max_dd = abs(bh_dd.min())
    else:
        bh_total = 0.0
        excess_return = total_return
        bh_max_dd = 0.0

    # 现金占比
    cash_days = (held_asset == cash_asset).sum()
    cash_pct = cash_days / max(len(held_asset), 1)

    # 各资产持仓占比
    allocation = held_asset.value_counts(normalize=True).to_dict()

    # 胜率（日度）
    active_returns = port_return[held_asset != cash_asset]
    win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0.0

    return {
        "sharpe": round(float(sharpe), 4),
        "total_return": round(float(total_return), 4),
        "ann_return": round(float(ann_return), 4),
        "ann_vol": round(float(ann_vol), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "excess_return": round(float(excess_return), 4),
        "bh_return": round(float(bh_total), 4),
        "bh_max_dd": round(float(bh_max_dd), 4),
        "cash_pct": round(float(cash_pct), 4),
        "n_switches": n_switches,
        "win_rate": round(float(win_rate), 4),
        "n_days": n_days,
        "allocation": allocation,
    }


def walk_forward_rotation(signals: pd.Series,
                          all_prices: dict[str, pd.DataFrame],
                          cash_asset: str = "SHY",
                          benchmark: str = "SPY",
                          cost: float = 0.001,
                          sub_period: str = "yearly") -> list[dict]:
    """Walk-forward 多期验证（多资产版）。"""
    results = []

    if sub_period == "yearly":
        groups = signals.groupby(signals.index.year)
    elif sub_period == "quarterly":
        groups = signals.groupby(
            [signals.index.year, signals.index.quarter]
        )
    else:
        raise ValueError(f"Unknown sub_period: {sub_period}")

    for key, sub_signals in groups:
        if len(sub_signals) < 20:
            continue
        sub_prices = {}
        for name, df in all_prices.items():
            sub_df = df.loc[df.index.isin(sub_signals.index)]
            if len(sub_df) >= 20:
                sub_prices[name] = sub_df

        if len(sub_prices) < 2:
            continue

        result = backtest_rotation(
            sub_signals, sub_prices, cash_asset, benchmark, cost
        )
        result["period"] = str(key)
        results.append(result)

    return results
