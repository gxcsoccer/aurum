"""
Aurum 多资产轮动策略 — 自动组装自因子库
因子数量: 5
组装时间: 2026-04-08T05:38:44.001769
"""
import pandas as pd
import numpy as np

CASH = "SHY"
OFFENSIVE = ['SPY', 'QQQ', 'EFA', 'EEM']
DEFENSIVE = ['TLT', 'GLD', 'SHY']

# ════════════════════════════════════════════
#  内联因子函数
# ════════════════════════════════════════════

# ── Factor: base_defensive_score ──
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


def compute_base_defensive_score(prices, all_dates, assets):
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


# ── Factor: base_ma_filter ──
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


def compute_base_ma_filter(prices, all_dates, assets):
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


# ── Factor: base_market_regime ──
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


def compute_base_market_regime(prices, all_dates, assets):
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


# ── Factor: base_offensive_score ──
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


def compute_base_offensive_score(prices, all_dates, assets):
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


# ── Factor: evolved_015_regime_momentum_exhaustion ──
"""
Factor: regime_momentum_exhaustion
Category: regime
Description: Aggregates momentum acceleration across offensive assets to detect systemic trend exhaustion. High value indicates broad deceleration (high risk).
"""
import pandas as pd
import numpy as np

def compute_evolved_015_regime_momentum_exhaustion(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日
        assets: list[str] - 资产列表 (ignored for regime, uses internal pool)
    Returns:
        Series(index=all_dates, values=float) - High value = High Risk
    """
    # Define offensive assets for regime calculation
    offensive_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    
    # Store acceleration signals
    accel_series = []
    
    for asset in offensive_assets:
        if asset not in prices:
            continue
            
        # Get close prices and align to all_dates
        close = prices[asset]['close'].reindex(all_dates)
        
        # Calculate returns
        # 3-month (approx 63 days) and 6-month (approx 126 days) momentum
        ret_3m = close.pct_change(63)
        ret_6m = close.pct_change(126)
        
        # Momentum Acceleration = Short Mom - Long Mom
        # Positive = Accelerating, Negative = Decelerating
        accel = ret_3m - ret_6m
        
        accel_series.append(accel)
    
    if not accel_series:
        return pd.Series(0.0, index=all_dates)
    
    # Combine into DataFrame and take mean across assets
    accel_df = pd.concat(accel_series, axis=1)
    mean_accel = accel_df.mean(axis=1)
    
    # Normalize using rolling Z-score (252 days) to make signal stationary
    rolling_mean = mean_accel.rolling(252, min_periods=63).mean()
    rolling_std = mean_accel.rolling(252, min_periods=63).std()
    
    z_score = (mean_accel - rolling_mean) / (rolling_std + 1e-9)
    
    # Invert signal: Low/Negative Acceleration = High Risk = High Regime Score
    # We want high regime score to trigger conservative behavior
    regime_score = -z_score
    
    # Ensure no look-ahead bias
    regime_score = regime_score.shift(1)
    
    # Reindex and fill NaNs
    regime_score = regime_score.reindex(all_dates)
    regime_score = regime_score.ffill().fillna(0.0)
    
    return regime_score


# ════════════════════════════════════════════
#  组合器 + 信号生成
# ════════════════════════════════════════════

MOM_PERIODS = [21, 63, 126, 252]
MOM_WEIGHTS = [1, 2, 4, 6]

def generate_signals(prices):
    # 计算日期并集
    all_dates = None
    for df in prices.values():
        idx = df.index
        if all_dates is None:
            all_dates = idx
        else:
            all_dates = all_dates.union(idx)
    all_dates = all_dates.sort_values()

    # 计算所有因子
    def_base_defensive_score = compute_base_defensive_score(prices, all_dates, DEFENSIVE)
    filter_base_ma_filter = compute_base_ma_filter(prices, all_dates, DEFENSIVE)
    regime_base_market_regime = compute_base_market_regime(prices, all_dates, OFFENSIVE)
    off_base_offensive_score = compute_base_offensive_score(prices, all_dates, OFFENSIVE)
    regime_evolved_015_regime_momentum_exhaustion = compute_evolved_015_regime_momentum_exhaustion(prices, all_dates, OFFENSIVE)

    # 组合进攻型评分
    off_score = off_base_offensive_score * 1.0000

    # 组合防御型评分
    def_score = def_base_defensive_score * 1.0000

    # 市场 regime（0=正常, 1=高风险）
    regime = (regime_base_market_regime + regime_evolved_015_regime_momentum_exhaustion) / 2

    # 计算原始动量（用于 SPY 对比）
    momentums = {}
    for name, df in prices.items():
        close = df["close"].reindex(all_dates)
        mom_values = []
        for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
            mom = close.pct_change(period).shift(1)
            mom_values.append(mom * weight)
        momentums[name] = sum(mom_values) / sum(MOM_WEIGHTS)
    mom_df = pd.DataFrame(momentums).reindex(all_dates)

    # 生成信号
    signals = pd.Series(CASH, index=all_dates)
    current_asset = CASH

    for i, date in enumerate(all_dates):
        # 月度再平衡
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        row_off = off_score.loc[date].dropna() if date in off_score.index else pd.Series(dtype=float)
        row_mom = mom_df.loc[date].dropna() if date in mom_df.index else pd.Series(dtype=float)
        row_def = def_score.loc[date].dropna() if date in def_score.index else pd.Series(dtype=float)

        if len(row_off) == 0:
            signals.iloc[i] = current_asset
            continue

        # 选进攻型最强资产
        best_off = row_off.idxmax()
        best_off_mom = row_mom.get(best_off, -1)
        spy_mom = row_mom.get("SPY", 0)

        # 波动率调整门槛
        vol_adj = 0.01 if (date in regime.index and pd.notna(regime.loc[date]) and regime.loc[date] > 0.5) else 0.0

        if best_off_mom >= spy_mom + 0.0 + vol_adj:
            current_asset = best_off
        else:
            # 防御型选择（带 filter）
            if len(row_def) > 0:
                def_sorted = row_def.sort_values(ascending=False)
                selected = None
                for asset in def_sorted.index:
                    if asset == CASH:
                        selected = CASH
                        break
                    # 应用所有 filter
                    pass_filter = True
                    if date in filter_base_ma_filter.index and asset in filter_base_ma_filter.columns:
                        if pd.notna(filter_base_ma_filter.loc[date, asset]) and filter_base_ma_filter.loc[date, asset] < 0.5:
                            pass_filter = False
                    if pass_filter:
                        selected = asset
                        break
                current_asset = selected if selected else CASH
            else:
                current_asset = CASH

        signals.iloc[i] = current_asset

    return signals
