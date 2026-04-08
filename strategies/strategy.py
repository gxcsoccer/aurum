"""
Aurum 多资产轮动策略 — 自动组装自因子库
因子数量: 12
组装时间: 2026-04-09T03:17:28.978695
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

# ── Factor: evolved_020_offensive_trend_consistency ──
"""
Factor: offensive_trend_consistency
Category: offensive
Description: 衡量进攻型资产价格路径的持续性——上涨日的比例、连续上涨日数、以及上涨日幅度的稳定性。稳步上涨（高一致性）的资产比大起大落的资产更可能延续趋势，补充现有动量因子对价格路径质量的盲区。
"""
import pandas as pd
import numpy as np

def compute_evolved_020_offensive_trend_consistency(prices, all_dates, assets):
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
            scores[asset] = np.nan
            continue
            
        df = prices[asset].reindex(all_dates)
        close = df['close']
        
        # 计算日收益率（使用前一日收盘价）
        returns = close.pct_change()
        
        # 信号 1: 上涨日比例（20 日窗口）
        up_days = (returns > 0).astype(float)
        up_ratio = up_days.rolling(window=20, min_periods=10).mean()
        
        # 信号 2: 连续上涨日数（指数衰减加权）
        consecutive_up = up_days.rolling(window=10, min_periods=5).sum()
        
        # 信号 3: 上涨日幅度稳定性（上涨日收益率的标准差，越低越稳定）
        up_returns = returns.where(returns > 0, np.nan)
        up_std = up_returns.rolling(window=20, min_periods=10).std()
        # 稳定性 = 1 / (1 + std)，标准差越低分数越高
        up_stability = 1.0 / (1.0 + up_std.fillna(1.0))
        
        # 信号 4: 下跌日平均幅度 vs 上涨日平均幅度（盈亏比）
        down_returns = returns.where(returns < 0, np.nan)
        up_mean = up_returns.rolling(window=20, min_periods=10).mean()
        down_mean = down_returns.rolling(window=20, min_periods=10).mean()
        # 盈亏比 = 平均上涨 / |平均下跌 |，避免除零
        profit_loss_ratio = up_mean / (down_mean.abs() + 1e-6)
        profit_loss_ratio = profit_loss_ratio.clip(0, 5)  # 限制极端值
        
        # 组合信号：等权重标准化后加总
        # 所有信号都需要 shift(1) 避免前视
        s1 = up_ratio.shift(1)
        s2 = (consecutive_up / 10.0).shift(1)  # 归一化到 0-1
        s3 = up_stability.shift(1)
        s4 = (profit_loss_ratio / 5.0).shift(1)  # 归一化到 0-1
        
        # 等权重组合
        combined = (s1 + s2 + s3 + s4) / 4.0
        
        # 横截面标准化（z-score）
        combined = combined.reindex(all_dates)
        scores[asset] = combined
    
    # 横截面标准化：每个交易日对所有资产进行 z-score
    for date in all_dates:
        row = scores.loc[date]
        mean_val = row.mean()
        std_val = row.std()
        if std_val > 1e-6:
            scores.loc[date] = (row - mean_val) / std_val
        else:
            scores.loc[date] = 0.0
    
    return scores

# ── Factor: evolved_036_correlation_velocity_regime ──
"""
Factor: correlation_velocity_regime
Category: regime
Description: 衡量进攻型与防御型资产间相关性的变化速度（相关性动量）。相关性快速上升表明多元化失效风险累积，往往领先于波动率爆发，提供早期风险预警。
"""
import pandas as pd
import numpy as np

def compute_evolved_036_correlation_velocity_regime(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame (columns: open/high/low/close/volume)
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        regime 因子：Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    # 定义进攻型和防御型资产
    offensive_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    defensive_assets = ['TLT', 'GLD', 'SHY']
    
    # 获取收盘价并对齐到所有日期
    close_dict = {}
    for asset in offensive_assets + defensive_assets:
        if asset in prices:
            close = prices[asset]['close'].reindex(all_dates)
            close_dict[asset] = close
    
    # 计算收益率（使用 shift(1) 确保无前视）
    ret_dict = {}
    for asset, close in close_dict.items():
        ret = close.pct_change().shift(1)
        ret_dict[asset] = ret
    
    # 计算进攻型与防御型资产间的相关性
    # 使用 60 日滚动窗口计算每对资产的相关性
    corr_window = 60
    
    correlation_series_list = []
    
    for off_asset in offensive_assets:
        for def_asset in defensive_assets:
            if off_asset in ret_dict and def_asset in ret_dict:
                off_ret = ret_dict[off_asset]
                def_ret = ret_dict[def_asset]
                
                # 计算滚动相关性
                rolling_corr = off_ret.rolling(window=corr_window, min_periods=20).corr(def_ret)
                
                # 计算相关性变化速度（相关性的动量）
                # 使用 20 日窗口衡量相关性变化的加速度
                corr_velocity = rolling_corr.diff(20).shift(1)
                
                correlation_series_list.append(corr_velocity)
    
    if len(correlation_series_list) == 0:
        return pd.Series(0.0, index=all_dates)
    
    # 将所有相关性速度信号取平均，得到综合的相关性变化速度指标
    corr_velocity_df = pd.concat(correlation_series_list, axis=1)
    avg_corr_velocity = corr_velocity_df.mean(axis=1)
    
    # 标准化信号（使用滚动标准化，确保可比性）
    rolling_mean = avg_corr_velocity.rolling(window=120, min_periods=60).mean()
    rolling_std = avg_corr_velocity.rolling(window=120, min_periods=60).std()
    
    # 计算 z-score，正值表示相关性正在快速上升（风险信号）
    regime_signal = (avg_corr_velocity - rolling_mean) / (rolling_std + 1e-8)
    
    # 确保信号在合理范围内
    regime_signal = regime_signal.clip(-3, 3)
    
    # 填充缺失值
    regime_signal = regime_signal.fillna(0)
    
    # 确保索引对齐
    regime_signal = regime_signal.reindex(all_dates).fillna(0)
    
    return regime_signal

# ── Factor: evolved_053_defensive_leadership_persisten ──
"""
Factor: defensive_leadership_persistence
Category: regime
Description: 衡量防御型资产相对于进攻型资产的超额收益持续性——持续多日的防御领先表明系统性风险规避，短暂出超可能是噪音。高值时策略应更保守。
"""
import pandas as pd
import numpy as np

def compute_evolved_053_defensive_leadership_persisten(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        regime 因子：Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    # 定义资产类别
    offensive_assets = ['SPY', 'QQQ', 'EFA', 'EEM']
    defensive_assets = ['TLT', 'GLD', 'SHY']
    
    # 获取可用的资产（处理可能缺失的资产）
    available_offensive = [a for a in offensive_assets if a in prices]
    available_defensive = [a for a in defensive_assets if a in prices]
    
    if not available_offensive or not available_defensive:
        return pd.Series(index=all_dates, data=0.0)
    
    # 获取收盘价并对齐到所有日期
    def get_close_aligned(asset_name):
        close = prices[asset_name]['close'].reindex(all_dates)
        return close
    
    # 计算进攻型和防御型的等权组合收益
    offensive_returns = []
    for asset in available_offensive:
        close = get_close_aligned(asset)
        ret = close.pct_change()
        offensive_returns.append(ret)
    offensive_avg_ret = pd.DataFrame(offensive_returns).mean(axis=0)
    
    defensive_returns = []
    for asset in available_defensive:
        close = get_close_aligned(asset)
        ret = close.pct_change()
        defensive_returns.append(ret)
    defensive_avg_ret = pd.DataFrame(defensive_returns).mean(axis=0)
    
    # 防御相对进攻的超额收益
    excess_return = defensive_avg_ret - offensive_avg_ret
    
    # 计算超额收益的符号（1=防御领先，-1=进攻领先，0=平手）
    excess_sign = np.sign(excess_return)
    
    # 计算连续防御领先的天数（持续性）
    # 使用滚动窗口计算防御领先的频率
    window = 20  # 20 交易日约 1 个月
    defensive_leadership_freq = excess_sign.rolling(window=window, min_periods=5).mean()
    
    # 防御领导频率越高，regime 值越高（应更保守）
    # 将频率从 [-1, 1] 映射到 [0, 1]
    regime_signal = (defensive_leadership_freq + 1) / 2
    
    # 确保无前视偏差
    regime_signal = regime_signal.shift(1)
    
    # 填充 NaN
    regime_signal = regime_signal.fillna(0)
    
    # 确保索引对齐
    regime_signal = regime_signal.reindex(all_dates).fillna(0)
    
    return regime_signal

# ── Factor: evolved_020_cross_asset_tlt_leading_signal ──
"""
Factor: cross_asset_tlt_leading_signal
Category: regime
Description: 使用 TLT 动量变化作为进攻型资产的领先指标——TLT 动量下降预示风险偏好上升（应进攻），TLT 动量上升预示风险规避（应保守）。捕捉跨资产领先关系而非简单相关性。
"""
import pandas as pd
import numpy as np

def compute_evolved_020_cross_asset_tlt_leading_signal(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        regime 因子：Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    # 确保所有价格数据对齐到 all_dates
    if 'TLT' not in prices:
        return pd.Series(index=all_dates, data=0.0)
    
    tlt_close = prices['TLT']['close'].reindex(all_dates)
    
    # 计算 TLT 的多时间窗口动量
    # 短期动量（5 日）- 捕捉近期变化
    tlt_mom_short = tlt_close.pct_change(5).shift(1)
    
    # 中期动量（20 日）- 捕捉趋势方向
    tlt_mom_mid = tlt_close.pct_change(20).shift(1)
    
    # 动量加速度 - 动量的变化率
    tlt_mom_accel = tlt_mom_short.diff(5).shift(1)
    
    # 标准化各信号（使用滚动标准化避免前视）
    def rolling_zscore(series, window=60):
        rolling_mean = series.rolling(window=window, min_periods=20).mean()
        rolling_std = series.rolling(window=window, min_periods=20).std()
        return (series - rolling_mean) / (rolling_std + 1e-8)
    
    tlt_mom_short_z = rolling_zscore(tlt_mom_short)
    tlt_mom_mid_z = rolling_zscore(tlt_mom_mid)
    tlt_mom_accel_z = rolling_zscore(tlt_mom_accel)
    
    # 组合信号：TLT 走强（正动量）= 风险规避信号 = 高 regime 值
    # 权重分配：中期动量为主，短期动量和加速度为确认
    regime_signal = (
        0.5 * tlt_mom_mid_z + 
        0.3 * tlt_mom_short_z + 
        0.2 * tlt_mom_accel_z
    )
    
    # 平滑信号以减少噪音
    regime_signal_smooth = regime_signal.rolling(window=5, min_periods=1).mean()
    
    # 确保输出对齐所有日期
    result = regime_signal_smooth.reindex(all_dates)
    result = result.fillna(0.0)
    
    return result

# ── Factor: evolved_022_yield_curve_regime ──
"""
Factor: yield_curve_regime
Category: regime
Description: 使用国债曲线动态（SHY vs TLT 相对表现）作为宏观制度信号。曲线趋平（SHY 跑赢 TLT）通常预示经济放缓/流动性收紧，曲线趋陡预示增长乐观。曲线动量变化比绝对曲线水平更能提前捕捉 regime 转换。
"""
import pandas as pd
import numpy as np

def compute_evolved_022_yield_curve_regime(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        Series(index=all_dates, values=float) - 高值=风险高/应保守
    """
    # Check if SHY and TLT are available
    if 'SHY' not in prices or 'TLT' not in prices:
        return pd.Series(0.0, index=all_dates)
    
    # Reindex to ensure consistent dates
    shy_close = prices['SHY']['close'].reindex(all_dates)
    tlt_close = prices['TLT']['close'].reindex(all_dates)
    
    # Calculate returns over medium-term window (1 month ~ 21 days)
    shy_ret = shy_close.pct_change(21)
    tlt_ret = tlt_close.pct_change(21)
    
    # Curve spread: SHY outperformance vs TLT (positive = flattening)
    # SHY up / TLT down = flight to safety / liquidity preference = risk signal
    curve_spread = shy_ret - tlt_ret
    
    # Momentum of curve spread (acceleration of flattening/steepening)
    # Accelerating flattening = increasing risk concern
    curve_momentum = curve_spread - curve_spread.shift(21)
    
    # Normalize by rolling std for stability
    curve_std = curve_momentum.rolling(63).std()
    curve_signal = curve_momentum / curve_std.replace(0, np.nan)
    
    # Shift to avoid look-ahead bias
    signal = curve_signal.shift(1)
    
    # Fill NaN with 0 (neutral regime)
    signal = signal.fillna(0)
    
    return signal

# ── Factor: evolved_044_evolved_043_defensive_vol_rati ──
"""
Factor: evolved_043_defensive_vol_ratio_regime
Category: regime
Description: 衡量防御型资产内部波动率比率（TLT vol / GLD vol）作为宏观风险类型信号。比率上升表明通缩/增长担忧主导，比率下降表明通胀/货币担忧主导。
"""
import pandas as pd
import numpy as np

def compute_evolved_044_evolved_043_defensive_vol_rati(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        regime 因子：Series(index=all_dates, values=float)，高值=风险高/应保守
    """
    # 获取防御型资产价格
    tlt_prices = prices.get('TLT', None)
    gld_prices = prices.get('GLD', None)
    
    # 如果缺少任一防御资产，返回中性信号
    if tlt_prices is None or gld_prices is None:
        return pd.Series(0.0, index=all_dates)
    
    # 对齐日期
    tlt_close = tlt_prices['close'].reindex(all_dates)
    gld_close = gld_prices['close'].reindex(all_dates)
    
    # 计算对数收益率
    tlt_ret = np.log(tlt_close / tlt_close.shift(1))
    gld_ret = np.log(gld_close / gld_close.shift(1))
    
    # 计算滚动波动率（21 日，约 1 个月）
    tlt_vol = tlt_ret.rolling(window=21, min_periods=10).std() * np.sqrt(252)
    gld_vol = gld_ret.rolling(window=21, min_periods=10).std() * np.sqrt(252)
    
    # 计算波动率比率（TLT / GLD）
    vol_ratio = tlt_vol / gld_vol.replace(0, np.nan)
    
    # 处理极端值和缺失值
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)
    vol_ratio = vol_ratio.fillna(vol_ratio.expanding().mean())
    
    # 标准化为 z-score（使用滚动窗口避免前视）
    vol_ratio_mean = vol_ratio.rolling(window=63, min_periods=21).mean()
    vol_ratio_std = vol_ratio.rolling(window=63, min_periods=21).std()
    vol_ratio_zscore = (vol_ratio - vol_ratio_mean) / vol_ratio_std.replace(0, np.nan)
    
    # 前视偏差修复：所有指标必须 shift(1)
    signal = vol_ratio_zscore.shift(1)
    
    # 填充早期缺失值
    signal = signal.fillna(0.0)
    
    # 限制极端值（避免单个异常值主导）
    signal = signal.clip(-3, 3)
    
    return signal

# ── Factor: evolved_054_range_expansion_momentum ──
"""
Factor: range_expansion_momentum
Category: offensive
Description: 捕捉交易区间压缩后的扩张信号——区间扩张表明机构活动增加，往往先于持续性方向突破，与基于收益率的波动率因子正交（区间可以宽但收盘接近，或区间窄但跳空），提供"能量积蓄"的独立信息。
"""
import pandas as pd
import numpy as np

def compute_evolved_054_range_expansion_momentum(prices, all_dates, assets):
    """
    Parameters:
        prices: dict[str, DataFrame] - key=资产名，value=OHLCV DataFrame
        all_dates: DatetimeIndex - 所有交易日（已排序）
        assets: list[str] - 要计算的资产列表
    Returns:
        DataFrame(index=all_dates, columns=assets, values=float scores)
    """
    scores = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    for asset in assets:
        if asset not in prices:
            scores[asset] = np.nan
            continue
        
        df = prices[asset].reindex(all_dates)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 交易区间（高 - 低）
        tr = high - low
        
        # 区间相对扩张度：当前区间 / 20 日均区间
        # 使用 min_periods=10 确保有足够数据
        tr_ma = tr.rolling(20, min_periods=10).mean()
        tr_ratio = tr / tr_ma
        
        # 动量成分（20 日收益率）
        momentum = close.pct_change(20)
        
        # 组合：区间扩张 × 动量方向
        # 区间扩张 + 正动量 = 高进攻分数（突破向上）
        # 区间扩张 + 负动量 = 低进攻分数（突破向下）
        # 区间压缩 = 低分数（方向不明）
        score = momentum.shift(1) * tr_ratio.shift(1)
        
        scores[asset] = score
    
    return scores


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
    off_evolved_020_offensive_trend_consistency = compute_evolved_020_offensive_trend_consistency(prices, all_dates, OFFENSIVE)
    regime_evolved_036_correlation_velocity_regime = compute_evolved_036_correlation_velocity_regime(prices, all_dates, OFFENSIVE)
    regime_evolved_053_defensive_leadership_persisten = compute_evolved_053_defensive_leadership_persisten(prices, all_dates, OFFENSIVE)
    regime_evolved_020_cross_asset_tlt_leading_signal = compute_evolved_020_cross_asset_tlt_leading_signal(prices, all_dates, OFFENSIVE)
    regime_evolved_022_yield_curve_regime = compute_evolved_022_yield_curve_regime(prices, all_dates, OFFENSIVE)
    regime_evolved_044_evolved_043_defensive_vol_rati = compute_evolved_044_evolved_043_defensive_vol_rati(prices, all_dates, OFFENSIVE)
    off_evolved_054_range_expansion_momentum = compute_evolved_054_range_expansion_momentum(prices, all_dates, OFFENSIVE)

    # 组合进攻型评分
    off_score = off_base_offensive_score * 0.3333 + off_evolved_020_offensive_trend_consistency * 0.3333 + off_evolved_054_range_expansion_momentum * 0.3333

    # 组合防御型评分
    def_score = def_base_defensive_score * 1.0000

    # 市场 regime（0=正常, 1=高风险）
    regime = (regime_base_market_regime + regime_evolved_015_regime_momentum_exhaustion + regime_evolved_036_correlation_velocity_regime + regime_evolved_053_defensive_leadership_persisten + regime_evolved_020_cross_asset_tlt_leading_signal + regime_evolved_022_yield_curve_regime + regime_evolved_044_evolved_043_defensive_vol_rati) / 7

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
