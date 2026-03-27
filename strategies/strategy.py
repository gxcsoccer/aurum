import pandas as pd
import numpy as np

# ============ 参数区 ============
LOOKBACK = 252          # 动量回望期（~12 个月）
VOL_LOOKBACK = 21       # 波动率回望期（~1 个月）
CASH = "SHY"            # 现金等价资产
OFFENSIVE = ["SPY", "QQQ", "EFA", "EEM"]   # 进攻型资产
DEFENSIVE = ["TLT", "GLD", "SHY"]          # 防御型资产
MOMENTUM_THRESHOLD_NORMAL = -0.05  # 正常动量阈值（-5%）
MOMENTUM_THRESHOLD_BREADTH = -0.02  # 广度恶化时的严格阈值（-2%）
BREADTH_THRESHOLD = 0.5  # 正动量资产比例阈值（50%）
QQQ_VOL_PENALTY_RATIO = 1.3  # QQQ 波动率相对于 SPY 的惩罚阈值
QQQ_VOL_PENALTY_FACTOR = 0.7  # QQQ 评分惩罚系数（当波动率过高时）

# ============ 信号逻辑区 ============
def generate_signals(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """
    输入：prices dict，key=资产名，value=OHLCV DataFrame
    输出：Series，index=日期，values=持有的资产名 (str)

    变异：结合 QQQ 波动率惩罚 + 市场广度自适应阈值
    1. 计算每个资产的动量和波动率
    2. 评分 = 动量 / 波动率
    3. 当 QQQ 波动率 > 1.3x SPY 波动率时，对 QQQ 评分施加 30% 惩罚
    4. 计算市场广度（进攻型资产中正动量的比例）
    5. 当广度 < 50% 时，使用更严格的 -2% 阈值（而非 -5%）
    6. 在进攻型资产中选评分最高的
    7. 如果最强资产的原始动量 > 当前阈值 → 持有它
    8. 否则 → 切换到防御型资产中评分最高的
    9. 每月初再平衡一次
    """
    # 获取公共日期
    all_dates = None
    for df in prices.values():
        idx = df.index
        if all_dates is None:
            all_dates = idx
        else:
            all_dates = all_dates.intersection(idx)
    all_dates = all_dates.sort_values()

    # 计算每个资产的动量和波动率（shift(1) 防止前视偏差）
    momentums = {}
    volatilities = {}
    for name, df in prices.items():
        close = df["close"].reindex(all_dates)
        # 动量：12 个月收益率
        momentums[name] = close.pct_change(LOOKBACK).shift(1)
        # 波动率：21 日收益率标准差
        volatilities[name] = close.pct_change().rolling(VOL_LOOKBACK).std().shift(1)

    mom_df = pd.DataFrame(momentums).reindex(all_dates)
    vol_df = pd.DataFrame(volatilities).reindex(all_dates)
    
    # 计算波动率调整动量评分 (避免除以 0)
    score_df = mom_df / (vol_df + 1e-6)

    # 生成信号
    signals = pd.Series(CASH, index=all_dates)
    current_asset = CASH

    for i, date in enumerate(all_dates):
        # 月度再平衡：仅在每月第一个交易日调仓
        if i > 0 and date.month == all_dates[i - 1].month:
            signals.iloc[i] = current_asset
            continue

        # 获取当日数据
        row_score = score_df.loc[date].dropna()
        row_mom = mom_df.loc[date].dropna()
        row_vol = vol_df.loc[date].dropna()
        
        if len(row_score) == 0:
            signals.iloc[i] = current_asset
            continue

        # 选进攻型资产中评分最高的
        off_score = {k: row_score[k] for k in OFFENSIVE if k in row_score}
        
        # QQQ 波动率惩罚：当 QQQ 波动率 > 1.3x SPY 波动率时，降低其评分
        if "QQQ" in off_score and "SPY" in row_vol:
            qqq_vol = row_vol.get("QQQ", 0)
            spy_vol = row_vol.get("SPY", 0)
            if spy_vol > 0 and qqq_vol > spy_vol * QQQ_VOL_PENALTY_RATIO:
                off_score["QQQ"] = off_score["QQQ"] * QQQ_VOL_PENALTY_FACTOR
        
        if off_score:
            best_off = max(off_score, key=off_score.get)
            best_off_mom = row_mom.get(best_off, -1)
        else:
            best_off = CASH
            best_off_mom = -1

        # 计算市场广度（进攻型资产中正动量的比例）
        off_moms = {k: row_mom.get(k, -1) for k in OFFENSIVE if k in row_mom}
        if off_moms:
            positive_count = sum(1 for v in off_moms.values() if v > 0)
            breadth = positive_count / len(off_moms)
        else:
            breadth = 0
        
        # 广度自适应阈值：当广度 < 50% 时使用更严格的 -2% 阈值
        if breadth < BREADTH_THRESHOLD:
            momentum_threshold = MOMENTUM_THRESHOLD_BREADTH
        else:
            momentum_threshold = MOMENTUM_THRESHOLD_NORMAL

        # 动量严重程度过滤：只有动量 < 阈值才切换防御
        if best_off_mom > momentum_threshold:
            current_asset = best_off
        else:
            # 选防御型中评分最高的
            def_score = {k: row_score[k] for k in DEFENSIVE if k in row_score}
            if def_score:
                current_asset = max(def_score, key=def_score.get)
            else:
                current_asset = CASH

        signals.iloc[i] = current_asset

    return signals