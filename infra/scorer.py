"""
评分函数 — 不可变层，agent 不能修改此文件。
多资产轮动版：核心目标是跑赢 SPY buy-and-hold。

v5: 加入 Deflated Sharpe Ratio (M2 抗过拟合)
  - DSR 校正：跑 N 次实验后找到 Sharpe=1.5 可能只是运气
  - DSR 随实验次数增加自动提高门槛
  - 保留 v4 的现金惩罚和超额收益权重
"""
import os
import numpy as np
from scipy import stats


def _deflated_sharpe_ratio(observed_sharpe: float,
                           sharpe_std: float,
                           n_trials: int,
                           n_obs: int) -> float:
    """
    Deflated Sharpe Ratio (Lopez de Prado, 2014)

    校正多重假设检验偏差：跑了 n_trials 次实验，
    最优 Sharpe 在统计上是否显著优于随机？

    参数：
      observed_sharpe: 观测到的最优 Sharpe
      sharpe_std:      Sharpe 的标准差（跨子期间）
      n_trials:        总实验次数（含失败）
      n_obs:           观测数量（子期间数）

    返回：
      DSR 值 (0-1)，越高越有信心不是运气
    """
    if n_trials <= 1 or sharpe_std < 1e-9 or n_obs < 2:
        return 1.0  # 不足以校正，跳过

    # 期望最大 Sharpe（在 n_trials 次独立试验中的期望最大值）
    # E[max(Z_1, ..., Z_n)] ≈ (1 - γ) * Φ^{-1}(1 - 1/n) + γ * Φ^{-1}(1 - 1/(n*e))
    # 简化近似：使用 Euler-Mascheroni 常数
    euler_mascheroni = 0.5772156649
    z_max = stats.norm.ppf(1 - 1 / n_trials) * (1 - euler_mascheroni) + \
            euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))

    expected_max_sharpe = sharpe_std * z_max

    # PSR (Probabilistic Sharpe Ratio)
    # 测试 observed_sharpe 是否显著大于 expected_max_sharpe
    if sharpe_std < 1e-9:
        return 1.0

    test_stat = (observed_sharpe - expected_max_sharpe) / (sharpe_std / np.sqrt(n_obs))
    dsr = float(stats.norm.cdf(test_stat))

    return dsr


def _get_total_trials() -> int:
    """读取实验日志，获取总实验次数。"""
    results_file = "experiments/results.tsv"
    if not os.path.exists(results_file):
        return 1
    with open(results_file) as f:
        lines = f.readlines()
    return max(len(lines) - 1, 1)  # 减去 header


def score(sub_results: list[dict],
          max_dd_limit: float = 0.30,
          min_sharpe: float = 0.0) -> float:
    """
    基于 walk-forward 多期结果计算综合评分。越高越好。

    v5 = v4 + Deflated Sharpe Ratio
    DSR 的作用：你跑了 200 次实验才找到 Sharpe=1.5 的策略，
    DSR 会告诉你"这有多大概率只是数据挖掘的运气"。
    """
    if not sub_results:
        return -999.0

    sharpes = [r["sharpe"] for r in sub_results]
    returns = [r["total_return"] for r in sub_results]
    drawdowns = [r["max_drawdown"] for r in sub_results]
    excess_returns = [r["excess_return"] for r in sub_results]
    cash_pcts = [r["cash_pct"] for r in sub_results]
    n_switches_list = [r["n_switches"] for r in sub_results]

    avg_cash = np.mean(cash_pcts)

    # ── 硬约束 ──
    if max(drawdowns) > max_dd_limit:
        return -999.0
    if np.mean(sharpes) < min_sharpe:
        return -999.0
    if avg_cash > 0.70:
        return -999.0

    # ── Deflated Sharpe Ratio ──
    n_trials = _get_total_trials()
    observed_sharpe = float(np.mean(sharpes))
    sharpe_std = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0
    n_obs = len(sharpes)

    dsr = _deflated_sharpe_ratio(observed_sharpe, sharpe_std, n_trials, n_obs)

    # ── 综合评分 ──
    s = (
        # 超额收益（最核心）
        np.mean(excess_returns) * 3.0

        # 绝对收益
        + np.mean(returns) * 1.0

        # 风险调整收益（DSR 加权：信心越高，Sharpe 贡献越大）
        + np.mean(sharpes) * 0.4 * dsr

        # 一致性
        - np.std(excess_returns) * 0.3

        # 回撤惩罚
        - max(drawdowns) * 0.5

        # 现金惩罚
        - max(0, avg_cash - 0.3) * 2.0
        - max(0, avg_cash - 0.5) * 3.0

        # 过度交易惩罚
        - max(0, np.mean(n_switches_list) / 12 - 2) * 0.2
    )
    return round(float(s), 6)
