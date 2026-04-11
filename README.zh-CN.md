# Aurum

面向个人投资者的自进化量化投资系统。

灵感来自 [Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch) 和 [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent) —— LLM agent 自主发现并积累 alpha 因子，人类通过评分函数和研究指令把控方向。

## 工作原理

两种进化模式：

```
模式 1: 整体变异 (loop.py)                模式 2: 因子积累 (factor_loop.py) ← 推荐
  LLM 重写整个策略                          LLM 提出一个小因子
  → 沙盒 → 评分                             → 质量检查 → 加入因子库 → 重新组装
  → 保留或丢弃整个文件                       → 保留因子或丢弃（其他因子不受影响）
```

```
人类层             →  scorer.py / program.md / config.yaml
Agent 层 (LLM)     →  factors/（因子库，LLM 的画布）
组装层             →  factor_loop.py 将因子组装为 strategies/strategy.py
基础设施层（锁定）  →  data.py / backtest.py / sandbox.py / llm.py
```

**因子积累**远优于整体变异：50 轮产出 4 个有效因子，而整体变异 200 轮 0 改进。

## 成果

**当前策略：7 资产多期动量轮动**

| 资产池 | 角色 |
|---|---|
| SPY, QQQ, EFA, EEM | 进攻型（股票） |
| TLT, GLD, SHY | 防御型（债券、黄金、现金） |

### 1 万美元累计增长 (2008–2024)

| | Aurum | SPY 买入持有 |
|---|---|---|
| **最终金额** | **$160,084** | $56,023 |
| 累计收益 | +1500% | +460% |
| 年化收益 | +18.0% | +10.6% |
| Sharpe | 0.99 | ~0.55 |
| 最大回撤 | 28.6% | ~51% |

### 逐年表现

| 年份 | Aurum | SPY | 超额 | 赢家 |
|---|---|---|---|---|
| 2008 | +6.43% | -36.24% | **+42.67%** | Aurum |
| 2009 | +44.67% | +22.65% | +22.01% | Aurum |
| 2010 | +17.84% | +13.14% | +4.70% | Aurum |
| 2011 | -1.64% | +0.85% | -2.49% | SPY |
| 2012 | +11.95% | +14.17% | -2.22% | SPY |
| 2013 | +25.50% | +29.00% | -3.50% | SPY |
| 2014 | +17.87% | +14.56% | +3.31% | Aurum |
| 2015 | +9.65% | +1.29% | +8.37% | Aurum |
| 2016 | -3.01% | +13.59% | -16.60% | SPY |
| 2017 | +22.98% | +20.78% | +2.20% | Aurum |
| 2018 | -0.94% | -5.25% | +4.31% | Aurum |
| 2019 | +62.04% | +31.09% | +30.96% | Aurum |
| 2020 | +41.28% | +17.24% | +24.04% | Aurum |
| 2021 | +25.57% | +30.51% | -4.94% | SPY |
| 2022 | -11.62% | -18.65% | +7.02% | Aurum |
| 2023 | +26.42% | +26.71% | -0.29% | SPY |
| 2024 | +22.29% | +26.05% | -3.76% | SPY |

**胜率：10/17 年 (59%)** — 跌的年份明显少亏，涨的年份基本跟住。

### 样本外（完全未见数据，进化期间从未使用）

| 时段 | Aurum | SPY | 超额 |
|---|---|---|---|
| 2025 全年（8 因子版） | **+29.08%** | +18.60% | **+12.15%** |
| 2025 全年（4 因子基线） | +20.22% | +18.60% | +1.62% |

因子进化后的 8 因子版在 holdout 数据上大幅跑赢基线，证明进化出的因子具备真正的泛化能力。

## 关键创新

**基础因子**（LLM 整体变异发现）：
1. **多期动量加权** — 1/3/6/12 个月收益率按 1/2/4/6 权重组合
2. **波动率调整排名** — 进攻型资产使用 动量/波动率 排序
3. **相对 SPY 阈值** — 进攻型资产动量需 >= SPY 动量才持有
4. **防御型纯动量** — 防止 SHY 因波动率极低导致评分虚高
5. **市场波动率过滤** — 高波动环境下收紧进攻门槛

**进化因子**（LLM 因子积累发现）：
6. **动量耗尽预警** — 进攻型资产整体动量减速，领先于波动率飙升
7. **趋势一致性** — 偏好稳步上涨的资产，而非暴涨暴跌的
8. **相关性变速** — 跨资产相关性快速上升是危机先兆
9. **防御领导持续性** — 防御型资产持续跑赢进攻型，确认风险规避 regime

## 执行层现金替代

策略信号使用 SHY（1-3 年期国债）计算（回测需要 2008+ 历史数据）。
实际执行时，SHY 自动映射为 **SGOV**（iShares 0-3 个月国债 ETF）：

| | SGOV | SHY |
|---|---|---|
| 年化收益 | **2.9%** | 1.5% |
| 波动率 | **0.24%** | 1.83% |
| 最大回撤 | **-0.03%** | -5.71% |
| 2022（加息冲击） | **+1.59%** | -3.77% |

此映射在 `publish_signal.py` 中实现，可通过 `--no-substitute` 禁用。

## 快速开始

```bash
# 安装
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 设置 API Key（百炼平台）
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY

# 运行因子进化（推荐，50 轮，约 1 小时）
python factor_loop.py -n 50

# 或运行整体变异进化（旧版，50 轮，约 30 分钟）
python loop.py -n 50

# 验证保留期表现
python validate.py

# 发布月度信号（需要 Supabase 凭证）
python publish_signal.py                # SHY 自动映射为 SGOV
python publish_signal.py --dry-run      # 仅预览
python publish_signal.py --no-substitute  # 不做 SHY→SGOV 替换
```

## 项目结构

```
aurum/
├── config.yaml              # 资产池、时间窗口、LLM 设置
├── factor_config.yaml       # 因子注册表、权重、组合器参数
├── program.md               # 整体变异 agent 工作手册
├── program_factor.md        # 因子挖掘 agent 工作手册
├── loop.py                  # 整体变异进化引擎（旧版）
├── factor_loop.py           # 因子积累进化引擎（推荐）
├── validate.py              # 保留期验证
├── publish_signal.py        # 月度信号发布（SHY→SGOV 映射）
├── portfolio.py             # 多策略组合器
│
├── factors/                 # 因子库（基础 + LLM 进化）
│   ├── base_offensive_score.py       # 波动率调整多期动量
│   ├── base_defensive_score.py       # 纯多期动量
│   ├── base_market_regime.py         # 市场波动率 regime
│   ├── base_ma_filter.py             # 63 日均线趋势过滤
│   ├── evolved_015_*.py              # 动量耗尽预警
│   ├── evolved_020_*.py              # 趋势一致性
│   ├── evolved_036_*.py              # 相关性变速
│   └── evolved_053_*.py              # 防御领导持续性
│
├── infra/                   # 不可变层 — 进化期间禁止修改
│   ├── data.py              # yfinance 多资产数据 + parquet/pickle 缓存
│   ├── backtest.py          # 多资产轮动回测 + walk-forward 多期验证
│   ├── scorer.py            # 评分函数（含 Deflated Sharpe Ratio）
│   ├── sandbox.py           # 子进程沙盒安全执行
│   └── llm.py               # LLM 客户端（百炼/DashScope，OpenAI 兼容）
│
├── strategies/
│   └── strategy.py          # 当前最优策略（由因子库自动组装）
│
├── experiments/
│   ├── results.tsv          # 整体变异实验日志
│   └── factor_results.tsv   # 因子进化实验日志
│
└── .github/workflows/
    ├── publish-signal.yml   # 月度自动发布轮动信号
    └── evolve.yml           # 季度自动进化策略（创建 PR）
```

## 评分哲学

评分函数（`infra/scorer.py`）就是你的**投资哲学的代码化表达**。改变权重，agent 就会向完全不同的方向进化。

当前设计（v5）：
- **超额收益 vs SPY 买入持有** — 最核心指标（权重 3.0）
- **绝对收益** — 原始表现（权重 1.0）
- **Deflated Sharpe Ratio** — 抗过拟合：根据总实验次数调整信心
- **现金惩罚** — >30% 开始扣分，>70% 硬淘汰
- **一致性** — 年度间超额收益方差要低

每一版评分函数最终都被 agent "破解"了：

| 漏洞 | Agent 行为 | 修复方式 |
|---|---|---|
| 回撤惩罚太高 | 学会不交易（0 交易 = 0 回撤） | 加参与率惩罚 |
| 回撤优势加分 | 100% 持现金得满分 | 删除该加分项 |
| 现金惩罚 >60% 才触发 | 保持 59% 现金刚好不触发 | 改为 >30% 渐进式惩罚 |

## Volta 集成

Aurum 与 [Volta](https://github.com/gxcsoccer/volta)（AI 交易竞技场）集成，实现纸交易：

```
Aurum（离线）                Volta（在线）
  │                            │
  │  publish_signal.py         │  每 15 分钟
  │         │                  │       │
  ▼         ▼                  ▼       ▼
  策略    → Supabase ──────→ Aurum Rotator agent
  进化      信号表             读取信号，执行交易
```

- **战略层（月度）：** Aurum 计算持有哪个资产
- **战术层（15 分钟）：** Volta 检查熔断条件（stop_loss=8%）
- **反馈层（月度）：** 真实交易数据校准回测假设

## 抗过拟合机制

- **Walk-Forward 验证** — 按年度子期间独立评估（非单次 train/test 分割）
- **Deflated Sharpe Ratio**（Lopez de Prado）— 实验次数越多，门槛越高
- **保留期** — 2025 年数据在进化期间从未使用
- **沙盒执行** — 策略在子进程中运行，无法篡改评分函数
- **因子数量纪律** — 实验证明 8 因子为最佳数量；12 因子导致 holdout 超额从 +12% 下降到 +7%（过拟合）

## 配置

`config.yaml` 核心设置：

```yaml
universe: [SPY, QQQ, EFA, EEM, TLT, GLD, SHY]
eval_start: "2008-01-01"
eval_end: "2024-12-31"
holdout_start: "2025-01-01"
max_drawdown_limit: 0.30
cost_per_trade: 0.001

llm:
  base_url: "https://coding.dashscope.aliyuncs.com/v1"
  models: ["qwen3.5-plus"]
```

## 许可证

MIT
