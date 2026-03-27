# Aurum

面向个人投资者的自进化量化投资系统。

灵感来自 [Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch) —— LLM agent 通过贪心爬山循环自主迭代策略代码，人类通过评分函数和研究指令把控方向。

## 工作原理

```
人类层           →  scorer.py / program.md / config.yaml
Agent 层 (LLM)   →  strategies/strategy.py（唯一可变文件）
基础设施层（锁定） →  data.py / backtest.py / sandbox.py / llm.py
```

**核心循环：** LLM 提出变异 → 沙盒执行 → Walk-Forward 回测 → 评分 → 保留或丢弃。完全自主，无需人工干预。

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
| 2025 全年 | +20.22% | +18.60% | **+1.62%** |
| 2026 Q1 (1-3月) | -1.47% | -5.32% | **+3.85%** |

## 关键创新（由 LLM agent 自主发现）

1. **多期动量加权** — 1/3/6/12 个月收益率按 1/2/4/6 权重组合
2. **波动率调整排名** — 进攻型资产使用 动量/波动率 排序
3. **相对 SPY 阈值** — 进攻型资产动量需 >= SPY 动量才持有
4. **防御型纯动量** — 防止 SHY 因波动率极低导致评分虚高
5. **市场波动率过滤** — 高波动环境下收紧进攻门槛

## 快速开始

```bash
# 安装
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 设置 API Key（百炼平台）
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY

# 运行进化（50 轮，约 30 分钟）
python loop.py -n 50

# 验证保留期表现
python validate.py

# 发布月度信号（需要 Supabase 凭证）
python publish_signal.py
python publish_signal.py --dry-run    # 仅预览
```

## 项目结构

```
aurum/
├── config.yaml              # 资产池、时间窗口、LLM 设置
├── program.md               # agent 工作手册（调控研究方向）
├── loop.py                  # autoresearch 风格进化引擎
├── validate.py              # 保留期验证
├── publish_signal.py        # 月度信号发布到 Supabase
├── portfolio.py             # 多策略组合器
│
├── infra/                   # 不可变层 — 进化期间禁止修改
│   ├── data.py              # yfinance 多资产数据 + parquet/pickle 缓存
│   ├── backtest.py          # 多资产轮动回测 + walk-forward 多期验证
│   ├── scorer.py            # 评分函数（含 Deflated Sharpe Ratio）
│   ├── sandbox.py           # 子进程沙盒安全执行
│   └── llm.py               # LLM 客户端（百炼/DashScope，OpenAI 兼容）
│
├── strategies/
│   └── strategy.py          # 当前最优策略（LLM 进化此文件）
│
├── experiments/
│   └── results.tsv          # 完整实验日志（含失败记录）
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
