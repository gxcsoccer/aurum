# Aurum × Volta 集成方案

> Aurum = 策略进化实验室（离线研发）
> Volta = AI 交易竞技场（在线纸交易）

## 一、设计理念

```
┌─────────────────────────────────────────────────────────────────┐
│                        完整工作流                                │
│                                                                 │
│  ┌──────────┐   信号    ┌───────────┐   交易    ┌──────────┐   │
│  │  Aurum   │ ───────→  │ Supabase  │ ───────→  │  Volta   │   │
│  │ (Python) │           │   (DB)    │           │ (Next.js)│   │
│  │          │   反馈    │           │   记录    │          │   │
│  │ 策略进化  │ ←──────── │  信号表    │ ←──────── │ 纸交易    │   │
│  └──────────┘           └───────────┘           └──────────┘   │
│                                                                 │
│  离线：历史数据回测         共享：PostgreSQL         在线：实时行情  │
│  频率：按需进化             存储：策略信号            频率：每15分钟  │
│  输出：月度持仓信号          桥梁：两端通信            输出：真实交易  │
└─────────────────────────────────────────────────────────────────┘
```

**核心原则：Aurum 负责"想"，Volta 负责"做"。**

- Aurum 跑几百次回测找到最优策略，每月输出一个信号："本月持有 EEM"
- Volta 接收信号，用真实行情执行交易，记录真实盈亏
- 两者通过 Supabase 数据库通信，完全解耦

### 双层频率设计

单纯月度调仓有一个致命弱点：**遇到急速崩盘（如 2020.03 一周跌 20%），要干等到下月才能调仓。**

专业机构的做法是分两层：

| 层级 | 频率 | 职责 | 触发方式 |
|---|---|---|---|
| **战略层** | 月度 | 选哪个资产 | 日历触发（每月初） |
| **战术层** | 每 15 分钟 | 要不要紧急避险 | 条件触发（价格熔断） |

```
正常市场：
  月初 → Aurum 信号说持有 QQQ → Volta 买入 → 持有整月 ✅

黑天鹅：
  月中 → QQQ 突然从高点跌 8% → Volta 熔断检测触发 → 自动切 SHY 🛡️
  下月初 → Aurum 重新计算信号 → 回归正常流程
```

战略层需要 LLM + 复杂动量计算（Aurum 负责），战术层只需要简单价格规则（Volta 内置），两者独立互不干扰。

## 二、系统架构

### 2.1 信号流（双层频率）

```
┌──────────────────────────────────────────────────────────────┐
│  战略层（月度）                                                │
│                                                              │
│  Aurum (每月运行一次)                                         │
│    │                                                         │
│    │  python publish_signal.py                               │
│    │                                                         │
│    ▼                                                         │
│  Supabase: aurum_signals 表                                  │
│    │  {"target_asset": "EEM", "valid_from": "2025-08-01"}    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  战术层（每 15 分钟）                                          │
│                                                              │
│  Volta cron (每15分钟)                                       │
│    │                                                         │
│    ├─→ 读取 Aurum 月度信号（目标资产）                         │
│    │                                                         │
│    ├─→ 熔断检测：持仓资产从近期高点跌幅 > 8%?                  │
│    │     YES → 紧急卖出，买入 SHY（现金避险）                  │
│    │     NO  → 按月度信号执行（没变就 hold）                    │
│    │                                                         │
│    ▼                                                         │
│  Volta Agent "Aurum Rotator"                                 │
│    │                                                         │
│    │  执行交易                                                │
│    ▼                                                         │
│  Supabase: trades / positions / snapshots 表                 │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  反馈层（月度）                                                │
│                                                              │
│  python sync_feedback.py                                     │
│    │                                                         │
│    ▼                                                         │
│  Aurum: 校准回测引擎的成本/滑点参数                            │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 数据边界

| 数据 | 来源 | 消费方 |
|---|---|---|
| 历史价格 (2008-2024) | yfinance → Aurum | Aurum 回测 |
| 实时行情 | Alpaca API → Volta | Volta 交易 |
| 策略信号 | Aurum → Supabase | Volta 读取 |
| 交易记录 | Volta → Supabase | Aurum 反馈 |
| Agent 排名 | Volta 计算 | Web 仪表盘展示 |

## 三、Supabase 数据模型扩展

### 3.1 新增表：aurum_signals

```sql
-- Aurum 策略信号表
CREATE TABLE aurum_signals (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,

  -- 策略标识
  strategy_name TEXT NOT NULL DEFAULT 'rotation_v1',

  -- 信号内容
  target_asset TEXT NOT NULL,         -- 目标持仓资产，如 "EEM", "SPY", "TLT"
  target_weight NUMERIC(4,2) DEFAULT 1.0,  -- 目标权重（当前为 1.0 = 全仓）

  -- 策略元数据
  metadata JSONB DEFAULT '{}',        -- {
                                      --   "score": 0.61,
                                      --   "sharpe": 1.10,
                                      --   "momentum_scores": {"EEM": 0.35, "SPY": 0.28, ...},
                                      --   "lookback_months": 12,
                                      --   "reason": "EEM 多期动量加权最强"
                                      -- }

  -- 有效期
  valid_from DATE NOT NULL,           -- 信号生效日（当月第一个交易日）
  valid_until DATE,                   -- 信号失效日（下月第一个交易日前）

  -- 审计
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 唯一约束：防止同月重复插入
ALTER TABLE aurum_signals ADD CONSTRAINT uq_signal_month
  UNIQUE (strategy_name, valid_from);

-- 索引：快速查找当前有效信号
CREATE INDEX idx_aurum_signals_valid ON aurum_signals (strategy_name, valid_from DESC);

-- RLS：公开读，服务端写
ALTER TABLE aurum_signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read" ON aurum_signals FOR SELECT USING (true);
CREATE POLICY "Service write" ON aurum_signals FOR ALL USING (true) WITH CHECK (true);
```

### 3.2 新增表：aurum_feedback（可选，Phase 3）

```sql
-- Volta 真实交易反馈，用于校准 Aurum 回测
CREATE TABLE aurum_feedback (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  month TEXT NOT NULL,                -- "2025-07"
  target_asset TEXT NOT NULL,         -- Aurum 推荐的资产
  actual_entry_price NUMERIC(10,2),   -- Volta 实际买入均价
  actual_exit_price NUMERIC(10,2),    -- Volta 实际卖出均价
  actual_return_pct NUMERIC(8,4),     -- 实际收益率
  backtest_return_pct NUMERIC(8,4),   -- Aurum 回测预估收益率
  slippage_bps NUMERIC(6,2),          -- 实际滑点 (basis points)
  fees_total NUMERIC(8,4),            -- 实际总费用
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 四、Aurum 端改造

### 4.1 新增：publish_signal.py

每月运行一次，计算当前策略信号并发布到 Supabase。

```python
# aurum/publish_signal.py
"""
计算当月策略信号并发布到 Supabase，供 Volta 读取执行。

用法：
  python publish_signal.py                    # 计算并发布
  python publish_signal.py --dry-run          # 只计算，不发布
  python publish_signal.py --month 2025-08    # 指定月份
"""
import os
import json
import argparse
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv
from supabase import create_client

from infra.data import get_multi_prices, save_multi_prices
from infra.sandbox import run_strategy

load_dotenv()


def get_current_signal(config: dict) -> dict:
    """运行策略，获取最新信号。"""
    universe = config["universe"]

    # 获取最近 18 个月数据（策略需要 12 个月回望）
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=548)).strftime("%Y-%m-%d")

    print(f"📡 获取数据 ({start_date} ~ {end_date})...")
    prices = get_multi_prices(universe, start_date, end_date)
    data_path = "data_cache/multi_latest.pkl"
    save_multi_prices(prices, data_path)

    print("🔄 运行策略...")
    strategy_code = open("strategies/strategy.py").read()
    signals = run_strategy(strategy_code, data_path)

    # 取最后一个有效信号（最近的交易日）
    latest_signal = signals.iloc[-1]
    latest_date = signals.index[-1]

    # 计算所有资产的简单 12 个月动量（仅供参考，实际策略用多期加权）
    momentum_scores = {}
    for name, df in prices.items():
        close = df["close"]
        if len(close) > 252:
            mom = close.iloc[-1] / close.iloc[-253] - 1
            momentum_scores[name] = round(float(mom), 4)

    return {
        "target_asset": str(latest_signal),
        "as_of_date": latest_date.strftime("%Y-%m-%d"),
        "momentum_scores": momentum_scores,  # 简化参考值，非策略实际评分
    }


def publish_to_supabase(signal: dict, month: str) -> None:
    """将信号写入 Supabase。"""
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise ValueError(
            "请设置环境变量：\n"
            "  NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co\n"
            "  SUPABASE_SERVICE_ROLE_KEY=eyJxxx"
        )

    client = create_client(url, key)

    # 解析月份的第一天作为 valid_from
    valid_from = f"{month}-01"

    # 下月第一天作为 valid_until
    year, mon = int(month[:4]), int(month[5:7])
    if mon == 12:
        valid_until = f"{year+1}-01-01"
    else:
        valid_until = f"{year}-{mon+1:02d}-01"

    record = {
        "strategy_name": "rotation_v1",
        "target_asset": signal["target_asset"],
        "target_weight": 1.0,
        "metadata": {
            "momentum_scores": signal["momentum_scores"],
            "as_of_date": signal["as_of_date"],
            "published_at": datetime.now().isoformat(),
        },
        "valid_from": valid_from,
        "valid_until": valid_until,
    }

    # upsert：同月重复运行会更新而非重复插入
    result = client.table("aurum_signals").upsert(
        record, on_conflict="strategy_name,valid_from"
    ).execute()
    print(f"✅ 已发布到 Supabase: {month} → 持有 {signal['target_asset']}")


def main():
    parser = argparse.ArgumentParser(description="Publish Aurum signal to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Only compute, don't publish")
    parser.add_argument("--month", type=str, default=None, help="Target month (YYYY-MM)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # 默认当月
    month = args.month or datetime.now().strftime("%Y-%m")

    print(f"📊 Aurum Signal Publisher — {month}")
    print()

    signal = get_current_signal(config)

    print(f"\n{'='*50}")
    print(f"  📌 {month} 策略信号")
    print(f"{'='*50}")
    print(f"  持有资产: {signal['target_asset']}")
    print(f"  计算日期: {signal['as_of_date']}")
    print(f"  动量评分:")
    for name, score in sorted(signal["momentum_scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(max(0, score) * 30)
        print(f"    {name:5s} {score:+.2%} {bar}")
    print(f"{'='*50}")

    if args.dry_run:
        print("\n🔍 Dry run — 未发布")
    else:
        publish_to_supabase(signal, month)


if __name__ == "__main__":
    main()
```

### 4.2 新增依赖

```toml
# pyproject.toml 新增
"supabase>=2.0",
```

### 4.3 .env 扩展

```bash
# Aurum .env 新增（与 Volta 共享 Supabase 凭证）
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJxxx
```

## 五、Volta 端改造

### 5.1 新增 Tool：aurum_signal

```typescript
// volta/src/lib/tools/aurum-signal.ts
import { supabaseAdmin } from "../supabase";
import type { ToolDefinition, ToolResult } from "./types";

export const aurumSignalTool: ToolDefinition = {
  name: "aurum_signal",
  description:
    "获取 Aurum 多资产轮动策略的当前信号。返回本月应持有的目标资产和理由。" +
    "该信号由 Aurum 策略进化系统基于多期动量、波动率调整等因子计算得出。",
  parameters: {
    type: "object" as const,
    properties: {},
    required: [],
  },
  execute: async (): Promise<ToolResult> => {
    const today = new Date().toISOString().split("T")[0];

    const { data, error } = await supabaseAdmin
      .from("aurum_signals")
      .select("*")
      .eq("strategy_name", "rotation_v1")
      .lte("valid_from", today)
      .order("valid_from", { ascending: false })
      .limit(1)
      .single();

    if (error || !data) {
      return {
        success: false,
        data: {
          error: "No active Aurum signal found",
          fallback: "无信号时默认持有 SHY（现金避险）",
        },
      };
    }

    // 信号过期检测（>45 天未更新视为过期）
    const signalDate = new Date(data.valid_from);
    const daysSinceSignal = Math.floor(
      (Date.now() - signalDate.getTime()) / (1000 * 60 * 60 * 24)
    );
    const isStale = daysSinceSignal > 45;

    const metadata = data.metadata as Record<string, unknown>;
    const scores = metadata?.momentum_scores as Record<string, number> || {};

    const ranking = Object.entries(scores)
      .sort(([, a], [, b]) => b - a)
      .map(([asset, score], i) => `${i + 1}. ${asset}: ${(score * 100).toFixed(1)}%`)
      .join("\n");

    return {
      success: true,
      data: {
        target_asset: isStale ? "SHY" : data.target_asset,
        target_weight: data.target_weight,
        valid_from: data.valid_from,
        valid_until: data.valid_until,
        is_stale: isStale,
        stale_warning: isStale
          ? `⚠️ 信号已过期 ${daysSinceSignal} 天，自动降级为持有 SHY。请检查信号发布流程。`
          : null,
        momentum_ranking: ranking,
      },
    };
  },
};
```

### 5.2 注册 Tool

```typescript
// volta/src/lib/tools/registry.ts 新增
import { aurumSignalTool } from "./aurum-signal";

// 在 toolRegistry 中添加
aurum_signal: aurumSignalTool,
```

### 5.3 扩展 Stock Pool

```typescript
// volta/src/lib/stock-pool.ts 新增 ETF
export const AURUM_ETFS = [
  "SPY",  // 已有
  "QQQ",  // 已有
  "EFA",  // 国际发达市场
  "EEM",  // 新兴市场
  "TLT",  // 长期美债
  "GLD",  // 黄金
  "SHY",  // 短期国债
];
```

### 5.4 创建 Aurum Agent

通过 Volta Admin API 或直接在 Supabase 中创建：

```json
{
  "name": "Aurum Rotator",
  "description": "基于 Aurum 自进化系统的多资产轮动策略",
  "model": "qwen3.5-plus",
  "provider": "bailian",
  "is_passive": false,
  "is_active": true,
  "watchlist": ["SPY", "QQQ", "EFA", "EEM", "TLT", "GLD", "SHY"],
  "config": {
    "model": {
      "primary": "qwen3.5-plus",
      "temperature": 0.1,
      "max_tokens": 1024
    },
    "identity": {
      "soul": "你是 Aurum Rotator，一个系统化的多资产轮动策略执行者。\n\n## 执行流程\n1. 调用 aurum_signal 获取当前月的目标持仓\n2. 如果信号标记为过期（is_stale=true），持有 SHY\n3. 对比当前持仓与目标资产\n4. 如需调仓：先卖后买\n5. 如无需调仓：hold\n\n## 核心纪律\n- 你是执行者，不是决策者\n- 严格执行信号，不做主观判断\n- 止损由系统自动处理（stop_loss_pct=8），无需你操心\n- 无信号时默认持有 SHY",
      "description": "Aurum 自进化多资产轮动策略"
    },
    "tools": ["aurum_signal"],
    "skills": [],
    "rules": {
      "max_position_pct": 100,
      "min_cash_pct": 1,
      "max_trades_per_round": 3,
      "stop_loss_pct": 8
    }
  }
}
```

### 5.5 新增 Skill（可选）：Aurum 策略说明

```markdown
# Aurum 多资产轮动策略

## 策略概要
Aurum 是一个通过 LLM 自进化发现的多资产轮动策略。
它在 7 个资产（SPY、QQQ、EFA、EEM、TLT、GLD、SHY）中，
每月选择一个风险调整动量最强的资产持有。

## 关键机制
1. **多期动量加权**：1/3/6/12 个月动量加权（权重 1/2/4/6）
2. **波动率调整**：动量 ÷ 波动率，优选风险调整后最佳资产
3. **相对 SPY 阈值**：进攻型资产动量需 ≥ SPY 动量才持有
4. **防御型纯动量**：TLT/GLD 用原始动量排名（避免 SHY 评分虚高）

## 历史表现 (2008-2024 回测)
- Sharpe: 1.10
- 年化收益: 14.3%
- 超额收益 (vs SPY): +5.26%/年
- 最大回撤: 28.6%

## 2025 Holdout 验证
- 收益: +20.22% (SPY: +18.60%)
- 最大回撤: 18.25% (SPY: 18.76%)

## 风控机制
- **战略层**：月度调仓，每月仅需交易 0-2 次
- **战术层**：每 15 分钟熔断检测，持仓资产从近期高点跌 >8% 自动切 SHY
- 熔断后不抄底，等下月信号重新评估

## 执行纪律
- 严格跟随信号，不做主观判断
- 唯一允许偏离信号的情况是熔断触发
```

## 六、竞技场 Agent 对阵表

部署后，Volta 仪表盘将展示：

| Agent | 类型 | 策略 | 模型 |
|---|---|---|---|
| **Aurum Rotator** | 🤖 系统化 | 多资产轮动（Aurum 进化） | qwen3.5-plus |
| Value Victor | 🧠 AI 主观 | 价值投资 | qwen3.5-plus |
| Momentum Max | 🧠 AI 主观 | 动量交易 | kimi-k2.5 |
| Index Ian | 📊 被动 | SPY 买入持有 | — |

**这就是最终的验证**：Aurum 的进化策略 vs 其他 AI 策略 vs 买入持有，同一赛场、同一规则、实时对比。

## 七、运营工作流

### 月度流程（~15 分钟/月）

```
每月第 1 个交易日前一天（周末）：

1. 更新数据 + 计算信号
   $ cd aurum
   $ python publish_signal.py
   → 输出：本月持有 XXX
   → 写入 Supabase

2. 检查 Volta 仪表盘
   → Aurum Rotator 在下个交易日自动调仓
   → 无需手动操作

3. 回顾上月表现
   → Volta 仪表盘查看 Aurum Rotator vs 其他 agent
   → 记录实际收益 vs 回测预测的差异
```

### 季度流程（~2 小时/季度）

```
1. 继续进化
   $ python loop.py -n 50
   → 让 agent 在最新数据上继续迭代
   → 如果发现更好的策略，自动更新 strategy.py

2. Holdout 验证
   $ python validate.py
   → 确认新策略在未见数据上仍有效

3. 反馈校准（Phase 3）
   $ python sync_feedback.py
   → 对比 Volta 实际交易 vs Aurum 回测
   → 调整成本/滑点假设
```

## 八、实施路线

| 阶段 | 内容 | 工作量 | 依赖 |
|---|---|---|---|
| **Phase 1: 信号桥** | Supabase 新表 + `publish_signal.py` | 半天 | Aurum 已完成 |
| **Phase 2: Volta 集成** | 新 Tool + 新 Agent + Stock Pool 扩展 | 半天 | Phase 1 |
| **Phase 3: 首次实战** | 发布第一个月度信号，观察 Volta 执行 | 1 小时 | Phase 2 |
| **Phase 4: 反馈循环** | `sync_feedback.py` + 回测校准 | 1 天 | 运行 1-2 个月后 |
| **Phase 5: 自动化** | GitHub Actions 定时发布信号 + 定期进化 | 1 天 | Phase 4 稳定后 |

## 九、风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| Supabase 连接失败 | Volta 读不到信号 | Agent soul 中兜底逻辑：无信号时 hold |
| Aurum 信号严重错误 | 一个月的亏损 | 月度人工检查 + **8% 熔断自动止损** |
| 月中急速崩盘 | 等不到月底调仓 | **战术层每 15 分钟熔断检测**，从高点跌 >8% 自动切 SHY |
| Alpaca 不支持某 ETF | 无法执行交易 | 预先验证所有 ETF 在 Alpaca 可交易 |
| 策略过拟合 | 实盘表现远差于回测 | 持续 Holdout 验证 + Volta 实际数据反馈 |
| 模型 API 不可用 | Agent 无法决策 | Volta 已有 fallback 机制 |

## 十、未来展望

### 短期（1-3 个月）
- 基础集成跑通
- 积累 2-3 个月的 Volta 实盘数据
- 与 Value Victor、Momentum Max 直接对比

### 中期（3-6 个月）
- 反馈循环成熟，回测越来越贴近实际
- Aurum 持续进化，策略逐步改进
- 加入更多 Volta agent（不同 LLM、不同策略风格）

### 长期（6-12 个月）
- Aurum 支持多策略同时进化
- Volta 支持多个 Aurum 策略 agent 同时运行
- 从纸交易过渡到实盘（通过 Alpaca live trading API）
- 完整的自动化管道：进化 → 验证 → 发布 → 执行 → 反馈 → 进化

## 附录 A：GitHub Actions 自动化

通过 GitHub Actions 实现 Aurum 的定时任务，无需本地运行。

### A.1 月度信号发布

```yaml
# .github/workflows/publish-signal.yml
name: Publish Monthly Signal

on:
  schedule:
    # 每月第 1 个周六 UTC 14:00（美东周六上午，市场休息，为下周一准备）
    - cron: "0 14 1-7 * 6"
  workflow_dispatch:  # 支持手动触发

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"           # 缓存 pip 依赖

      - name: Cache market data
        uses: actions/cache@v4
        with:
          path: data_cache/
          key: market-data-${{ github.run_id }}
          restore-keys: market-data-

      - name: Install dependencies
        run: pip install -e .

      - name: Publish signal
        env:
          DASHSCOPE_API_KEY: ${{ secrets.DASHSCOPE_API_KEY }}
          NEXT_PUBLIC_SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
        run: python publish_signal.py

      - name: Upload signal log
        uses: actions/upload-artifact@v4
        with:
          name: signal-${{ github.run_id }}
          path: data_cache/
          retention-days: 90
```

### A.2 季度策略进化

进化产出通过 PR 提交，需人工 review 后合并，避免自动推送有问题的策略。

```yaml
# .github/workflows/evolve.yml
name: Quarterly Evolution

on:
  schedule:
    # 每季度第一个周六 UTC 06:00
    - cron: "0 6 1-7 1,4,7,10 6"
  workflow_dispatch:
    inputs:
      iterations:
        description: "Number of evolution iterations"
        default: "50"

jobs:
  evolve:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Cache market data
        uses: actions/cache@v4
        with:
          path: data_cache/
          key: market-data-${{ github.run_id }}
          restore-keys: market-data-

      - name: Install dependencies
        run: pip install -e .

      - name: Run evolution
        env:
          DASHSCOPE_API_KEY: ${{ secrets.DASHSCOPE_API_KEY }}
        run: |
          python loop.py -n ${{ inputs.iterations || '50' }}
          python validate.py

      - name: Create PR with evolved strategy
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          BRANCH="evolve/$(date +%Y-%m-%d)"
          git checkout -b "$BRANCH"
          git config user.name "Aurum Bot"
          git config user.email "aurum@github-actions"
          git add strategies/strategy.py experiments/
          git diff --cached --quiet && echo "No changes" && exit 0
          git commit -m "evolve: auto-evolution $(date +%Y-%m-%d)"
          git push origin "$BRANCH"
          gh pr create \
            --title "Evolve: $(date +%Y-%m-%d) strategy update" \
            --body "$(cat <<'EOF'
          ## Auto Evolution Results

          This PR was created by the quarterly evolution workflow.

          **Please review the strategy changes before merging.**

          - Check `strategies/strategy.py` for the evolved strategy logic
          - Check `experiments/results.tsv` for evolution history
          - Run `python validate.py` locally to verify holdout performance
          EOF
          )" \
            --base main
```

### A.3 Secrets 配置

在 GitHub repo Settings → Secrets and variables → Actions 中添加：

| Secret | 值 | 来源 |
|---|---|---|
| `DASHSCOPE_API_KEY` | `sk-sp-xxx` | 百炼平台 |
| `SUPABASE_URL` | `https://xxx.supabase.co` | Supabase 项目设置 |
| `SUPABASE_SERVICE_KEY` | `eyJxxx` | Supabase 项目设置 → API |

## 附录 B：集成验收 Checklist

每个 Phase 完成后，逐项验证：

### Phase 1（信号桥）验收

```
□ Supabase 中 aurum_signals 表创建成功（含唯一约束）
□ python publish_signal.py --dry-run 正常输出信号
□ python publish_signal.py 成功写入 Supabase
□ 重复运行同月 publish 走 upsert 而非报错
□ Supabase 中能查到刚写入的记录
```

### Phase 2（Volta 集成）验收

```
□ Volta 中 aurum_signal tool 注册成功
□ Aurum Rotator agent 创建成功（含 $100K 账户）
□ Agent watchlist 包含所有 7 个 ETF
□ Alpaca API 能获取 EFA/EEM/TLT/GLD/SHY 的行情
□ 手动触发一轮交易，agent 能调用 aurum_signal tool
□ Agent 正确读取到目标资产并执行买入
```

### Phase 3（首次实战）验收

```
□ 发布当月真实信号到 Supabase
□ 下一个交易日 Volta cron 触发后，Aurum Rotator 完成调仓
□ Volta 仪表盘上能看到 Aurum Rotator 的持仓和收益
□ 与 Index Ian (SPY B&H) 的对比数据正确显示
```

### 边界场景验证

```
□ 信号过期测试：将 valid_from 改为 60 天前，验证 tool 返回 is_stale=true
□ 无信号测试：清空 aurum_signals 表，验证 agent 默认持有 SHY
□ 止损测试：手动将 stop_loss_pct 改为 1%，验证大跌时触发卖出
□ 重复交易测试：同一轮内不会重复买入已持有的资产
```
