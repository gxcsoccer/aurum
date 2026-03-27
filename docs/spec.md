好，我来完整拆解，从零到跑起来。先看整体架构，再逐步讲。整个系统就是这个闭环。现在逐步拆解每一层怎么做。

---

## Step 1：搭数据管道（地基，不能动）

这层放在 `prepare.py`，agent 绝对不碰。

```python
# prepare.py — 你写一次，永远不改
import finnhub
import pandas as pd

client = finnhub.Client(api_key="你的KEY")

def get_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """严格按日期切割，绝不泄露未来数据"""
    candles = client.stock_candles(symbol, 'D', 
                                    int(pd.Timestamp(start).timestamp()),
                                    int(pd.Timestamp(end).timestamp()))
    df = pd.DataFrame(candles)
    df.index = pd.to_datetime(df['t'], unit='s')
    return df[['o','h','l','c','v']].rename(columns={
        'o':'open','h':'high','l':'low','c':'close','v':'volume'
    })

# 固定的时间分割（绝对不能让 agent 知道 OOS 数据的存在）
TRAIN_START = "2020-01-01"
TRAIN_END   = "2023-12-31"  # agent 能看到的
OOS_START   = "2024-01-01"  # 人工持有，最终验证用
OOS_END     = "2024-12-31"
```

唯一要注意的一件事：**时间戳必须以 close 价格为准**。今天的信号只能用今天收盘前的数据，明天的 open 才能执行。这条写错了，后面所有回测全作废。

---

## Step 2：定义 strategy.py 模板（agent 的"沙盒"）

给 agent 一个初始可运行的策略，参数区和逻辑区明确分开。

```python
# strategy.py — agent 可以改这里的所有内容
# ============ 参数区（agent 重点关注） ============
LOOKBACK     = 20      # 动量回望期（天）
ENTRY_THRESH = 0.05    # 入场阈值（涨幅%）
EXIT_THRESH  = -0.03   # 止损阈值
VOL_FILTER   = True    # 是否过滤高波动期
VOL_WINDOW   = 10      # 波动率计算窗口
HOLD_DAYS    = 5       # 最长持仓天数

# ============ 信号逻辑区（agent 可重写） ============
def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    输入：OHLCV DataFrame
    输出：signals，值为 1（做多）或 0（空仓）
    规则：不能使用未来数据（shift(1) 确保）
    """
    returns = df['close'].pct_change(LOOKBACK).shift(1)
    signal = (returns > ENTRY_THRESH).astype(int)
    
    if VOL_FILTER:
        vol = df['close'].pct_change().rolling(VOL_WINDOW).std().shift(1)
        high_vol = vol > vol.rolling(60).mean() * 1.5
        signal[high_vol] = 0
    
    return signal
```

这个模板是起点，agent 会在这个基础上迭代。初始策略不需要好，只需要能跑通。

---

## Step 3：写评分函数（你的投资哲学具象化）

这是整个系统最重要的一步。评分函数决定 agent 会朝什么方向进化。

```python
# scorer.py
def backtest(signals: pd.Series, prices: pd.DataFrame, 
             cost_per_trade: float = 0.001) -> dict:
    """走完整个回测，返回统计数据"""
    pos = signals.shift(1)  # 次日开盘执行
    daily_ret = prices['close'].pct_change()
    strat_ret = pos * daily_ret - cost_per_trade * pos.diff().abs()
    
    # 计算各项指标
    sharpe = strat_ret.mean() / strat_ret.std() * (252 ** 0.5)
    cumret = (1 + strat_ret).cumprod()
    max_dd = (cumret / cumret.cummax() - 1).min()
    turnover = pos.diff().abs().mean()
    win_rate = (strat_ret[pos == 1] > 0).mean()
    
    return {
        'sharpe': sharpe,
        'max_drawdown': abs(max_dd),
        'turnover': turnover,
        'win_rate': win_rate,
        'total_return': cumret.iloc[-1] - 1
    }

def score(results: dict) -> float:
    """综合评分，值越高越好"""
    # 硬约束（直接淘汰）
    if results['max_drawdown'] > 0.20:  return -999.0
    if results['turnover'] > 1.5:       return -999.0
    if results['sharpe'] < 0:           return -999.0
    
    # 综合加权（这里的权重是你的判断）
    return (
        results['sharpe'] * 1.0
        - results['max_drawdown'] * 2.0   # 回撤惩罚加倍
        - results['turnover'] * 0.5        # 换手率轻惩罚
        + results['win_rate'] * 0.3        # 胜率小奖励
    )
```

**花时间在这个函数上比花时间研究策略更值得**。你改一次权重，agent 就会往完全不同的方向进化。

---

## Step 4：写 program.md（给 agent 的约束手册）

这是你和 agent 之间的"合同"，每次调用 LLM 都放在 system prompt 里。

```markdown
# 量化策略研究员工作手册

## 你的角色
你是一个量化策略研究员。你的目标是通过修改 strategy.py 
来提升策略的综合评分。

## 你可以修改的内容
- 参数区的所有数值（LOOKBACK, ENTRY_THRESH 等）
- generate_signals 函数的逻辑
- 增加新的技术指标（必须使用 .shift(1) 防止前视偏差）

## 你绝对不能修改的内容
- prepare.py（数据管道）
- scorer.py（评分函数）
- 任何数据读取相关的代码

## 每次变异的规则
1. 每次只改一处（一个参数 或 一段逻辑），不允许同时修改多处
2. 提出变异前，先用一句话说明你的假设："我认为...因为..."
3. 变异必须有明确的可解释性，不允许随机改数字
4. 不允许添加任何会导致前视偏差的代码

## 前视偏差检查清单（每次变异后自查）
- [ ] 所有信号计算是否都有 .shift(1)？
- [ ] 是否使用了当天的 close 来做当天的交易决策？
- [ ] rolling 计算是否都是向后看的（不包含未来数据）？
```

---

## Step 5：搭 autoquant loop（核心引擎）

```python
# autoquant_loop.py
import sqlite3
import anthropic
import importlib.util
from datetime import datetime

client = anthropic.Anthropic()

def setup_db():
    conn = sqlite3.connect('experiments.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            strategy_code TEXT,
            sharpe REAL, max_dd REAL, turnover REAL,
            score REAL, mutation_desc TEXT, kept INTEGER
        )
    ''')
    return conn

def get_recent_history(conn, n=20) -> str:
    """给 agent 看最近 N 次实验历史，帮助它做更好的决策"""
    rows = conn.execute(
        'SELECT mutation_desc, score, kept FROM experiments ORDER BY id DESC LIMIT ?', 
        (n,)
    ).fetchall()
    lines = [f"{'✅' if r[2] else '❌'} {r[1]:.3f} | {r[0]}" for r in rows]
    return "\n".join(lines)

def ask_llm_for_mutation(strategy_code: str, last_result: dict, 
                          history: str, program_md: str) -> tuple[str, str]:
    """让 LLM 提出一个变异，返回 (新代码, 变异描述)"""
    prompt = f"""
{program_md}

---
## 最近实验历史（供参考，不要重复已失败的方向）
{history}

---
## 当前策略代码
```python
{strategy_code}
```

## 上次回测结果
- Sharpe: {last_result['sharpe']:.3f}
- 最大回撤: {last_result['max_drawdown']:.1%}
- 换手率: {last_result['turnover']:.2f}x/月
- 综合评分: {last_result['score']:.3f}

请提出一个单一变异。先说明你的假设，然后输出完整的 strategy.py 代码。
格式：
HYPOTHESIS: [你的假设]
CODE:
```python
[完整的 strategy.py]
```
"""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.content[0].text
    hypothesis = text.split("CODE:")[0].replace("HYPOTHESIS:", "").strip()
    code = text.split("```python")[1].split("```")[0].strip()
    return code, hypothesis

def run_strategy(code: str, prices) -> dict:
    """动态执行策略代码，返回回测结果"""
    exec_globals = {}
    exec(code, exec_globals)
    signals = exec_globals['generate_signals'](prices)
    results = backtest(signals, prices)
    results['score'] = score(results)
    return results

def run_loop(symbol: str, n_iterations: int = 100):
    from prepare import get_price_data, TRAIN_START, TRAIN_END
    from scorer import backtest, score
    
    conn = setup_db()
    prices = get_price_data(symbol, TRAIN_START, TRAIN_END)
    program_md = open('program.md').read()
    
    # 读初始策略
    current_code = open('strategy.py').read()
    current_result = run_strategy(current_code, prices)
    print(f"初始评分: {current_result['score']:.3f}")
    
    for i in range(n_iterations):
        print(f"\n--- 第 {i+1} 次迭代 ---")
        history = get_recent_history(conn)
        
        # LLM 提变异
        new_code, hypothesis = ask_llm_for_mutation(
            current_code, current_result, history, program_md
        )
        print(f"假设: {hypothesis[:80]}...")
        
        # 跑回测
        try:
            new_result = run_strategy(new_code, prices)
        except Exception as e:
            print(f"执行报错: {e}，跳过")
            continue
        
        kept = new_result['score'] > current_result['score']
        
        # 记录实验
        conn.execute('''
            INSERT INTO experiments 
            (timestamp, strategy_code, sharpe, max_dd, turnover, score, mutation_desc, kept)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), new_code,
              new_result['sharpe'], new_result['max_drawdown'],
              new_result['turnover'], new_result['score'],
              hypothesis, int(kept)))
        conn.commit()
        
        if kept:
            current_code = new_code
            current_result = new_result
            open('strategy.py', 'w').write(current_code)
            print(f"✅ 改进! {current_result['score']:.3f} → {new_result['score']:.3f}")
        else:
            print(f"❌ 回退. ({new_result['score']:.3f} < {current_result['score']:.3f})")
    
    print(f"\n完成! 最终评分: {current_result['score']:.3f}")
    print(f"最终 Sharpe: {current_result['sharpe']:.3f}")

if __name__ == "__main__":
    run_loop("AAPL", n_iterations=100)
```

---

## Step 6：运行 + 监控

**启动方式：**
```bash
# 跑 100 次迭代，大约需要 1-2 小时（取决于 API 速度）
python autoquant_loop.py

# 或者挂后台跑
nohup python autoquant_loop.py > autoquant.log 2>&1 &
```

**监控实验进展（另开一个终端）：**
```python
# monitor.py — 随时查看实验历史
import sqlite3, pandas as pd

conn = sqlite3.connect('experiments.db')
df = pd.read_sql('SELECT * FROM experiments ORDER BY id', conn)

print(f"总实验次数: {len(df)}")
print(f"成功改进次数: {df['kept'].sum()} ({df['kept'].mean():.0%})")
print(f"\n最优结果:")
print(df.loc[df['score'].idxmax(), ['sharpe','max_dd','turnover','score','mutation_desc']])

print(f"\n最近10次实验:")
print(df.tail(10)[['score','kept','mutation_desc']].to_string())
```

---

## Step 7：OOS 验证（最终把关，不能省）

等 loop 跑完，拿出你从没动过的 2024 年数据，验证一次。

```python
# validate_oos.py
from prepare import get_price_data, OOS_START, OOS_END
from scorer import backtest, score

# 读取 loop 跑完后的最优策略
final_code = open('strategy.py').read()
prices_oos = get_price_data("AAPL", OOS_START, OOS_END)

exec_globals = {}
exec(final_code, exec_globals)
signals = exec_globals['generate_signals'](prices_oos)
results = backtest(signals, prices_oos)

print("=== OOS 验证结果 ===")
print(f"Sharpe:   {results['sharpe']:.3f}")
print(f"最大回撤: {results['max_drawdown']:.1%}")
print(f"总收益:   {results['total_return']:.1%}")

# 判断标准（可以调整）
if results['sharpe'] > 0.8 and results['max_drawdown'] < 0.15:
    print("\n✅ 通过 OOS 验证，可以进入纸交易观察")
else:
    print("\n❌ OOS 表现不达标，策略可能过拟合，继续优化或换方向")
```

---

## 整体时间表

| 阶段 | 工作 | 预计时间 |
|---|---|---|
| Step 1-3 | 搭数据管道 + 策略模板 + 评分函数 | 1天 |
| Step 4-5 | 写 program.md + autoquant loop | 半天 |
| Step 6 | 跑第一轮100次迭代 | 1-2小时挂机 |
| Step 7 | OOS 验证 + 判断 | 30分钟 |
| 迭代 | 根据结果调整评分函数 / 换标的 / 换策略框架 | 持续进行 |

---

**最后一个判断原则：** OOS Sharpe > in-sample Sharpe 的 60% 才算健康。如果 in-sample 跑出 1.5，OOS 只有 0.3，说明过拟合严重，回去检查评分函数里的惩罚力度是否不够。这不是失败，是系统在正确地告诉你什么。