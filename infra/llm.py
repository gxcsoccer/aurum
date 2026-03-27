"""
LLM 客户端 — 不可变层，agent 不能修改此文件。
通过百炼平台的 OpenAI 兼容接口调用大模型。
"""
import os
import re

from openai import OpenAI


def create_client(config: dict) -> OpenAI:
    """创建百炼 API 客户端。"""
    env_name = config["llm"].get("api_key_env", "DASHSCOPE_API_KEY")
    api_key = os.getenv(env_name)
    if not api_key:
        raise ValueError(
            f"请设置环境变量 {env_name}\n"
            f"  export {env_name}=sk-xxxxxxxx\n"
        )
    return OpenAI(
        api_key=api_key,
        base_url=config["llm"]["base_url"],
    )


def ask_mutation(client: OpenAI, model: str,
                 strategy_code: str, current_score: float,
                 current_metrics: dict, history: str,
                 program_md: str, hint: str = "") -> tuple[str, str]:
    """请求 LLM 提出一个策略变异。返回 (new_code, hypothesis)。"""

    # 格式化资产配置
    alloc = current_metrics.get("allocation", {})
    alloc_str = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(alloc.items(), key=lambda x: -x[1]))

    prompt = f"""{program_md}

---
## 最近实验历史（参考，避免重复失败方向）
{history}

---
## 当前策略代码
```python
{strategy_code}
```

## 当前回测结果（多年度 walk-forward 平均）
- 综合评分: {current_score:.4f}
- Sharpe Ratio: {current_metrics.get('sharpe', 0):.4f}
- 年化收益: {current_metrics.get('ann_return', 0):.2%}
- 最大回撤: {current_metrics.get('max_drawdown', 0):.2%}
- **超额收益（vs SPY B&H）: {current_metrics.get('excess_return', 0):.2%}**
- 现金占比: {current_metrics.get('cash_pct', 0):.2%}
- 年均切换次数: {current_metrics.get('n_switches', 0):.1f}
- 胜率: {current_metrics.get('win_rate', 0):.2%}
- 资产配置: {alloc_str}

注意：超额收益（vs SPY 买入持有）是最重要的指标！策略必须能跑赢大盘。

{hint}

请提出一个单一变异来改进策略。严格按以下格式输出：

HYPOTHESIS: [你的假设，一句话说明为什么这个改动会改善表现]

CODE:
```python
[完整的策略代码，包含所有 import 和 generate_signals 函数定义]
```"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.7,
    )

    text = response.choices[0].message.content
    if not text:
        raise ValueError("LLM returned empty response")

    # 解析 hypothesis
    hypothesis = ""
    if "HYPOTHESIS:" in text:
        hyp_part = text.split("CODE:")[0] if "CODE:" in text else text
        hypothesis = hyp_part.split("HYPOTHESIS:")[-1].strip()

    # 解析 code
    code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not code_blocks:
        raise ValueError("Failed to parse code block from LLM response")

    code = code_blocks[-1].strip()

    if "generate_signals" not in code:
        raise ValueError("Code missing generate_signals function")
    if "def generate_signals" not in code:
        raise ValueError("Code missing generate_signals function definition")

    return code, hypothesis
