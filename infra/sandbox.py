"""
策略执行沙盒 — 不可变层，agent 不能修改此文件。
在子进程中安全执行 LLM 生成的策略代码。
支持多资产轮动策略。
"""
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd


# 沙盒 runner 代码
_RUNNER_CODE = r'''
import pandas as pd
import numpy as np
import json
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

data_path = sys.argv[1]
strategy_path = sys.argv[2]

# 加载多资产价格数据
with open(data_path, "rb") as f:
    prices = pickle.load(f)

# 在全局作用域执行策略代码
with open(strategy_path) as f:
    exec(f.read(), globals())

# 调用策略函数
signals = generate_signals(prices)

# 验证输出
assert isinstance(signals, pd.Series), \
    f"generate_signals must return pd.Series, got {type(signals)}"

# 信号值必须是资产名（字符串）
unique_vals = set(str(v) for v in signals.dropna().unique())
valid_assets = set(prices.keys())
invalid = unique_vals - valid_assets
assert not invalid, \
    f"Invalid asset names in signals: {invalid}. Valid assets: {valid_assets}"

# 结构化 JSON 输出
output = {
    "signals": [str(v) for v in signals.values],
    "index": [d.isoformat() for d in signals.index],
}
print(json.dumps(output))
'''


def run_strategy(strategy_code: str, data_path: str,
                 timeout: int = 120) -> pd.Series:
    """
    在隔离子进程中执行策略代码，返回信号 Series。
    信号值为资产名字符串。
    """
    runner_fd, runner_path = tempfile.mkstemp(suffix="_runner.py")
    strategy_fd, strategy_path = tempfile.mkstemp(suffix="_strategy.py")

    try:
        with os.fdopen(runner_fd, "w") as f:
            f.write(_RUNNER_CODE)
        with os.fdopen(strategy_fd, "w") as f:
            f.write(strategy_code)

        result = subprocess.run(
            [sys.executable, runner_path, data_path, strategy_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            stderr = result.stderr[-1500:] if result.stderr else "Unknown error"
            raise RuntimeError(f"Strategy execution failed:\n{stderr}")

        if not result.stdout.strip():
            raise RuntimeError("Strategy produced no output")

        output = json.loads(result.stdout)
        signals = pd.Series(
            output["signals"],
            index=pd.to_datetime(output["index"]),
            name="signal",
        )
        return signals

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Strategy execution timed out ({timeout}s)")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse strategy output: {e}")
    finally:
        for p in (runner_path, strategy_path):
            try:
                os.unlink(p)
            except OSError:
                pass
