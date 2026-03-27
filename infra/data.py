"""
数据管道 — 不可变层，agent 不能修改此文件。
支持多资产数据获取和对齐。
"""
import os
import pickle
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data_cache")


def get_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    """获取单个资产的 OHLCV 日线数据，带本地缓存。"""
    CACHE_DIR.mkdir(exist_ok=True)
    safe_symbol = symbol.replace("/", "_").replace(".", "_")
    cache_file = CACHE_DIR / f"{safe_symbol}_{start}_{end}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print(f"  下载 {symbol} ({start} ~ {end}) ...")
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({start} ~ {end})")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.dropna()

    df.to_parquet(cache_file)
    return df


def get_multi_prices(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    获取多个资产的价格数据，并对齐到公共日期索引。

    返回 dict: {symbol: DataFrame}，所有 DataFrame 共享相同的 DatetimeIndex。
    """
    raw = {}
    for symbol in symbols:
        raw[symbol] = get_prices(symbol, start, end)

    # 对齐到公共日期（取交集）
    common_idx = None
    for df in raw.values():
        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)

    common_idx = common_idx.sort_values()

    aligned = {}
    for symbol, df in raw.items():
        aligned[symbol] = df.reindex(common_idx).dropna()

    print(f"  {len(symbols)} 个资产, {len(common_idx)} 个公共交易日")
    return aligned


def save_multi_prices(prices: dict[str, pd.DataFrame], path: str) -> None:
    """将多资产价格数据保存为 pickle 文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(prices, f)


def load_multi_prices(path: str) -> dict[str, pd.DataFrame]:
    """加载多资产价格数据。"""
    with open(path, "rb") as f:
        return pickle.load(f)
