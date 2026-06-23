"""
文件安全读写工具模块

封装 pandas CSV 读写操作，统一处理异常、编码、目录创建，
避免各业务模块重复编写文件 I/O 安全代码。

Usage:
    from utils import safe_read_csv, safe_write_csv
    df = safe_read_csv("data/raw_data.csv", logger=logger)
    safe_write_csv(df, "data/processed/clean_data.csv", logger=logger)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd


def safe_read_csv(
    file_path: Path,
    encoding: str = "utf-8",
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    安全读取 CSV 文件

    执行入参校验、文件存在性检查、异常捕获，
    读取失败时返回空 DataFrame 并记录错误日志。

    Args:
        file_path: CSV 文件路径（Path 对象）
        encoding:  文件编码，默认 utf-8
        logger:    logger 实例，无则使用 root logger
        **kwargs:  透传给 pd.read_csv 的额外参数

    Returns:
        pd.DataFrame: 读取成功返回数据框，失败返回空 DataFrame
    """
    log = logger or logging.getLogger(__name__)

    # ---- 入参校验 ----
    if not isinstance(file_path, Path):
        log.warning("file_path 应为 Path 对象，尝试自动转换")
        file_path = Path(file_path)

    if not file_path.exists():
        log.error("文件不存在: %s", file_path)
        return pd.DataFrame()

    if not file_path.is_file():
        log.error("路径不是文件: %s", file_path)
        return pd.DataFrame()

    # ---- 读取 ----
    try:
        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        log.info("成功读取 %s，共 %d 行 × %d 列", file_path.name, len(df), len(df.columns))
        return df
    except UnicodeDecodeError:
        log.warning("编码 %s 读取失败，尝试 gbk 编码", encoding)
        try:
            df = pd.read_csv(file_path, encoding="gbk", **kwargs)
            log.info("以 gbk 编码成功读取 %s", file_path.name)
            return df
        except Exception as e:
            log.error("gbk 编码也读取失败: %s", e)
            return pd.DataFrame()
    except Exception as e:
        log.error("读取 CSV 异常: %s", e)
        return pd.DataFrame()


def safe_write_csv(
    df: pd.DataFrame,
    file_path: Path,
    encoding: str = "utf-8",
    index: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> bool:
    """
    安全写入 CSV 文件

    自动创建父目录，写入失败时记录错误日志并返回 False。

    Args:
        df:        待写入的 DataFrame
        file_path: 目标文件路径（Path 对象）
        encoding:  文件编码，默认 utf-8
        index:     是否写入行索引，默认 False
        logger:    logger 实例
        **kwargs:  透传给 pd.DataFrame.to_csv 的额外参数

    Returns:
        bool: 写入成功返回 True，失败返回 False
    """
    log = logger or logging.getLogger(__name__)

    # ---- 入参校验 ----
    if not isinstance(df, pd.DataFrame):
        log.error("入参 df 不是 pandas DataFrame，类型为 %s", type(df).__name__)
        return False

    if df.empty:
        log.warning("DataFrame 为空，将写入空文件 %s", file_path)
        # 空 DataFrame 仍允许写入（某些场景下有意为之）

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    # ---- 确保目录存在 ----
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("无法创建目录 %s: %s", file_path.parent, e)
        return False

    # ---- 写入 ----
    try:
        df.to_csv(file_path, index=index, encoding=encoding, **kwargs)
        log.info("成功写入 %s，共 %d 行 × %d 列", file_path.name, len(df), len(df.columns))
        return True
    except Exception as e:
        log.error("写入 CSV 异常: %s", e)
        return False
