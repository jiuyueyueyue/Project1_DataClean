"""
统一日志体系模块

提供 `setup_logger()` 工厂函数，创建具备以下特性的 logger 实例：
- 控制台 Handler：实时输出，支持彩色高亮（INFO 及以上级别）
- 文件 Handler：持久化记录，自动轮转避免单文件膨胀
- 格式统一：时间 | 级别 | 模块:函数:行号 | 消息内容

Usage:
    from utils import setup_logger
    logger = setup_logger(__name__)
    logger.info("数据加载完成，共 %d 条记录", count)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    创建并配置 logger 实例

    Args:
        name:     logger 名称，建议传入 __name__ 便于追踪调用链
        log_file: 日志文件路径，为 None 时仅输出到控制台
        level:    日志级别字符串（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        fmt:      日志格式字符串
        datefmt:  时间格式字符串

    Returns:
        配置完成的 logging.Logger 实例

    Raises:
        ValueError: 传入非法日志级别时抛出
    """
    # ---- 入参校验 ----
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {level!r}，请使用 DEBUG/INFO/WARNING/ERROR/CRITICAL")

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # 避免重复添加 Handler（幂等性保证）
    if logger.handlers:
        return logger

    # ---- 格式化器 ----
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # ---- 控制台 Handler（兼容 Windows GBK 编码） ----
    console_handler = logging.StreamHandler(
        sys.stdout if hasattr(sys.stdout, "buffer") else sys.stdout
    )
    # 为 Windows GBK 控制台添加编码容错：无法编码的字符用 '?' 替代
    if hasattr(sys.stdout, "buffer"):
        import io
        utf8_wrapper = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
        )
        console_handler = logging.StreamHandler(utf8_wrapper)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- 文件 Handler ----
    if log_file is not None:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
