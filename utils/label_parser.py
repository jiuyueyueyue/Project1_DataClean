"""
Label Studio 标注数据安全解析模块

原代码使用 eval() 解析 sentiment 列中的标注字典，存在任意代码执行风险。
本模块使用 ast.literal_eval 安全解析，同时增加：
- 入参校验（类型检查、空值处理）
- 多层解析容错（str → dict → list 多路径尝试）
- "choices" 字段健壮提取
- 解析统计日志

Usage:
    from utils import parse_label_studio_annotations
    texts, labels = parse_label_studio_annotations(df, logger=logger)
"""

import ast
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _safe_parse_cell(
    cell_value: Any,
    logger: logging.Logger,
) -> Optional[List[str]]:
    """
    安全解析单个标注单元格

    支持以下格式：
    1. Python dict 字符串  → ast.literal_eval 解析
    2. JSON 字符串        → json.loads 解析
    3. 已解析的 dict/list → 直接提取

    Args:
        cell_value: 单元格原始值（str / dict / list / NaN）
        logger:     logger 实例

    Returns:
        解析成功返回标签列表，失败返回 None
    """
    # ---- 空值过滤 ----
    if cell_value is None:
        return None
    if isinstance(cell_value, float) and pd.isna(cell_value):
        return None
    if isinstance(cell_value, str) and not cell_value.strip():
        return None

    # ---- 已是 dict 或 list ----
    if isinstance(cell_value, dict):
        return cell_value.get("choices", [])
    if isinstance(cell_value, list):
        return cell_value

    # ---- 字符串解析 ----
    if isinstance(cell_value, str):
        # 路径1: ast.literal_eval（安全，不执行任意代码）
        try:
            parsed = ast.literal_eval(cell_value)
            if isinstance(parsed, dict):
                return parsed.get("choices", [])
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

        # 路径2: json.loads（处理标准 JSON）
        try:
            parsed = json.loads(cell_value)
            if isinstance(parsed, dict):
                return parsed.get("choices", [])
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        logger.debug("无法解析标注单元格: %s", cell_value[:80])

    return None


def parse_label_studio_annotations(
    df: pd.DataFrame,
    text_column: str = "text",
    annotation_column: str = "sentiment",
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[List[str]]]:
    """
    从 Label Studio 导出的 DataFrame 中安全提取文本和标签

    遍历 DataFrame 的每一行，安全解析标注列中的 JSON/Python-dict 字符串，
    提取 choices 多标签列表。

    Args:
        df:                Label Studio 导出数据的 DataFrame
        text_column:       文本列名，默认 "text"
        annotation_column: 标注列名，默认 "sentiment"
        logger:            logger 实例

    Returns:
        (texts, labels_list) 二元组：
        - texts:        文本列表 List[str]
        - labels_list:  标签列表 List[List[str]]，每条文本对应一个标签子列表

    Raises:
        ValueError: text_column 或 annotation_column 不存在时抛出
    """
    log = logger or logging.getLogger(__name__)

    # ---- 入参校验 ----
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df 参数必须为 pandas DataFrame，实际类型: {type(df).__name__}")
    if df.empty:
        log.warning("传入的 DataFrame 为空，返回空列表")
        return [], []

    for col_name, col_desc in [(text_column, "文本"), (annotation_column, "标注")]:
        if col_name not in df.columns:
            raise ValueError(
                f"{col_desc}列 '{col_name}' 不存在。可用列: {list(df.columns)}"
            )

    # ---- 逐行解析 ----
    texts: List[str] = []
    labels_list: List[List[str]] = []
    parse_success_count = 0
    parse_fail_count = 0

    for _idx, row in df.iterrows():
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        ann = row[annotation_column]

        labels = _safe_parse_cell(ann, log)

        if labels is not None:
            texts.append(text)
            labels_list.append(labels)
            parse_success_count += 1
        else:
            parse_fail_count += 1

    # ---- 统计日志 ----
    log.info(
        "标注解析完成: 成功 %d 条, 跳过 %d 条, 提取到 %d 个不同标签",
        parse_success_count,
        parse_fail_count,
        len(set(lbl for sublist in labels_list for lbl in sublist)),
    )

    return texts, labels_list
