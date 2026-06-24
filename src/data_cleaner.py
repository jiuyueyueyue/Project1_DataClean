"""
电商语料清洗管道模块

对四类业务数据进行标准化清洗，核心清洗逻辑保持不变:
  1. 去除首尾空白字符
  2. 过滤异常短文本（按数据类型差异化阈值）
  3. 文本去重（保留首次出现）
  4. 统一格式输出

新增: clean_by_type() 按数据类型分流，对不同列应用差异化清洗规则，
     适配 FAQ/商品/规则/对话四类数据的不同 Schema。

原文件: clean_pipeline.py
改造点:
  - 通用清洗逻辑保留，新增按数据类型的差异化过滤阈值
  - 支持四类电商数据的列 Schema 自适应
  - settings 引用从智能家居常量切换为电商业务常量
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import settings
from utils.io_utils import safe_read_csv, safe_write_csv


# ============================================================================
# 核心清洗逻辑（保留不变）
# ============================================================================
def _apply_universal_cleaning(
    df: pd.DataFrame,
    text_column: str,
    min_length: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    通用文本清洗四步骤（核心逻辑不变）

    Step 1 - 空白清理: 去除指定列首尾空格 → clean_text 列
    Step 2 - 长度过滤: 剔除 clean_text < min_length 的行
    Step 3 - 文本去重:  按 clean_text 去重，保留首次出现
    Step 4 - 空值过滤:  剔除 clean_text 为空的残留行

    Args:
        df:         待清洗 DataFrame
        text_column: 需清洗的文本列名
        min_length:  最小文本长度阈值
        logger:      logger 实例

    Returns:
        清洗后的 DataFrame（会新增 clean_text 列）
    """
    before_count = len(df)

    # Step 1: 空白清理
    df["clean_text"] = df[text_column].astype(str).str.strip()

    # Step 2: 长度过滤
    df = df[df["clean_text"].str.len() >= min_length]
    after_len_filter = len(df)

    # Step 3: 文本去重
    df = df.drop_duplicates(subset=["clean_text"], keep="first")
    after_dedup = len(df)

    # Step 4: 空值过滤
    df = df[df["clean_text"].notna() & (df["clean_text"] != "")]
    final_count = len(df)

    logger.debug(
        "清洗: %d → (长度) %d → (去重) %d → (空值) %d, 剔除率 %.1f%%",
        before_count, after_len_filter, after_dedup, final_count,
        (before_count - final_count) / max(before_count, 1) * 100,
    )

    return df.reset_index(drop=True)


# ============================================================================
# 按数据类型分流清洗
# ============================================================================
def clean_by_type(
    data_type: str,
    input_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    按数据类型执行差异化清洗

    不同数据类型使用不同的文本列和长度阈值:
      - faq:        清洗 question + answer 列，过滤过短问答
      - product:    清洗 description 列，过滤空描述
      - aftersales: 清洗 rule_content 列，过滤碎片规则
      - chatlog:    清洗 message 列，过滤无效消息

    Args:
        data_type:   数据类型标识 (faq/product/aftersales/chatlog)
        input_path:  输入 CSV 路径
        output_path: 清洗后输出 CSV 路径
        logger:      logger 实例

    Returns:
        清洗后的 DataFrame

    Raises:
        ValueError: 输入文件为空或缺少关键列
    """
    type_config = settings.CLEAN_BY_TYPE.get(data_type, {})
    if not type_config:
        logger.warning("未知数据类型 '%s'，使用默认清洗参数", data_type)

    df = safe_read_csv(input_path, encoding=settings.CSV_ENCODING, logger=logger)
    if df.empty:
        raise ValueError(f"[{data_type}] 数据文件为空或读取失败: {input_path}")

    original_count = len(df)
    logger.info("[%s] 原始数据 %d 条", data_type, original_count)

    if data_type == "faq":
        # FAQ: 清洗 question 和 answer
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError(f"FAQ 数据缺少 question/answer 列: {list(df.columns)}")
        min_q = type_config.get("min_question_len", 4)
        min_a = type_config.get("min_answer_len", 10)
        # 对 question 和 answer 拼接后统一清洗
        df["_text_for_clean"] = df["question"].astype(str) + " " + df["answer"].astype(str)
        df = _apply_universal_cleaning(df, "_text_for_clean", min_q, logger)
        df = df.drop(columns=["_text_for_clean"])
        # 额外过滤 answer 过短的行
        df = df[df["answer"].astype(str).str.len() >= min_a]

    elif data_type == "product":
        # 商品资料: 清洗 description
        if "description" not in df.columns:
            raise ValueError(f"商品数据缺少 description 列: {list(df.columns)}")
        min_desc = type_config.get("min_desc_len", 20)
        df = _apply_universal_cleaning(df, "description", min_desc, logger)

    elif data_type == "aftersales":
        # 售后规则: 清洗 rule_content
        if "rule_content" not in df.columns:
            raise ValueError(f"售后规则数据缺少 rule_content 列: {list(df.columns)}")
        min_content = type_config.get("min_content_len", 30)
        df = _apply_universal_cleaning(df, "rule_content", min_content, logger)

    elif data_type == "chatlog":
        # 对话日志: 清洗 message
        if "message" not in df.columns:
            raise ValueError(f"对话日志缺少 message 列: {list(df.columns)}")
        min_msg = type_config.get("min_message_len", 2)
        max_msg = type_config.get("max_message_len", 500)
        df = _apply_universal_cleaning(df, "message", min_msg, logger)
        # 对话日志额外过滤过长消息（可能是异常日志混杂）
        df = df[df["clean_text"].str.len() <= max_msg]

    clean_count = len(df)
    if clean_count == 0:
        raise ValueError(f"[{data_type}] 清洗后无有效数据，请检查清洗参数")

    delete_rate = round((original_count - clean_count) / original_count * 100, 1)
    logger.info("[%s] 清洗后 %d 条, 剔除率 %.1f%%", data_type, clean_count, delete_rate)

    safe_write_csv(df, output_path, encoding=settings.CSV_ENCODING, logger=logger)
    return df


def clean_corpus(
    data_paths: Optional[Dict[str, Path]] = None,
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """
    对全部四类业务数据执行清洗，输出统一语料

    Args:
        data_paths:  各类型数据文件路径字典，为空时使用 settings 默认路径
        output_path: 合并后输出路径，默认 settings.CLEAN_DATA_PATH
        logger:      logger 实例

    Returns:
        {"faq": df, "product": df, "aftersales": df, "chatlog": df}
    """
    log = logger or logging.getLogger(__name__)
    output_path = output_path or settings.CLEAN_DATA_PATH

    log.info("=" * 60)
    log.info("  电商语料清洗管道")
    log.info("=" * 60)

    if data_paths is None:
        data_paths = {
            "faq":        settings.FAQ_DATA_PATH,
            "product":    settings.PRODUCT_DATA_PATH,
            "aftersales": settings.AFTERSALES_DATA_PATH,
            "chatlog":    settings.CHATLOG_DATA_PATH,
        }

    results: Dict[str, pd.DataFrame] = {}
    all_cleaned: List[pd.DataFrame] = []

    for data_type, input_path in data_paths.items():
        if not input_path.exists():
            log.warning("[%s] 数据文件不存在，跳过: %s", data_type, input_path)
            continue
        # 清洗后输出到 processed 目录
        type_output = settings.PROCESSED_DIR / f"clean_{data_type}.csv"
        try:
            df_clean = clean_by_type(data_type, input_path, type_output, log)
            results[data_type] = df_clean
            all_cleaned.append(df_clean)
        except Exception as e:
            log.exception("[%s] 清洗失败: %s", data_type, e)
            results[data_type] = pd.DataFrame()

    # 合并输出统一语料
    if all_cleaned:
        merged = pd.concat(all_cleaned, ignore_index=True)
        safe_write_csv(merged, output_path, encoding=settings.CSV_ENCODING, logger=log)
        total = sum(len(r) for r in results.values())
        log.info("清洗全流程完成，总计有效语料 %d 条 → %s", total, output_path)
    else:
        log.warning("所有数据类型清洗后均为空，请检查源数据")

    return results


# ============================================================================
# 模块独立运行入口
# ============================================================================
if __name__ == "__main__":
    from utils.logger import setup_logger

    logger = setup_logger(
        "data_cleaner",
        log_file=settings.LOG_FILE_PATH,
        level=settings.LOG_LEVEL,
    )
    logger.info("=" * 60)
    logger.info("独立运行: 电商语料清洗模块")
    logger.info("=" * 60)

    try:
        result = clean_corpus(logger=logger)
        for dtype, ddf in result.items():
            if not ddf.empty:
                logger.info("[%s] 清洗结果 %d 条", dtype, len(ddf))
    except Exception as e:
        logger.exception("数据清洗失败: %s", e)
