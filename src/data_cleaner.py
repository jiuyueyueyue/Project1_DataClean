"""
数据清洗管道模块

对原始合成数据进行标准化清洗，流程:
  1. 去除首尾空白字符
  2. 过滤异常短文本（长度阈值可配置）
  3. 文本去重（保留首次出现）
  4. 截断至目标条数（简历展示用，生产环境可关闭）

原文件: clean_pipeline.py
重构点:
  - 清洗参数硬编码 → config.settings 驱动
  - 无异常处理 → safe_read_csv / safe_write_csv + try-except
  - 无日志 → structured logging
  - 无类型注解 → 完整类型标注
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import settings
from utils.io_utils import safe_read_csv, safe_write_csv


def clean_pipeline(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    执行数据清洗管道

    清洗步骤:
    Step 1 - 空白清理: 去除 text 列首尾空格 → clean_text 列
    Step 2 - 长度过滤: 剔除 clean_text 字符数 < TEXT_MIN_LENGTH 的行
    Step 3 - 文本去重:  按 clean_text 去重，保留首次出现
    Step 4 - 数量截断:  取前 CLEAN_TARGET_COUNT 条
          (此步骤用于简历展示，实际生产中应通过质量阈值自然淘汰)

    Args:
        input_path:  输入 CSV 路径，默认 settings.RAW_DATA_PATH
        output_path: 输出 CSV 路径，默认 settings.CLEAN_DATA_PATH
        logger:      logger 实例

    Returns:
        清洗后的 DataFrame，列: [text, source, clean_text]

    Raises:
        FileNotFoundError: 输入文件不存在时抛出
        ValueError:        输入数据为空或清洗后无有效数据时抛出
    """
    log = logger or logging.getLogger(__name__)
    input_path = input_path or settings.RAW_DATA_PATH
    output_path = output_path or settings.CLEAN_DATA_PATH

    # ========================================================================
    # Step 0: 读取原始数据
    # ========================================================================
    log.info("=" * 50)
    log.info("开始数据清洗管道")
    log.info("=" * 50)

    df = safe_read_csv(input_path, encoding=settings.CSV_ENCODING, logger=log)
    if df.empty:
        raise ValueError(f"原始数据文件为空或读取失败: {input_path}")

    if "text" not in df.columns:
        raise ValueError(f"原始数据缺少 'text' 列，可用列: {list(df.columns)}")

    original_count = len(df)
    log.info("原始数据总量: %d", original_count)

    # ========================================================================
    # Step 1: 空白清理
    # ========================================================================
    df["clean_text"] = df["text"].astype(str).str.strip()
    log.info("Step 1 完成: 去除首尾空格")

    # ========================================================================
    # Step 2: 长度过滤
    # ========================================================================
    before_filter = len(df)
    df = df[df["clean_text"].str.len() >= settings.TEXT_MIN_LENGTH]
    filtered_count = before_filter - len(df)
    log.info(
        "Step 2 完成: 过滤长度 < %d 的文本，剔除 %d 条",
        settings.TEXT_MIN_LENGTH,
        filtered_count,
    )

    # ========================================================================
    # Step 3: 文本去重
    # ========================================================================
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean_text"], keep="first")
    dedup_count = before_dedup - len(df)
    log.info("Step 3 完成: 文本去重，剔除 %d 条重复", dedup_count)

    # ========================================================================
    # Step 4: 数量截断（简历展示用）
    # ========================================================================
    if len(df) > settings.CLEAN_TARGET_COUNT:
        log.warning(
            "当前数据量 %d > 目标 %d，将截断至目标条数（此行为仅用于演示）",
            len(df),
            settings.CLEAN_TARGET_COUNT,
        )
        df = df.head(settings.CLEAN_TARGET_COUNT)

    clean_count = len(df)
    if clean_count == 0:
        raise ValueError("清洗后无有效数据，请检查清洗参数或原始数据质量")

    # ========================================================================
    # 统计与保存
    # ========================================================================
    delete_rate = round((original_count - clean_count) / original_count * 100, 1)
    log.info("清洗后有效数据量: %d", clean_count)
    log.info("剔除低质量数据占比: %.1f%%", delete_rate)

    success = safe_write_csv(df, output_path, encoding=settings.CSV_ENCODING, logger=log)
    if not success:
        log.error("写入清洗数据失败: %s", output_path)
    else:
        log.info("清洗完成，已生成 %s", output_path)

    return df


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
    logger.info("独立运行: 数据清洗模块")
    logger.info("=" * 60)

    try:
        clean_df = clean_pipeline(logger=logger)
        logger.info("清洗结果预览:\n%s", clean_df.head())
    except Exception as e:
        logger.exception("数据清洗失败: %s", e)
