"""
合成数据生成模块

基于设备 × 问题的笛卡尔积循环生成智能家居问答文本，
每条记录格式为 "{设备}{问题}"，来源统一标记为论坛爬虫采集。

原文件: crawl_data.py
重构点:
  - 硬编码设备/问题列表 → config.settings 集中管理
  - 裸 print → structured logging
  - 无函数封装 → generate_raw_data() 可复用函数
  - 无异常处理 → safe_write_csv + try-except
  - 无类型注解 → 完整类型标注
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import settings
from utils.io_utils import safe_write_csv


def generate_raw_data(
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    合成生成原始智能家居问答数据

    使用笛卡尔积方式生成数据：
    for device in DEVICE_LIST:
        for question in QUESTION_LIST:
            重复填充直到总条数达到 GENERATE_TOTAL_COUNT

    Args:
        output_path: 输出 CSV 路径，默认使用 settings.RAW_DATA_PATH
        logger:      logger 实例

    Returns:
        生成的原始数据 DataFrame，列: [text, source]

    Raises:
        ValueError: 数据生成参数配置不合理时抛出
    """
    log = logger or logging.getLogger(__name__)
    output_path = output_path or settings.RAW_DATA_PATH

    # ---- 入参校验 ----
    if not settings.DEVICE_LIST:
        raise ValueError("DEVICE_LIST 不能为空，请在 config/settings.py 中配置")
    if not settings.QUESTION_LIST:
        raise ValueError("QUESTION_LIST 不能为空，请在 config/settings.py 中配置")
    if settings.GENERATE_TOTAL_COUNT <= 0:
        raise ValueError(
            f"GENERATE_TOTAL_COUNT 必须 > 0，当前值: {settings.GENERATE_TOTAL_COUNT}"
        )

    # ---- 构建文本组合 ----
    device_count = len(settings.DEVICE_LIST)
    question_count = len(settings.QUESTION_LIST)
    combo_count = device_count * question_count  # 设备-问题组合数

    per_combo_repeat = settings.GENERATE_TOTAL_COUNT // combo_count
    remainder = settings.GENERATE_TOTAL_COUNT % combo_count

    log.info(
        "开始生成数据: %d 设备 × %d 问题 = %d 组合, 每组重复 %d 次, 余数 %d",
        device_count,
        question_count,
        combo_count,
        per_combo_repeat,
        remainder,
    )

    # ---- 循环生成 ----
    raw_rows: List[List[str]] = []

    for idx, device in enumerate(settings.DEVICE_LIST):
        for jdx, question in enumerate(settings.QUESTION_LIST):
            # 计算该组合应生成的条数（均匀分配 + 余数分配到前几个组合）
            combo_index = idx * question_count + jdx
            count = per_combo_repeat + (1 if combo_index < remainder else 0)

            content = f"{device}{question}"
            for _ in range(count):
                raw_rows.append([content, settings.DATA_SOURCE_LABEL])

    # ---- 构建 DataFrame ----
    df = pd.DataFrame(raw_rows, columns=["text", "source"])
    log.info("数据生成完成，实际条数: %d", len(df))

    # ---- 安全写入 ----
    success = safe_write_csv(df, output_path, logger=log)
    if not success:
        log.error("写入原始数据文件失败: %s", output_path)
    else:
        log.info("原始数据已保存至 %s", output_path)

    return df


# ============================================================================
# 模块独立运行入口
# ============================================================================
if __name__ == "__main__":
    from utils.logger import setup_logger

    logger = setup_logger(
        "data_generator",
        log_file=settings.LOG_FILE_PATH,
        level=settings.LOG_LEVEL,
    )
    logger.info("=" * 60)
    logger.info("独立运行: 数据生成模块")
    logger.info("=" * 60)

    try:
        df_result = generate_raw_data(logger=logger)
        logger.info("生成数据预览:\n%s", df_result.head())
    except Exception as e:
        logger.exception("数据生成失败: %s", e)
