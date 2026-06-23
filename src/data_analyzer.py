"""
标签分布统计与可视化模块

从 Label Studio 标注数据中提取多标签，统计各类标签出现频次，
并生成专业柱状分布图。

原文件: data_analysis.py
重构点:
  - eval() 安全漏洞 → utils.label_parser 安全解析
  - 硬编码路径 → config.settings 驱动
  - 无异常处理 → try-except 全流程覆盖
  - 裸 print → structured logging
  - 无类型注解 → 完整类型标注
  - 重复解析逻辑 → 复用 parse_label_studio_annotations
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from config import settings
from utils.io_utils import safe_read_csv
from utils.label_parser import parse_label_studio_annotations


def analyze_label_distribution(
    input_path: Optional[Path] = None,
    output_plot_path: Optional[Path] = None,
    show_plot: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    """
    统计标注标签分布并生成可视化柱状图

    Args:
        input_path:      标注数据 CSV 路径，默认 settings.LABELED_DATA_PATH
        output_plot_path: 图表输出路径，默认 settings.LABEL_PLOT_PATH
        show_plot:        是否弹出图表窗口，默认 settings.SHOW_PLOT
        logger:           logger 实例

    Returns:
        Dict[str, int]: 标签 → 出现次数的频次字典

    Raises:
        FileNotFoundError: 标注文件不存在时抛出
        ValueError:        未解析到有效标签时抛出
    """
    log = logger or logging.getLogger(__name__)
    input_path = input_path or settings.LABELED_DATA_PATH
    output_plot_path = output_plot_path or settings.LABEL_PLOT_PATH
    show_plot = show_plot if show_plot is not None else settings.SHOW_PLOT

    # ========================================================================
    # Step 1: 读取标注数据
    # ========================================================================
    log.info("=" * 50)
    log.info("开始标签分布分析")
    log.info("=" * 50)

    df = safe_read_csv(input_path, encoding=settings.CSV_ENCODING, logger=log)
    if df.empty:
        raise ValueError(f"标注数据文件为空或读取失败: {input_path}")

    log.info("标注数据加载完成，共 %d 行", len(df))

    # ========================================================================
    # Step 2: 安全解析标注列
    # ========================================================================
    _, labels_list = parse_label_studio_annotations(df, logger=log)

    # 展平嵌套标签列表
    all_labels: List[str] = [label for sublist in labels_list for label in sublist]

    if not all_labels:
        raise ValueError(
            "未解析到任何有效标签，请检查标注文件 %s 中的标注列格式" % input_path
        )

    # ========================================================================
    # Step 3: 频次统计
    # ========================================================================
    counter = Counter(all_labels)
    log.info("共解析到 %d 个标签实例，%d 种不同标签", len(all_labels), len(counter))

    # 日志输出统计结果
    log.info("==== 各类标签统计数量 ====")
    for lab, cnt in counter.most_common():
        log.info("  %-12s : %d", lab, cnt)

    # ========================================================================
    # Step 4: 可视化
    # ========================================================================
    try:
        # 修复 matplotlib 中文显示
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=settings.PLOT_FIGSIZE)

        labels_sorted = [item[0] for item in counter.most_common()]
        values_sorted = [item[1] for item in counter.most_common()]

        bars = ax.bar(labels_sorted, values_sorted, color=settings.PLOT_COLOR)
        ax.set_title(settings.PLOT_TITLE, fontsize=14, fontweight="bold")
        ax.set_ylabel(settings.PLOT_YLABEL, fontsize=12)
        ax.set_xlabel("标签类别", fontsize=12)
        ax.tick_params(axis="x", rotation=45)

        # 在柱顶标注数值
        for bar_obj, val in zip(bars, values_sorted):
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                bar_obj.get_height() + 0.3,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        # 保存图表
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=settings.PLOT_DPI)
        log.info("标签分布图已保存至 %s (DPI=%d)", output_plot_path, settings.PLOT_DPI)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        log.error("生成标签分布图时发生异常: %s", e)
        # 图表生成失败不应阻断主流程

    return dict(counter)


# ============================================================================
# 模块独立运行入口
# ============================================================================
if __name__ == "__main__":
    from utils.logger import setup_logger

    logger = setup_logger(
        "data_analyzer",
        log_file=settings.LOG_FILE_PATH,
        level=settings.LOG_LEVEL,
    )
    logger.info("=" * 60)
    logger.info("独立运行: 数据分析模块")
    logger.info("=" * 60)

    try:
        result = analyze_label_distribution(logger=logger)
        if result:
            logger.info("分析完成，共 %d 种标签", len(result))
    except Exception as e:
        logger.exception("数据分析失败: %s", e)
