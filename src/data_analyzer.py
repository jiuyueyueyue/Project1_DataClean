"""
电商语料数据统计与可视化模块

对清洗后的语料数据进行多维度统计分析:
  1. 四类数据占比分布
  2. 文本长度分布（箱线图 + 直方图）
  3. 各品类/场景下的样本分布
  4. 数据质量摘要报告

原文件: data_analysis.py（标签分布分析）
改造点:
  - 分析目标从"标签频次"切换为"语料统计"
  - 新增数据类型占比和文本长度分布可视化
  - 复用 utils 安全 I/O
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from config import settings
from utils.io_utils import safe_read_csv


def analyze_corpus(
    data_paths: Optional[Dict[str, Path]] = None,
    output_plot_path: Optional[Path] = None,
    show_plot: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, int]]:
    """
    电商语料数据统计分析

    读取清洗后的四类数据，输出:
      - 数据类型占比饼图（左上）
      - 文本长度分布直方图（右上）
      - FAQ 品类分布柱状图（左下）
      - 对话日志场景分布柱状图（右下）

    Args:
        data_paths:      各数据类型清洗后文件路径字典
        output_plot_path: 图表输出路径
        show_plot:        是否弹出图表窗口
        logger:           logger 实例

    Returns:
        统计结果字典: {data_type: {category: count}}
    """
    log = logger or logging.getLogger(__name__)
    output_plot_path = output_plot_path or settings.CORPUS_PLOT_PATH
    show_plot = show_plot if show_plot is not None else settings.SHOW_PLOT

    log.info("=" * 60)
    log.info("  电商语料统计分析")
    log.info("=" * 60)

    if data_paths is None:
        data_paths = {
            "FAQ问答":    settings.PROCESSED_DIR / "clean_faq.csv",
            "商品资料":    settings.PROCESSED_DIR / "clean_product.csv",
            "售后规则":    settings.PROCESSED_DIR / "clean_aftersales.csv",
            "对话日志":    settings.PROCESSED_DIR / "clean_chatlog.csv",
        }

    # ========================================================================
    # Step 1: 加载所有数据
    # ========================================================================
    datasets: Dict[str, pd.DataFrame] = {}
    for label, path in data_paths.items():
        if path.exists():
            df = safe_read_csv(path, encoding=settings.CSV_ENCODING, logger=log)
            if not df.empty:
                datasets[label] = df
                log.info("加载 [%s]: %d 条", label, len(df))
        else:
            log.warning("[%s] 文件不存在: %s", label, path)

    if not datasets:
        raise ValueError("未找到任何清洗后的数据文件，请先运行数据生成和清洗")

    # ========================================================================
    # Step 2: 数据量占比统计
    # ========================================================================
    counts = {label: len(df) for label, df in datasets.items()}
    total_samples = sum(counts.values())
    log.info("总语料量: %d 条", total_samples)
    for label, cnt in counts.items():
        log.info("  %s: %d 条 (%.1f%%)", label, cnt, cnt / total_samples * 100)

    # ========================================================================
    # Step 3: 各数据类型的子类别分布
    # ========================================================================
    distribution: Dict[str, Dict[str, int]] = {}

    for label, df in datasets.items():
        # 根据数据类型选择分类列
        if "category" in df.columns:
            col = "category"
        elif "rule_category" in df.columns:
            col = "rule_category"
        elif "intent_label" in df.columns:
            col = "intent_label"
        else:
            continue

        # 过滤空值
        valid = df[df[col].notna() & (df[col] != "")]
        if valid.empty:
            continue
        distribution[label] = dict(Counter(valid[col].tolist()))
        log.info("[%s] 子类分布: %s", label,
                 {k: v for k, v in sorted(distribution[label].items(), key=lambda x: -x[1])[:5]})

    # ========================================================================
    # Step 4: 文本长度统计
    # ========================================================================
    length_stats: Dict[str, Dict[str, float]] = {}
    for label, df in datasets.items():
        # 选择文本列
        if "clean_text" in df.columns:
            text_col = "clean_text"
        elif "description" in df.columns:
            text_col = "description"
        elif "answer" in df.columns:
            text_col = "answer"
        elif "message" in df.columns:
            text_col = "message"
        else:
            continue

        lengths = df[text_col].astype(str).str.len()
        length_stats[label] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "min": float(lengths.min()),
            "max": float(lengths.max()),
            "std": float(lengths.std()),
        }
        log.info("[%s] 文本长度: 均值=%.0f, 中位数=%.0f, 范围=[%.0f, %.0f]",
                 label, length_stats[label]["mean"], length_stats[label]["median"],
                 length_stats[label]["min"], length_stats[label]["max"])

    # ========================================================================
    # Step 5: 可视化
    # ========================================================================
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 2, figsize=settings.PLOT_FIGSIZE)
        colors = settings.PLOT_COLOR_PALETTE

        # 子图1: 数据类型占比饼图
        ax1 = axes[0, 0]
        labels_pie = list(counts.keys())
        sizes_pie = list(counts.values())
        wedges, texts, autotexts = ax1.pie(
            sizes_pie, labels=labels_pie, autopct="%1.1f%%",
            colors=colors[:len(labels_pie)], startangle=90,
        )
        ax1.set_title("四类业务数据占比", fontsize=13, fontweight="bold")

        # 子图2: 文本长度分布直方图
        ax2 = axes[0, 1]
        for idx, (label, df) in enumerate(datasets.items()):
            text_col = next((c for c in ["clean_text", "description", "answer", "message"] if c in df.columns), None)
            if text_col:
                lengths = df[text_col].astype(str).str.len()
                # 截断极端值以改善显示
                clipped = lengths.clip(upper=lengths.quantile(0.95))
                ax2.hist(clipped, bins=30, alpha=0.6, label=label, color=colors[idx % len(colors)])
        ax2.set_title("文本长度分布（95分位数截断）", fontsize=13, fontweight="bold")
        ax2.set_xlabel("字符数")
        ax2.set_ylabel("样本数")
        ax2.legend(fontsize=8)

        # 子图3: FAQ 品类分布
        ax3 = axes[1, 0]
        if "FAQ问答" in distribution:
            faq_dist = distribution["FAQ问答"]
            faq_sorted = sorted(faq_dist.items(), key=lambda x: -x[1])
            faq_labels, faq_vals = zip(*faq_sorted) if faq_sorted else ([], [])
            ax3.barh(list(faq_labels), list(faq_vals), color=colors[0])
            ax3.set_title("FAQ 品类分布", fontsize=13, fontweight="bold")
            ax3.set_xlabel("问答对数")
        else:
            ax3.text(0.5, 0.5, "暂无FAQ数据", ha="center", va="center", transform=ax3.transAxes)

        # 子图4: 对话日志场景分布
        ax4 = axes[1, 1]
        if "对话日志" in distribution:
            chat_dist = distribution["对话日志"]
            chat_sorted = sorted(chat_dist.items(), key=lambda x: -x[1])
            chat_labels, chat_vals = zip(*chat_sorted) if chat_sorted else ([], [])
            ax4.barh(list(chat_labels), list(chat_vals), color=colors[3])
            ax4.set_title("客服对话场景分布", fontsize=13, fontweight="bold")
            ax4.set_xlabel("对话轮数")
        else:
            ax4.text(0.5, 0.5, "暂无对话日志数据", ha="center", va="center", transform=ax4.transAxes)

        plt.tight_layout()
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=settings.PLOT_DPI)
        log.info("语料统计图已保存至 %s", output_plot_path)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        log.error("生成统计图时发生异常: %s", e)

    return distribution


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
    logger.info("独立运行: 电商语料分析模块")
    logger.info("=" * 60)

    try:
        result = analyze_corpus(logger=logger)
        logger.info("统计完成，共 %d 类数据", len(result))
    except Exception as e:
        logger.exception("语料分析失败: %s", e)
