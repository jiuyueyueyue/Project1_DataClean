#!/usr/bin/env python3
"""
Project1_DataClean —— 智能家居问答多标签分类项目统一启动入口

支持三种运行模式:
  1. 全流程:   python main.py --pipeline all
  2. 分步执行: python main.py --pipeline generate   (仅数据生成)
              python main.py --pipeline clean       (仅数据清洗)
              python main.py --pipeline analyze     (仅数据分析)
              python main.py --pipeline train       (仅模型训练)
  3. 交互推理: python main.py --infer "空调无法连接WiFi"

Usage:
    python main.py --help      查看完整帮助
    python main.py --pipeline all --log-level DEBUG   开启调试日志
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中（支持任意目录执行）
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from utils.logger import setup_logger


def _build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="Project1_DataClean",
        description="智能家居问答多标签分类 —— 全流程一键执行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --pipeline all                    # 一键执行完整流程
  python main.py --pipeline generate               # 仅生成合成数据
  python main.py --pipeline clean                  # 仅执行数据清洗
  python main.py --pipeline analyze                # 仅分析标签分布
  python main.py --pipeline train                  # 仅训练模型
  python main.py --infer "空调开机报错"             # 交互推理测试
  python main.py --pipeline all --log-level DEBUG  # 调试模式
        """,
    )

    # ---- 主命令 ----
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["all", "generate", "clean", "analyze", "train"],
        default=None,
        help="执行指定阶段: all=全流程, generate=数据生成, clean=数据清洗, "
             "analyze=数据分析, train=模型训练",
    )
    parser.add_argument(
        "--infer",
        type=str,
        default=None,
        metavar="TEXT",
        help="交互推理模式: 输入文本，输出预测标签",
    )

    # ---- 全局选项 ----
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=settings.LOG_LEVEL,
        help="日志级别 (默认: %(default)s)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="数据分析时不弹出图表窗口",
    )

    return parser


def run_generate(logger) -> bool:
    """执行数据生成阶段"""
    from src.data_generator import generate_raw_data
    logger.info(">>> 阶段 1/4: 数据生成")
    try:
        generate_raw_data(logger=logger)
        logger.info("[OK] 数据生成完成")
        return True
    except Exception as e:
        logger.exception("[FAIL] 数据生成失败: %s", e)
        return False


def run_clean(logger) -> bool:
    """执行数据清洗阶段"""
    from src.data_cleaner import clean_pipeline
    logger.info(">>> 阶段 2/4: 数据清洗")
    try:
        clean_pipeline(logger=logger)
        logger.info("[OK] 数据清洗完成")
        return True
    except Exception as e:
        logger.exception("[FAIL] 数据清洗失败: %s", e)
        return False


def run_analyze(logger, show_plot: bool = True) -> bool:
    """执行数据分析阶段"""
    from src.data_analyzer import analyze_label_distribution
    logger.info(">>> 阶段 3/4: 数据分析")
    try:
        analyze_label_distribution(show_plot=show_plot, logger=logger)
        logger.info("[OK] 数据分析完成")
        return True
    except Exception as e:
        logger.exception("[FAIL] 数据分析失败: %s", e)
        return False


def run_train(logger) -> bool:
    """执行模型训练阶段"""
    from src.model_trainer import train_multilabel_classifier, predict_sentence
    logger.info(">>> 阶段 4/4: 模型训练")
    try:
        result = train_multilabel_classifier(logger=logger)

        # 训练完成后自动进行一次演示推理
        demo_text = "空调开机报错无法启动"
        logger.info("--- 演示推理 ---")
        predicted = predict_sentence(
            demo_text,
            result["model"],
            result["vectorizer"],
            result["mlb"],
            logger=logger,
        )
        logger.info("输入: %s", demo_text)
        logger.info("预测标签: %s", predicted)
        logger.info("[OK] 模型训练完成")
        return True
    except Exception as e:
        logger.exception("[FAIL] 模型训练失败: %s", e)
        return False


def run_full_pipeline(logger, show_plot: bool = True) -> bool:
    """执行全流程: 数据生成 → 清洗 → 分析 → 模型训练"""
    logger.info("=" * 60)
    logger.info("  智能家居问答多标签分类 —— 全流程启动")
    logger.info("=" * 60)

    stages = [
        ("数据生成", lambda: run_generate(logger)),
        ("数据清洗", lambda: run_clean(logger)),
        ("数据分析", lambda: run_analyze(logger, show_plot=show_plot)),
        ("模型训练", lambda: run_train(logger)),
    ]

    success_count = 0
    for stage_name, stage_fn in stages:
        if not stage_fn():
            logger.error("流水线在「%s」阶段失败，终止后续执行", stage_name)
            break
        success_count += 1

    logger.info("=" * 60)
    logger.info("全流程结束: %d/%d 阶段成功", success_count, len(stages))
    logger.info("=" * 60)

    return success_count == len(stages)


def run_inference(text: str, logger) -> bool:
    """
    交互推理模式: 加载已训练的模型并对输入文本进行预测

    注意: 此模式需要已存在的 labeled_data.csv 用于训练模型
    """
    from src.model_trainer import train_multilabel_classifier, predict_sentence

    logger.info("=" * 60)
    logger.info("  交互推理模式")
    logger.info("  输入文本: %s", text)
    logger.info("=" * 60)

    try:
        # 先训练模型（或后续可扩展为加载已保存的模型文件）
        logger.info("正在加载/训练模型...")
        result = train_multilabel_classifier(logger=logger)

        # 推理预测
        predicted = predict_sentence(
            text,
            result["model"],
            result["vectorizer"],
            result["mlb"],
            logger=logger,
        )

        # 输出结果
        print("\n" + "=" * 50)
        print(f"  输入文本: {text}")
        print(f"  预测标签: {predicted}")
        print("=" * 50 + "\n")

        return True
    except Exception as e:
        logger.exception("推理失败: %s", e)
        return False


# ============================================================================
# 主函数
# ============================================================================
def main(argv: Optional[list] = None) -> int:
    """
    项目主入口

    Args:
        argv: 命令行参数列表，None 时使用 sys.argv

    Returns:
        退出码: 0=成功, 1=失败
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # ---- 初始化日志 ----
    logger = setup_logger(
        "main",
        log_file=settings.LOG_FILE_PATH,
        level=args.log_level,
    )

    logger.info("Project1_DataClean 启动")
    logger.info("日志级别: %s | 日志文件: %s", args.log_level, settings.LOG_FILE_PATH)

    # ---- 路由分发 ----
    try:
        # 交互推理模式
        if args.infer:
            success = run_inference(args.infer, logger)
            return 0 if success else 1

        # 流水线模式
        if args.pipeline is None:
            parser.print_help()
            return 0

        if args.pipeline == "all":
            success = run_full_pipeline(logger, show_plot=not args.no_plot)
        elif args.pipeline == "generate":
            success = run_generate(logger)
        elif args.pipeline == "clean":
            success = run_clean(logger)
        elif args.pipeline == "analyze":
            success = run_analyze(logger, show_plot=not args.no_plot)
        elif args.pipeline == "train":
            success = run_train(logger)
        else:
            logger.error("未知流水线阶段: %s", args.pipeline)
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        return 130
    except Exception as e:
        logger.exception("未捕获的异常: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
