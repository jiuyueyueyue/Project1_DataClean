#!/usr/bin/env python3
"""
电商智能客服知识库 & 对话语料预处理流水线 —— 统一启动入口

对齐京小智、飞鸽后台数据生产流程，为上层意图识别模型和 RAG 向量库
提供标准化数据集输入。

支持运行模式:
  1. 全流程:   python main.py --pipeline all
  2. 分步执行: python main.py --pipeline generate     (数据生成)
              python main.py --pipeline clean         (数据清洗)
              python main.py --pipeline analyze       (语料分析)
              python main.py --pipeline kb_preprocess (知识库切片)
              python main.py --pipeline train         (意图模型训练)
  3. 意图推理: python main.py --infer "我的快递到哪了"
  4. RAG导出:  python main.py --pipeline all --export-rag ./rag_output/

Usage:
    python main.py --help      查看完整帮助
    python main.py --pipeline all --log-level DEBUG   调试模式
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
        prog="EcomDataPipeline",
        description="电商智能客服知识库 & 对话语料预处理流水线 —— "
                    "数据生成 → 清洗 → 分析 → 知识库切片 → 意图模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --pipeline all                       # 一键执行完整流程
  python main.py --pipeline generate                  # 仅生成四类业务数据
  python main.py --pipeline clean                     # 仅执行语料清洗
  python main.py --pipeline analyze                   # 仅分析语料分布
  python main.py --pipeline kb_preprocess             # 仅构建知识库切片
  python main.py --pipeline train                     # 仅训练意图识别模型
  python main.py --infer "我的快递到哪了"              # 交互意图推理
  python main.py --pipeline all --export-rag ./output/ # 全流程 + 导出RAG数据集
  python main.py --pipeline all --log-level DEBUG     # 调试模式
        """,
    )

    # ---- 主命令 ----
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["all", "generate", "clean", "analyze", "kb_preprocess", "train"],
        default=None,
        help="执行指定阶段: all=全流程, generate=数据生成, clean=数据清洗, "
             "analyze=语料分析, kb_preprocess=知识库切片, train=模型训练",
    )
    parser.add_argument(
        "--infer",
        type=str,
        default=None,
        metavar="TEXT",
        help="交互推理模式: 输入用户问句，输出预测意图标签",
    )
    parser.add_argument(
        "--export-rag",
        type=str,
        default=None,
        metavar="DIR",
        help="导出 RAG 就绪数据集到指定目录（含知识库切片 JSONL + 清洗语料 CSV）",
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


# ============================================================================
# 各阶段执行函数
# ============================================================================

def run_generate(logger) -> bool:
    """执行数据生成阶段"""
    from src.data_generator import generate_all_data
    logger.info(">>> 阶段 1/5: 四类业务数据生成")
    try:
        results = generate_all_data(logger=logger)
        for dtype, df in results.items():
            logger.info("  [%s] 生成 %d 条", dtype, len(df))
        logger.info("[OK] 数据生成完成")
        return True
    except Exception as e:
        logger.exception("[FAIL] 数据生成失败: %s", e)
        return False


def run_clean(logger) -> bool:
    """执行数据清洗阶段"""
    from src.data_cleaner import clean_corpus
    logger.info(">>> 阶段 2/5: 语料清洗")
    try:
        results = clean_corpus(logger=logger)
        total = sum(len(df) for df in results.values())
        logger.info("[OK] 语料清洗完成, 有效数据 %d 条", total)
        return True
    except Exception as e:
        logger.exception("[FAIL] 语料清洗失败: %s", e)
        return False


def run_analyze(logger, show_plot: bool = True) -> bool:
    """执行语料分析阶段"""
    from src.data_analyzer import analyze_corpus
    logger.info(">>> 阶段 3/5: 语料统计分析")
    try:
        analyze_corpus(show_plot=show_plot, logger=logger)
        logger.info("[OK] 语料分析完成")
        return True
    except Exception as e:
        logger.exception("[FAIL] 语料分析失败: %s", e)
        return False


def run_kb_preprocess(logger) -> bool:
    """执行知识库切片预处理阶段"""
    from src.kb_preprocessor import build_knowledge_base_chunks
    logger.info(">>> 阶段 4/5: 知识库切片预处理")
    try:
        chunks = build_knowledge_base_chunks(logger=logger)
        logger.info("[OK] 知识库切片完成, 共 %d 个切片", len(chunks))
        return True
    except Exception as e:
        logger.exception("[FAIL] 知识库切片失败: %s", e)
        return False


def run_train(logger) -> bool:
    """执行意图识别模型训练阶段"""
    from src.model_trainer import train_intent_classifier, predict_intent
    logger.info(">>> 阶段 5/5: 意图识别模型训练")
    try:
        result = train_intent_classifier(logger=logger)

        # 演示推理
        demo_texts = [
            "我的快递到哪了？",
            "收到的商品有质量问题怎么办？",
            "优惠券为什么用不了？",
        ]
        logger.info("--- 演示推理 ---")
        for dt in demo_texts:
            predicted = predict_intent(
                dt, result["model"], result["vectorizer"], result["mlb"], logger=logger
            )
            logger.info("  输入: %s → 意图: %s", dt, predicted)

        logger.info("[OK] 模型训练完成 (Hamming Loss: %.4f)", result["hamming_loss"])
        return True
    except Exception as e:
        logger.exception("[FAIL] 模型训练失败: %s", e)
        return False


# ============================================================================
# 全流程编排
# ============================================================================

def run_full_pipeline(logger, show_plot: bool = True) -> bool:
    """执行全流程: 生成 → 清洗 → 分析 → 知识库切片 → 模型训练"""
    logger.info("=" * 60)
    logger.info("  电商智能客服数据预处理 —— 全流程启动")
    logger.info("=" * 60)

    stages: list = [
        ("数据生成",       lambda: run_generate(logger)),
        ("语料清洗",       lambda: run_clean(logger)),
        ("语料分析",       lambda: run_analyze(logger, show_plot=show_plot)),
        ("知识库切片",     lambda: run_kb_preprocess(logger)),
        ("意图模型训练",   lambda: run_train(logger)),
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


def export_rag_dataset(export_dir: str, logger) -> bool:
    """
    导出 RAG 就绪数据集

    将知识库切片 JSONL + 清洗语料 CSV 复制到指定目录，
    便于直接接入 LangChain / LlamaIndex 等 RAG 框架。

    Args:
        export_dir: 导出目标目录
        logger:     logger 实例

    Returns:
        是否导出成功
    """
    import shutil

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  导出 RAG 就绪数据集 → %s", export_path)
    logger.info("=" * 60)

    # 导出知识库切片
    kb_src = settings.KB_CHUNKS_PATH
    if kb_src.exists():
        kb_dst = export_path / kb_src.name
        shutil.copy2(kb_src, kb_dst)
        logger.info("[OK] 知识库切片: %s", kb_dst)
    else:
        logger.warning("知识库切片文件不存在: %s，请先运行 kb_preprocess", kb_src)

    # 导出清洗语料
    corpus_src = settings.CLEAN_DATA_PATH
    if corpus_src.exists():
        corpus_dst = export_path / corpus_src.name
        shutil.copy2(corpus_src, corpus_dst)
        logger.info("[OK] 清洗语料: %s", corpus_dst)
    else:
        logger.warning("清洗语料不存在: %s，请先运行 clean", corpus_src)

    # 导出各类型清洗后数据
    for dtype in ["faq", "product", "aftersales", "chatlog"]:
        src = settings.PROCESSED_DIR / f"clean_{dtype}.csv"
        if src.exists():
            dst = export_path / f"clean_{dtype}.csv"
            shutil.copy2(src, dst)
            logger.info("[OK] %s 清洗数据: %s", dtype, dst)

    # 生成 README
    readme_path = export_path / "README.txt"
    readme_path.write_text(
        f"""RAG 就绪数据集导出
==================
来源: 电商智能客服知识库 & 对话语料预处理流水线
导出时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

文件说明:
- kb_chunks.jsonl    知识库向量切片 (FAQ+商品+规则)，可直接导入向量数据库
- clean_faq.csv      清洗后的 FAQ 问答对
- clean_product.csv  清洗后的商品资料
- clean_aftersales.csv 清洗后的售后规则
- clean_chatlog.csv  清洗后的客服对话日志 (含意图标签)
- clean_corpus.csv   全部语料合并文件

使用方式 (以 LangChain 为例):
  from langchain.document_loaders import JSONLoader
  loader = JSONLoader(file_path="kb_chunks.jsonl", jq_schema=".content", text_content=False)
  docs = loader.load()
  # → 将 docs 导入向量库进行 RAG 检索
""",
        encoding="utf-8",
    )
    logger.info("[OK] 导出说明: %s", readme_path)
    logger.info("RAG 数据集导出完成: %s", export_path)

    return True


# ============================================================================
# 交互推理
# ============================================================================

def run_inference(text: str, logger) -> bool:
    """
    交互推理模式: 训练模型并对输入问句进行意图识别
    """
    from src.model_trainer import train_intent_classifier, predict_intent

    logger.info("=" * 60)
    logger.info("  交互意图识别")
    logger.info("  输入: %s", text)
    logger.info("=" * 60)

    try:
        logger.info("正在训练模型...")
        result = train_intent_classifier(logger=logger)

        predicted = predict_intent(
            text, result["model"], result["vectorizer"], result["mlb"], logger=logger
        )

        print("\n" + "=" * 50)
        print(f"  用户问句: {text}")
        print(f"  预测意图: {predicted}")
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
        退出码: 0=成功, 1=失败, 130=用户中断
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # ---- 初始化日志 ----
    logger = setup_logger(
        "EcomPipeline",
        log_file=settings.LOG_FILE_PATH,
        level=args.log_level,
    )

    logger.info("电商智能客服数据预处理流水线 启动")
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

        pipeline_map = {
            "all":           lambda: run_full_pipeline(logger, show_plot=not args.no_plot),
            "generate":      lambda: run_generate(logger),
            "clean":         lambda: run_clean(logger),
            "analyze":       lambda: run_analyze(logger, show_plot=not args.no_plot),
            "kb_preprocess": lambda: run_kb_preprocess(logger),
            "train":         lambda: run_train(logger),
        }

        fn = pipeline_map.get(args.pipeline)
        if fn is None:
            logger.error("未知流水线阶段: %s", args.pipeline)
            return 1

        success = fn()

        # 额外 RAG 导出
        if success and args.export_rag:
            export_rag_dataset(args.export_rag, logger)

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        return 130
    except Exception as e:
        logger.exception("未捕获的异常: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
