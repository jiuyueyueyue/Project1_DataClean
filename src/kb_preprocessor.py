"""
知识库切片预处理模块

将 FAQ、商品资料、售后规则等长文本数据切分为适合 RAG 向量库的语义片段。
采用滑动窗口切片策略，确保相邻切片保留重叠上下文，避免语义断裂。

切片流程:
  1. 读取指定数据类型的清洗后 CSV
  2. 提取文本列，按标点/换行切分为句子
  3. 滑动窗口聚合句子至 chunk_size 字符数
  4. 相邻窗口保留 chunk_overlap 字符重叠
  5. 滤除过短碎片
  6. 导出 JSONL（适配 LangChain / LlamaIndex 直接加载）

导出格式:
  {"id": "faq_001_chunk_0", "content": "...", "metadata": {"source_type": "faq", ...}}

Usage:
    from src.kb_preprocessor import build_knowledge_base_chunks
    chunks = build_knowledge_base_chunks(logger=logger)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings
from utils.io_utils import safe_read_csv


def _split_sentences(text: str) -> List[str]:
    """
    中文分句 —— 按标点符号切割为独立句子

    切割符: 。！？；\n
    保留空句过滤，确保每个句子至少包含一个中文字符或字母。

    Args:
        text: 原始文本

    Returns:
        句子列表
    """
    if not text or not text.strip():
        return []
    # 按中文标点和换行切割
    raw_parts = re.split(r"[。！？；\n]+", text)
    # 过滤空句和纯空白句
    sentences = [s.strip() for s in raw_parts if s.strip() and len(s.strip()) >= 2]
    return sentences


def _sliding_window_chunk(
    sentences: List[str],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    min_chunk_length: int = 50,
) -> List[str]:
    """
    滑动窗口切片 —— 将句子列表聚合为指定大小的文本块

    算法:
      从头遍历句子，累加字符数。
      当累加长度 >= chunk_size 时，截断为一个 chunk。
      下一个 chunk 从 (当前 chunk 末尾向前回溯 overlap 字符) 的位置开始。

    Args:
        sentences:        句子列表
        chunk_size:       每个切片最大字符数
        chunk_overlap:    相邻切片重叠字符数
        min_chunk_length: 最小切片长度（低于此值的碎片丢弃）

    Returns:
        文本切片列表
    """
    if not sentences:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0
    i = 0

    while i < len(sentences):
        sent = sentences[i]
        sent_len = len(sent)

        # 当前句子加入后仍未超过 chunk_size，继续累加
        if current_len + sent_len <= chunk_size:
            current_chunk.append(sent)
            current_len += sent_len
            i += 1
        else:
            # 当前 chunk 已满，保存
            if current_chunk:
                chunk_text = "。".join(current_chunk) + "。"
                if len(chunk_text) >= min_chunk_length:
                    chunks.append(chunk_text)

            # 计算回溯位置（保留 overlap 字符的上下文）
            if chunk_overlap > 0 and current_chunk:
                overlap_chars = 0
                backtrack_idx = len(current_chunk) - 1
                while backtrack_idx >= 0 and overlap_chars < chunk_overlap:
                    overlap_chars += len(current_chunk[backtrack_idx])
                    backtrack_idx -= 1
                # 从回溯位置重新开始
                if backtrack_idx >= 0:
                    current_chunk = current_chunk[backtrack_idx + 1:]
                    current_len = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_len = 0
            else:
                current_chunk = []
                current_len = 0

            # 如果单个句子本身就超过 chunk_size，强制截断
            if sent_len > chunk_size:
                truncated = sent[:chunk_size]
                if len(truncated) >= min_chunk_length:
                    chunks.append(truncated)
                i += 1

    # 收尾: 保存最后一个未满的 chunk
    if current_chunk:
        chunk_text = "。".join(current_chunk) + "。"
        if len(chunk_text) >= min_chunk_length:
            chunks.append(chunk_text)

    return chunks


def _build_chunk_for_row(
    row: pd.Series,
    data_type: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_length: int,
) -> List[Dict[str, Any]]:
    """
    为单行数据构建切片列表

    根据数据类型选择不同的文本列进行切片:
      - faq:        question + answer 合并
      - product:    spu_name + description 合并
      - aftersales: rule_title + rule_content 合并

    Args:
        row:              DataFrame 行
        data_type:        数据类型
        chunk_size:       切片大小
        chunk_overlap:    重叠大小
        min_chunk_length: 最小切片长度

    Returns:
        该行数据切的切片列表 (dict 格式，含 id/content/metadata)
    """
    # 按数据类型选择文本列
    if data_type == "faq":
        text = f"{row.get('question', '')}。{row.get('answer', '')}"
        row_id = row.get("category", "faq")
        metadata = {
            "source_type": "faq",
            "category": row.get("category", ""),
        }
    elif data_type == "product":
        text = f"{row.get('spu_name', '')}。{row.get('description', '')}"
        row_id = row.get("spu_id", "product")
        metadata = {
            "source_type": "product",
            "spu_id": row.get("spu_id", ""),
            "category": row.get("category", ""),
            "brand": row.get("brand", ""),
        }
    elif data_type == "aftersales":
        text = f"{row.get('rule_title', '')}。{row.get('rule_content', '')}"
        row_id = row.get("rule_id", "aftersales")
        metadata = {
            "source_type": "aftersales",
            "rule_category": row.get("rule_category", ""),
            "rule_id": row.get("rule_id", ""),
        }
    else:
        return []

    sentences = _split_sentences(text)
    chunk_texts = _sliding_window_chunk(sentences, chunk_size, chunk_overlap, min_chunk_length)

    chunks = []
    for idx, chunk_content in enumerate(chunk_texts):
        chunks.append({
            "id": f"{row_id}_chunk_{idx}",
            "content": chunk_content,
            "metadata": {**metadata, "chunk_index": idx},
        })

    return chunks


def build_knowledge_base_chunks(
    source_types: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    min_chunk_length: Optional[int] = None,
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    构建知识库切片 —— 主入口

    读取清洗后的 FAQ/商品/售后数据，按滑动窗口切分为语义片段，
    导出为 JSONL 格式，可直接导入 LangChain / LlamaIndex 向量库。

    Args:
        source_types:     参与切片的数据类型列表，默认 settings.KB_SOURCE_TYPES
        chunk_size:       切片最大字符数，默认 settings.KB_CHUNK_SIZE
        chunk_overlap:    重叠字符数，默认 settings.KB_CHUNK_OVERLAP
        min_chunk_length: 最小切片长度，默认 settings.KB_MIN_CHUNK_LENGTH
        output_path:      JSONL 输出路径，默认 settings.KB_CHUNKS_PATH
        logger:           logger 实例

    Returns:
        切片字典列表，每个元素包含 id/content/metadata
    """
    log = logger or logging.getLogger(__name__)
    source_types = source_types or settings.KB_SOURCE_TYPES
    chunk_size = chunk_size or settings.KB_CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.KB_CHUNK_OVERLAP
    min_chunk_length = min_chunk_length or settings.KB_MIN_CHUNK_LENGTH
    output_path = output_path or settings.KB_CHUNKS_PATH

    log.info("=" * 60)
    log.info("  知识库切片预处理")
    log.info("  切片大小: %d | 重叠: %d | 最小长度: %d", chunk_size, chunk_overlap, min_chunk_length)
    log.info("  数据类型: %s", source_types)
    log.info("=" * 60)

    # 数据类型 → 文件路径映射
    type_path_map = {
        "faq":        settings.PROCESSED_DIR / "clean_faq.csv",
        "product":    settings.PROCESSED_DIR / "clean_product.csv",
        "aftersales": settings.PROCESSED_DIR / "clean_aftersales.csv",
    }

    all_chunks: List[Dict[str, Any]] = []

    for data_type in source_types:
        file_path = type_path_map.get(data_type)
        if file_path is None:
            log.warning("未知数据类型 '%s'，跳过", data_type)
            continue
        if not file_path.exists():
            log.warning("[%s] 清洗数据不存在: %s，请先运行数据生成和清洗", data_type, file_path)
            continue

        df = safe_read_csv(file_path, encoding=settings.CSV_ENCODING, logger=log)
        if df.empty:
            log.warning("[%s] 数据为空，跳过切片", data_type)
            continue

        type_chunks: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            chunks = _build_chunk_for_row(row, data_type, chunk_size, chunk_overlap, min_chunk_length)
            type_chunks.extend(chunks)

        log.info("[%s] %d 条数据 → %d 个切片", data_type, len(df), len(type_chunks))
        all_chunks.extend(type_chunks)

    # 导出 JSONL
    if all_chunks:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        log.info("知识库切片导出完成: %d 个切片 → %s", len(all_chunks), output_path)
    else:
        log.warning("未生成任何切片，请检查源数据")

    return all_chunks


# ============================================================================
# 模块独立运行入口
# ============================================================================
if __name__ == "__main__":
    from utils.logger import setup_logger

    logger = setup_logger(
        "kb_preprocessor",
        log_file=settings.LOG_FILE_PATH,
        level=settings.LOG_LEVEL,
    )
    logger.info("=" * 60)
    logger.info("独立运行: 知识库切片预处理模块")
    logger.info("=" * 60)

    try:
        chunks = build_knowledge_base_chunks(logger=logger)
        if chunks:
            logger.info("切片预览 (前3条):")
            for c in chunks[:3]:
                logger.info("  [%s] %s...", c["id"], c["content"][:80])
    except Exception as e:
        logger.exception("知识库切片失败: %s", e)
