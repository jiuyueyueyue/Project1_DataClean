"""
多标签文本分类模型训练模块

使用 TF-IDF 文本向量化 + 随机森林多输出分类器构建多标签分类模型，
支持:
  - 从 Label Studio 标注数据自动提取文本/标签
  - 训练/测试集划分
  - 模型训练与评估（Hamming Loss, Classification Report）
  - 单句推理预测

原文件: train_classifier.py
重构点:
  - eval() 安全漏洞 → utils.label_parser 安全解析
  - 硬编码模型参数 → config.settings 集中管理
  - 无异常处理 → try-except 全流程覆盖
  - 裸 print → structured logging
  - 无类型注解 → 完整类型标注
  - 重复解析逻辑 → 复用 parse_label_studio_annotations
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from config import settings
from utils.io_utils import safe_read_csv
from utils.label_parser import parse_label_studio_annotations


def train_multilabel_classifier(
    input_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    训练多标签文本分类模型（全流程）

    流程:
    1. 读取 Label Studio 导出数据
    2. 安全解析标注列，提取文本和多标签
    3. TF-IDF 文本向量化
    4. MultiLabelBinarizer 标签编码
    5. 训练/测试集划分（stratify 不可用于多标签，使用普通划分）
    6. RandomForest + MultiOutputClassifier 训练
    7. 模型评估（Hamming Loss + Classification Report）
    8. 返回模型及评估结果字典

    Args:
        input_path: 标注数据 CSV 路径，默认 settings.LABELED_DATA_PATH
        logger:     logger 实例

    Returns:
        包含以下键的字典:
        - "model":          训练好的模型对象
        - "vectorizer":     TF-IDF 向量化器
        - "mlb":            MultiLabelBinarizer 实例
        - "hamming_loss":   汉明损失值
        - "report":         分类报告字符串
        - "test_count":     测试集样本数
        - "train_count":    训练集样本数

    Raises:
        ValueError: 标注数据为空或标签类别不足以训练时抛出
    """
    log = logger or logging.getLogger(__name__)
    input_path = input_path or settings.LABELED_DATA_PATH

    log.info("=" * 50)
    log.info("开始多标签分类模型训练")
    log.info("=" * 50)

    # ========================================================================
    # Step 1: 读取并解析标注数据
    # ========================================================================
    df = safe_read_csv(input_path, encoding=settings.CSV_ENCODING, logger=log)
    if df.empty:
        raise ValueError(f"标注数据文件为空或读取失败: {input_path}")

    texts, labels_list = parse_label_studio_annotations(df, logger=log)

    if len(texts) == 0:
        raise ValueError("未能从标注文件中提取到有效文本-标签对")
    if len(texts) < 10:
        log.warning("标注数据量仅 %d 条，模型效果可能不理想", len(texts))

    log.info("提取到 %d 条有效标注数据", len(texts))

    # ========================================================================
    # Step 2: TF-IDF 文本向量化
    # ========================================================================
    log.info("正在进行 TF-IDF 向量化 (max_features=%d)...", settings.TFIDF_MAX_FEATURES)
    try:
        vectorizer = TfidfVectorizer(max_features=settings.TFIDF_MAX_FEATURES)
        X = vectorizer.fit_transform(texts)
        log.info("向量化完成，特征维度: %d", X.shape[1])
    except Exception as e:
        log.exception("TF-IDF 向量化失败: %s", e)
        raise

    # ========================================================================
    # Step 3: 多标签编码
    # ========================================================================
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels_list)
    log.info("标签编码完成，标签类别: %s", list(mlb.classes_))

    if len(mlb.classes_) < 2:
        log.warning("标签类别仅 %d 个，多标签分类意义有限", len(mlb.classes_))

    # ========================================================================
    # Step 4: 训练/测试集划分
    # ========================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TRAIN_TEST_SPLIT_RATIO,
        random_state=settings.RANDOM_SEED,
    )
    log.info(
        "数据集划分完成: 训练集 %d 条, 测试集 %d 条 (比例 %.0f/%.0f)",
        X_train.shape[0],
        X_test.shape[0],
        (1 - settings.TRAIN_TEST_SPLIT_RATIO) * 100,
        settings.TRAIN_TEST_SPLIT_RATIO * 100,
    )

    # ========================================================================
    # Step 5: 构建并训练模型
    # ========================================================================
    log.info(
        "构建模型: RandomForest(n_estimators=%d) + MultiOutputClassifier",
        settings.RF_N_ESTIMATORS,
    )
    base_clf = RandomForestClassifier(
        n_estimators=settings.RF_N_ESTIMATORS,
        random_state=settings.RANDOM_SEED,
    )
    model = MultiOutputClassifier(base_clf)

    try:
        model.fit(X_train, y_train)
        log.info("模型训练完成")
    except Exception as e:
        log.exception("模型训练失败: %s", e)
        raise

    # ========================================================================
    # Step 6: 模型评估
    # ========================================================================
    y_pred = model.predict(X_test)

    h_loss = float(hamming_loss(y_test, y_pred))
    report = classification_report(
        y_test, y_pred,
        target_names=mlb.classes_,
        zero_division=0,
    )

    log.info("===== 模型评估结果 =====")
    log.info("汉明损失 (Hamming Loss): %.4f", h_loss)
    log.info("分类报告:\n%s", report)

    # ========================================================================
    # Step 7: 返回结果
    # ========================================================================
    return {
        "model": model,
        "vectorizer": vectorizer,
        "mlb": mlb,
        "hamming_loss": h_loss,
        "report": report,
        "train_count": X_train.shape[0],
        "test_count": X_test.shape[0],
    }


def predict_sentence(
    sentence: str,
    model: Any,
    vectorizer: TfidfVectorizer,
    mlb: MultiLabelBinarizer,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    对单句文本进行多标签推理预测

    Args:
        sentence:   输入文本
        model:      训练好的 MultiOutputClassifier 模型
        vectorizer: TF-IDF 向量化器
        mlb:        MultiLabelBinarizer 实例
        logger:     logger 实例

    Returns:
        预测的标签列表 List[str]

    Raises:
        ValueError: 输入文本为空时抛出
    """
    log = logger or logging.getLogger(__name__)

    if not sentence or not sentence.strip():
        raise ValueError("输入文本不能为空")

    try:
        vec_x = vectorizer.transform([sentence])
        pred = model.predict(vec_x)
        res_labels = list(mlb.inverse_transform(pred)[0])
        log.info("推理完成: '%s' → %s", sentence[:50], res_labels)
        return res_labels
    except Exception as e:
        log.exception("推理预测失败: %s", e)
        raise


# ============================================================================
# 模块独立运行入口
# ============================================================================
if __name__ == "__main__":
    from utils.logger import setup_logger

    logger = setup_logger(
        "model_trainer",
        log_file=settings.LOG_FILE_PATH,
        level=settings.LOG_LEVEL,
    )
    logger.info("=" * 60)
    logger.info("独立运行: 模型训练模块")
    logger.info("=" * 60)

    try:
        result = train_multilabel_classifier(logger=logger)

        # 推理测试
        test_sent = "空调开机报错无法启动"
        logger.info("测试推理: %s", test_sent)
        predicted = predict_sentence(
            test_sent,
            result["model"],
            result["vectorizer"],
            result["mlb"],
            logger=logger,
        )
        logger.info("预测标签: %s", predicted)

    except Exception as e:
        logger.exception("模型训练失败: %s", e)
