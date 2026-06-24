"""
电商智能客服意图识别模型训练模块

基于 TF-IDF 文本向量化 + 随机森林多输出分类器，
对客服对话日志进行意图识别多标签分类模型训练。

使用场景:
  - 输入用户问句，预测意图标签（如: 物流查询、退换货、商品咨询等）
  - 训练数据来源: 清洗后的客服对话日志 (chatlog)
  - 为上层 NLU 引擎提供意图分类能力

原文件: train_classifier.py（智能家居多标签分类）
改造点:
  - 训练数据从 labeled_data.csv (Label Studio) 切换为对话日志 chatlog
  - 适配电商意图标签体系
  - 保留 TF-IDF + RandomForest 架构不变
  - 新增模型导出为 .pkl（可选持久化）
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


def train_intent_classifier(
    input_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    训练电商意图识别多标签分类模型

    流程:
    1. 读取清洗后的对话日志数据
    2. 提取 user 消息和 intent_label 作为训练样本
    3. TF-IDF 文本向量化
    4. MultiLabelBinarizer 标签编码
    5. 训练/测试集划分
    6. RandomForest + MultiOutputClassifier 训练
    7. 模型评估

    Args:
        input_path: 对话日志 CSV 路径，默认 settings.PROCESSED_DIR / "clean_chatlog.csv"
        logger:     logger 实例

    Returns:
        包含以下键的字典:
        - "model":          训练好的 MultiOutputClassifier
        - "vectorizer":     TF-IDF 向量化器
        - "mlb":            MultiLabelBinarizer 实例
        - "hamming_loss":   汉明损失
        - "report":         分类报告字符串
        - "train_count":    训练集样本数
        - "test_count":     测试集样本数

    Raises:
        ValueError: 训练数据为空或标签类别不足时抛出
    """
    log = logger or logging.getLogger(__name__)

    if input_path is None:
        input_path = settings.PROCESSED_DIR / "clean_chatlog.csv"

    log.info("=" * 60)
    log.info("  电商意图识别模型训练")
    log.info("=" * 60)

    # ========================================================================
    # Step 1: 读取并预处理对话数据
    # ========================================================================
    df = safe_read_csv(input_path, encoding=settings.CSV_ENCODING, logger=log)
    if df.empty:
        raise ValueError(f"对话日志数据为空或读取失败: {input_path}")

    # 仅使用 user 消息及其意图标签进行训练
    user_df = df[df["role"] == "user"].copy()
    if user_df.empty:
        raise ValueError("对话日志中无 user 角色数据，无法训练意图分类器")

    # 过滤无标签的样本
    user_df = user_df[user_df["intent_label"].notna() & (user_df["intent_label"] != "")]
    if user_df.empty:
        raise ValueError("对话日志中无有效意图标签，请检查 chatlog 数据的 intent_label 列")

    texts = user_df["message"].astype(str).tolist()
    # 将意图标签包装为列表（兼容 MultiLabelBinarizer）
    labels_list: List[List[str]] = [[lbl.strip()] for lbl in user_df["intent_label"].tolist()]

    unique_labels = sorted(set(lbl for sublist in labels_list for lbl in sublist))
    log.info("提取训练样本: %d 条, 意图类别: %d 种", len(texts), len(unique_labels))
    log.info("意图标签: %s", unique_labels)

    if len(unique_labels) < 2:
        log.warning("意图类别仅 %d 种，分类器效果可能有限", len(unique_labels))

    # ========================================================================
    # Step 2: TF-IDF 向量化
    # ========================================================================
    log.info("正在进行 TF-IDF 向量化 (max_features=%d)...", settings.TFIDF_MAX_FEATURES)
    try:
        vectorizer = TfidfVectorizer(
            max_features=settings.TFIDF_MAX_FEATURES,
            analyzer="char_wb",
            ngram_range=(2, 4),
        )
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
    log.info("标签编码完成，输出维度: %d × %d", y.shape[0], y.shape[1])

    # ========================================================================
    # Step 4: 训练/测试集划分
    # ========================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TRAIN_TEST_SPLIT_RATIO,
        random_state=settings.RANDOM_SEED,
    )
    log.info(
        "数据集划分: 训练集 %d 条, 测试集 %d 条 (%.0f/%.0f)",
        X_train.shape[0], X_test.shape[0],
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

    return {
        "model": model,
        "vectorizer": vectorizer,
        "mlb": mlb,
        "hamming_loss": h_loss,
        "report": report,
        "train_count": X_train.shape[0],
        "test_count": X_test.shape[0],
    }


def predict_intent(
    sentence: str,
    model: Any,
    vectorizer: TfidfVectorizer,
    mlb: MultiLabelBinarizer,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    对用户输入问句进行意图识别预测

    Args:
        sentence:   用户输入文本
        model:      训练好的 MultiOutputClassifier
        vectorizer: TF-IDF 向量化器
        mlb:        MultiLabelBinarizer 实例
        logger:     logger 实例

    Returns:
        预测的意图标签列表

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
        log.info("意图识别: '%s' → %s", sentence[:60], res_labels)
        return res_labels
    except Exception as e:
        log.exception("意图预测失败: %s", e)
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
    logger.info("独立运行: 电商意图识别模型训练模块")
    logger.info("=" * 60)

    try:
        result = train_intent_classifier(logger=logger)

        # 演示推理
        test_cases = [
            "我的快递到哪了？",
            "收到的商品有质量问题怎么退货？",
            "优惠券为什么用不了？",
        ]
        logger.info("--- 演示推理 ---")
        for tc in test_cases:
            predicted = predict_intent(
                tc, result["model"], result["vectorizer"], result["mlb"], logger=logger
            )
            logger.info("  输入: %s → 意图: %s", tc, predicted)

    except Exception as e:
        logger.exception("模型训练失败: %s", e)
