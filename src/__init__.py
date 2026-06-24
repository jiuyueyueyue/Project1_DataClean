"""
src 业务模块包初始化

电商智能客服核心业务逻辑:
- data_generator:   四类电商业务数据生成（FAQ/商品/规则/对话）
- data_cleaner:     语料清洗管道（按数据类型差异化清洗）
- kb_preprocessor:  知识库切片预处理（滑动窗口 → RAG 向量库 JSONL）
- data_analyzer:    语料多维度统计与可视化
- model_trainer:    电商意图识别模型训练（TF-IDF + 随机森林）
"""

__all__ = [
    "data_generator",
    "data_cleaner",
    "kb_preprocessor",
    "data_analyzer",
    "model_trainer",
]
