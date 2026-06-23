"""
src 业务模块包初始化

项目核心业务逻辑分为四个子模块：
- data_generator: 合成数据生成
- data_cleaner:   数据清洗管道
- data_analyzer:  标签分布统计与可视化
- model_trainer:  TF-IDF + 随机森林多标签分类模型
"""

__all__ = [
    "data_generator",
    "data_cleaner",
    "data_analyzer",
    "model_trainer",
]
