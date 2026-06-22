# 智能家居问答多标签文本分类项目
## 项目简介
本项目实现智能家居用户问句采集、数据清洗、人工标注、数据分析、多标签文本分类全流程，
输入用户问题，同时预测三类标签：设备类型、问题类型、紧急等级。

## 项目流程
1. 数据爬虫：crawl_data.py 生成原始数据集 raw_data.csv
2. 数据清洗：clean_pipeline.py 过滤无效文本，得到 clean_data.csv
3. 数据标注：使用 Label Studio 做多标签人工标注，导出 labeled_data.csv
4. 数据分析：data_analysis.py 统计标签分布并绘图
5. 模型训练：train_classifier.py TF-IDF + 随机森林实现多标签分类

## 环境依赖
```bash
pip install pandas matplotlib scikit-learn