# 🏠 智能家居问答多标签文本分类

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

端到端 NLP 多标签分类项目：从数据生成、清洗、标注到模型训练与评估的完整机器学习流水线。

---

## 📌 项目定位

本项目面向**智能家居场景**，对用户自然语言问句同时预测三个维度的标签：

| 标签维度 | 示例标签 | 业务价值 |
|---------|---------|---------|
| 设备类型 | 空调、灯光、窗帘、摄像头、机器人 | 精准路由至设备售后团队 |
| 问题类型 | 参数调节、故障报错、联网配置、离线异常 | 匹配对应知识库解决方案 |
| 紧急等级 | 普通咨询、紧急故障 | 客服工单优先级排序 |

**一句话描述**：输入 `"空调开机报错无法启动"`，模型同时预测 `["空调", "故障报错", "紧急故障"]`。

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    main.py (统一入口)                     │
│         python main.py --pipeline all                    │
└──────────┬──────────┬──────────┬──────────┬─────────────┘
           │          │          │          │
    ┌──────▼──┐ ┌────▼───┐ ┌───▼────┐ ┌──▼──────┐
    │generate │ │ clean  │ │analyze │ │ train   │
    │ 数据生成 │ │ 数据清洗│ │标签分析 │ │ 模型训练 │
    └────┬────┘ └───┬────┘ └───┬────┘ └───┬─────┘
         │          │          │          │
    ┌────▼────┐ ┌───▼───┐ ┌───▼────┐ ┌───▼─────┐
    │raw_data │ │clean   │ │label   │ │模型评估  │
    │  .csv   │ │_data   │ │_dist   │ │报告     │
    │         │ │ .csv   │ │ .png   │ │        │
    └─────────┘ └────────┘ └────────┘ └─────────┘
```

### 目录结构

```
Project1_DataClean/
├── config/                  # 🔧 配置层
│   ├── __init__.py          #    配置单例导出
│   ├── settings.py          #    全局配置数据类（零硬编码）
│   └── .env                 #    环境变量覆盖
│
├── src/                     # 🧠 业务逻辑层
│   ├── __init__.py
│   ├── data_generator.py    #    合成数据生成
│   ├── data_cleaner.py      #    数据清洗管道
│   ├── data_analyzer.py     #    标签分布分析
│   └── model_trainer.py     #    TF-IDF + 随机森林多标签分类
│
├── utils/                   # 🛠 通用工具层
│   ├── __init__.py
│   ├── logger.py            #    统一日志体系
│   ├── io_utils.py          #    文件安全读写
│   └── label_parser.py      #    标注数据安全解析
│
├── data/                    # 📊 数据层
│   ├── raw/                 #    原始合成数据
│   ├── processed/           #    清洗后数据
│   └── output/              #    模型产出、图表、日志
│
├── main.py                  # 🚀 统一启动入口
├── requirements.txt         # 📦 依赖清单
├── .gitignore               # 🚫 版本控制忽略规则
└── README.md                # 📖 本文档
```

### 技术栈

| 层次 | 技术选型 | 说明 |
|------|---------|------|
| 文本向量化 | TF-IDF (scikit-learn) | 经典稀疏特征，轻量可解释 |
| 多标签编码 | MultiLabelBinarizer | 将标签列表转为二值矩阵 |
| 分类模型 | RandomForest + MultiOutputClassifier | 为每个标签独立训练一个二分类器 |
| 数据处理 | pandas 2.x | 高性能 DataFrame 操作 |
| 可视化 | matplotlib 3.x | 标签分布柱状图 |
| 配置管理 | dataclass + .env | 类型安全 + 环境变量覆盖 |
| 日志系统 | logging (stdlib) | 控制台 + 文件双通道 |

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- pip 20.0+

### 1. 克隆项目

```bash
git clone https://github.com/jiuyueyueyue/Project1_DataClean.git
cd Project1_DataClean
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 一键运行全流程

```bash
python main.py --pipeline all
```

执行后将依次完成：
1. **数据生成** → `data/raw/raw_data.csv`（10000 条合成问答数据）
2. **数据清洗** → `data/processed/clean_data.csv`（去重、长度过滤）
3. **数据分析** → `data/output/label_distribution.png`（标签分布柱状图）
4. **模型训练** → 输出 Hamming Loss 和 Classification Report

### 4. 分步执行

```bash
# 仅生成数据
python main.py --pipeline generate

# 仅清洗数据
python main.py --pipeline clean

# 仅分析标签分布
python main.py --pipeline analyze

# 仅训练模型
python main.py --pipeline train

# 调试模式（输出详细日志）
python main.py --pipeline all --log-level DEBUG
```

### 5. 交互推理

```bash
python main.py --infer "空调无法连接WiFi"
```

输出示例：
```
  输入文本: 空调无法连接WiFi
  预测标签: ['空调', '联网配置', '普通咨询']
```

---

## 📊 运行示例

### 模型评估输出

```
===== 汉明损失 (Hamming Loss): 0.1250

===== 分类报告 =====
              precision    recall  f1-score   support

          空调       0.80      1.00      0.89         4
        客厅灯光       1.00      1.00      1.00         3
        电动窗帘       1.00      1.00      1.00         2
       监控摄像头       1.00      1.00      1.00         3
       扫地机器人       1.00      1.00      1.00         3
        参数调节       1.00      1.00      1.00         2
        故障报错       1.00      1.00      1.00         3
        联网配置       1.00      1.00      1.00         2
        离线异常       1.00      1.00      1.00         4
        普通咨询       1.00      1.00      1.00         2
        紧急故障       1.00      1.00      1.00         2

   micro avg       0.97      1.00      0.98        30
   macro avg       0.98      1.00      0.99        30
weighted avg       0.97      1.00      0.98        30
 samples avg       0.97      1.00      0.98        30
```

### 标签分布图

![标签分布](data/output/label_distribution.png)

---

## ✨ 工程化优化亮点

本版本相比初版的重构改进：

| 优化维度 | 初版问题 | 重构方案 |
|---------|---------|---------|
| 🏛️ **架构分层** | 4 个 py 文件平铺堆叠 | config / src / utils / data 四层解耦 |
| ⚙️ **配置管理** | 设备列表、路径、参数全部硬编码 | dataclass + .env 环境变量驱动，零硬编码 |
| 🔒 **安全加固** | `eval()` 直接执行标注字符串 | `ast.literal_eval` + `json.loads` 多路径安全解析 |
| 📝 **代码规范** | 无类型注解，无 docstring | 完整类型标注 + Google 风格文档字符串 |
| 📊 **可观测性** | `print()` 调试输出 | logging 模块，控制台 + 文件双通道 |
| 🛡️ **异常处理** | 文件读写无 try-except | 全流程 try-except + 入参校验 + 降级策略 |
| ♻️ **DRY 原则** | 标注解析逻辑两处重复 | 统一封装至 `utils/label_parser.py` |
| 🚀 **一键执行** | 需逐个手动运行脚本 | `main.py` 统一入口，支持全流程/分步/推理 |
| 📦 **依赖管理** | 无 requirements.txt | 版本锁定依赖清单 |
| 🚫 **版本控制** | 缺 .gitignore | 标准 Python 项目忽略规则 |

---

## 🎯 简历适配说明

本项目适合在**校招简历**中展示以下能力：

### 推荐展示位置

- **项目经历** / **机器学习项目** 板块

### 简历描述模板

> **智能家居问答多标签文本分类系统** | Python, scikit-learn, pandas
> - 设计并实现端到端 NLP 流水线：数据生成 → 清洗 → 标注 → 模型训练 → 评估
> - 使用 TF-IDF + 随机森林多输出分类器，实现设备类型/问题类型/紧急等级三维标签同时预测
> - 遵循 PEP8 工业级规范进行项目工程化重构：分层架构设计、配置解耦、安全加固（`ast.literal_eval` 替代 `eval`）
> - 引入 logging 日志体系、异常处理全链路覆盖、`main.py` 统一 CLI 入口

### 面试可延展方向

- **为什么选 TF-IDF + 随机森林**：适合小样本场景，可解释性强，比深度学习更轻量
- **如何改进**：替换为 BERT + 多标签分类头，引入数据增强提升泛化
- **如何处理标签不平衡**：class_weight='balanced'、Focal Loss
- **部署方案**：将模型序列化为 `.pkl`，通过 Flask/FastAPI 提供 REST API

---

## 📄 License

MIT License — 详见 [LICENSE](LICENSE) 文件

---

## 👤 作者

- **GitHub**: [@jiuyueyueyue](https://github.com/jiuyueyueyue)
- **项目仓库**: [Project1_DataClean](https://github.com/jiuyueyueyue/Project1_DataClean)
