# 🛒 电商智能客服知识库 & 对话语料预处理流水线

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **对齐京小智、飞鸽后台数据生产流程**，为上层意图识别模型和 RAG 向量库提供标准化数据集输入。

---

##  📌 项目定位

本项目是**整套电商智能客服系统的数据底座**，完成从原始业务数据到模型就绪数据的全链路预处理：

```
┌──────────────────────────────────────────────────────────────────┐
│                    电商智能客服系统架构                              │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────────┐ │
│  │  NLU 引擎 │   │ RAG 检索 │   │ 规则引擎  │   │ 对话分析      │ │
│  │ 意图识别  │   │ 知识问答 │   │ 售后路由  │   │ 质检/洞察     │ │
│  └─────┬─────┘   └────┬─────┘   └────┬─────┘   └───────┬───────┘ │
│        │              │              │                  │         │
│  ┌─────▼──────────────▼──────────────▼──────────────────▼───────┐ │
│  │              ◀ 本项目: 数据预处理流水线 ▶                      │ │
│  │  FAQ问答对 │ 商品资料 │ 售后规则 │ 客服对话日志 │ KB切片       │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 四类业务数据

| 数据类型 | 业务来源 | 用途 | 典型字段 |
|---------|---------|------|---------|
| **FAQ 问答对** | 知识库维护 | RAG 检索 / FAQ 匹配 | category, question, answer |
| **商品资料** | 商品中心 | 商品问答 / 属性检索 | spu_id, brand, specs, description |
| **售后规则** | 政策文档 | 规则引擎 / RAG 检索 | rule_category, rule_title, rule_content |
| **客服对话日志** | 飞鸽/京小智 | 意图识别模型训练 | session_id, role, message, intent_label |

---

## 🏗️ 技术架构

```
EcomDataPipeline
│
├── config/                     # 🔧 配置层
│   ├── settings.py             #    business constants + model params
│   └── .env                    #    env override
│
├── src/                        # 🧠 业务层
│   ├── data_generator.py       #    四类数据生成器
│   ├── data_cleaner.py         #    差异化清洗管道
│   ├── kb_preprocessor.py      #    RAG 知识库切片
│   ├── data_analyzer.py        #    语料多维统计
│   └── model_trainer.py        #    意图识别模型
│
├── utils/                      # 🛠 工具层
│   ├── logger.py               #    统一日志
│   ├── io_utils.py             #    安全 I/O
│   └── label_parser.py         #    标注解析
│
├── data/
│   ├── raw/                    #    原始生成数据（4类CSV）
│   ├── processed/              #    清洗后语料
│   └── output/                 #    图表、日志、KB切片JSONL
│
├── main.py                     # 🚀 统一入口（5阶段流水线）
├── requirements.txt
├── .gitignore
└── README.md
```

### 流水线流程

```
 generate           clean             analyze         kb_preprocess        train
 ┌───────┐        ┌───────┐         ┌──────────┐      ┌────────────┐    ┌──────────┐
 │ 4类数据 │  ──▶  │ 差异化  │  ──▶   │ 语料统计  │ ──▶ │ 滑动窗口切片│ ──▶│ 意图识别  │
 │ 生成    │       │ 清洗    │        │ 可视化    │     │ JSONL导出  │    │ 模型训练  │
 └───────┘        └───────┘         └──────────┘      └────────────┘    └──────────┘
     │                │                   │                   │               │
     ▼                ▼                   ▼                   ▼               ▼
 faq_data.csv    clean_faq.csv      corpus_dist.png    kb_chunks.jsonl   模型评估报告
 product_data.csv clean_product.csv                     (RAG向量库就绪)
 ...             ...
```

### 技术栈

| 层次 | 技术选型 | 说明 |
|------|---------|------|
| 文本处理 | pandas + jieba | 中文分词、数据清洗 |
| 知识库切片 | 滑动窗口算法 | chunk_size=512, overlap=128, JSONL导出 |
| 意图识别 | TF-IDF + RandomForest + MultiOutputClassifier | 轻量可解释，适合多标签 |
| 可视化 | matplotlib | 饼图/直方图/柱状图四合一 |
| 配置管理 | dataclass + .env | 类型安全 + 环境变量覆盖 |
| 日志系统 | logging | 控制台 + 文件双通道 |

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
1. **数据生成** → 四类电商业务数据（~7500条）
2. **数据清洗** → 差异化清洗，输出去重语料
3. **语料分析** → 四合一统计图 `data/output/corpus_distribution.png`
4. **知识库切片** → RAG 就绪 JSONL `data/output/kb_chunks.jsonl`
5. **意图模型训练** → Hamming Loss + 分类报告 + 演示推理

### 4. 分步执行

```bash
# 仅生成数据
python main.py --pipeline generate

# 仅清洗语料
python main.py --pipeline clean

# 仅分析语料分布
python main.py --pipeline analyze

# 仅构建知识库切片
python main.py --pipeline kb_preprocess

# 仅训练意图识别模型
python main.py --pipeline train

# 全流程 + 导出RAG数据集
python main.py --pipeline all --export-rag ./rag_output/
```

### 5. 交互意图识别

```bash
python main.py --infer "我的快递到哪了"
```

输出示例：
```
  用户问句: 我的快递到哪了
  预测意图: ['物流查询']
```

### 6. 调试模式

```bash
python main.py --pipeline all --log-level DEBUG
```

---

## 📊 运行示例

### 语料统计输出

```
===== 电商语料统计分析 =====
总语料量: 7462 条
  FAQ问答: 3000 条 (40.2%)
  商品资料: 2000 条 (26.8%)
  售后规则: 462 条 (6.2%)
  对话日志: 2000 条 (26.8%)

[FAQ问答] 子类分布: {'订单查询': 450, '退换货': 450, '物流配送': 450, ...}
[FAQ问答] 文本长度: 均值=128, 中位数=115, 范围=[45, 320]
[对话日志] 文本长度: 均值=38, 中位数=22, 范围=[3, 298]
```

### 知识库切片示例

```jsonl
{"id": "订单查询_chunk_0", "content": "如何查询订单物流状态。您可以在「我的订单」中找到对应订单...", "metadata": {"source_type": "faq", "category": "订单查询", "chunk_index": 0}}
{"id": "SPU000001_chunk_0", "content": "华为智能手机 001型。规格: 旗舰版 | 颜色: 星空灰...", "metadata": {"source_type": "product", "spu_id": "SPU000001", "chunk_index": 0}}
```

### 模型评估输出

```
===== 模型评估结果 =====
汉明损失 (Hamming Loss): 0.1750

===== 分类报告 =====
              precision    recall  f1-score   support

      商品咨询       0.85      0.90      0.87        42
      物流查询       0.92      0.88      0.90        38
      售后投诉       0.78      0.82      0.80        35
      账户管理       0.90      0.85      0.87        28
      活动咨询       0.88      0.92      0.90        25

   micro avg       0.87      0.88      0.87       168
   macro avg       0.87      0.87      0.87       168
```

---

## ✨ 工程化优化亮点

| 优化维度 | 说明 |
|---------|------|
| 🏛️ **分层架构** | config / src / utils / data 四层解耦，职责清晰 |
| ⚙️ **零硬编码** | dataclass + .env 环境变量驱动，自适应部署环境 |
| 🔒 **安全加固** | `ast.literal_eval` 替代 `eval`，安全解析标注数据 |
| 📝 **PEP8 规范** | 全函数类型注解 + Google 风格 docstring + logging |
| 🛡️ **全链路容错** | try-except + 入参校验 + 编码降级 + 阶段失败隔离 |
| ♻️ **高复用** | utils 工具层统一 I/O / 日志 / 解析 |
| 🚀 **一键执行** | `main.py` 支持 6 种运行模式 + RAG 数据集导出 |
| 🔪 **知识库切片** | 滑动窗口算法 + JSONL 导出，直接对接 LangChain/LlamaIndex |

---

## 🎯 简历适配说明

### 推荐展示位置

**项目经历** / **机器学习 & 数据工程** 板块

### 简历描述模板

> **电商智能客服知识库 & 对话语料预处理流水线** | Python, scikit-learn, pandas
> - 设计并实现端到端数据预处理流水线：FAQ/商品/规则/对话四类数据生成 → 差异化清洗 → 知识库切片 → 意图识别模型
> - 实现滑动窗口切片算法 (chunk_size=512, overlap=128)，输出 JSONL 格式直接对接 LangChain 向量库
> - 基于 TF-IDF + 随机森林多输出分类器构建意图识别模块，支持 10+ 电商意图类别
> - 遵循 PEP8 工业级规范：分层架构、配置解耦、全链路日志追踪、异常容错
> - 对齐京小智/飞鸽生产数据流程，可输出 RAG 就绪数据集和意图识别模型

### 面试可延展方向

- **切片策略选型**: 固定大小 vs 语义分块、Sentence-BERT 句子嵌入分块
- **向量库选型**: Milvus / Qdrant / Chroma，结合本项目 JSONL 一键导入
- **模型升级路径**: TF-IDF → BERT → 大模型微调 (ChatGLM/Qwen)
- **工程化加深**: 引入 Prefect/Airflow 调度、Docker 容器化、CI/CD 自动化测试
- **数据闭环**: 模型预测低置信度样本回流 → 人工复核 → 增量训练

---

## 📄 License

MIT License

---

## 👤 作者

- **GitHub**: [@jiuyueyueyue](https://github.com/jiuyueyueyue)
- **项目仓库**: [Project1_DataClean](https://github.com/jiuyueyueyue/Project1_DataClean)
