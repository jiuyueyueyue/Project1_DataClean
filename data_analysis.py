import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 修复中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取CSV，指定编码防止乱码
df = pd.read_csv("labeled_data.csv", encoding="utf-8")
all_labels = []

# 遍历sentiment列
for cell in df["sentiment"]:
    if pd.isna(cell):
        continue
    try:
        d = eval(str(cell))
        label_list = d.get("choices", [])
        all_labels.extend(label_list)
    except:
        continue

if len(all_labels) == 0:
    print("未解析到任何有效标签，请检查标注内容")
else:
    counter = Counter(all_labels)
    print("==== 各类标签统计数量 ====")
    for lab, cnt in counter.most_common():
        print(f"{lab} : {cnt}")

    plt.figure(figsize=(12, 6))
    plt.bar(counter.keys(), counter.values(), color="#4285F4")
    plt.xticks(rotation=45, ha="right")
    plt.title("智能家居问答数据集标签分布统计")
    plt.ylabel("样本数量")
    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=300)
    plt.show()