import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss

# 1. 读取数据，适配Label Studio导出字典格式
df = pd.read_csv("labeled_data.csv", encoding="utf-8")
texts = []
labels_list = []

for idx, row in df.iterrows():
    text = row["text"]
    ann_str = row["sentiment"]
    if pd.isna(ann_str):
        continue
    try:
        d = eval(str(ann_str))
        label_arr = d.get("choices", [])
        texts.append(text)
        labels_list.append(label_arr)
    except:
        continue

# 2. 文本向量化
vec = TfidfVectorizer()
X = vec.fit_transform(texts)

# 3. 多标签编码
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels_list)

# 4. 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. 构建多标签分类模型
base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(base_clf)
model.fit(X_train, y_train)

# 6. 模型评估
y_pred = model.predict(X_test)
print("===== 汉明损失 Hamming Loss =====", hamming_loss(y_test, y_pred))
print("\n===== 分类报告 =====")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# 7. 推理测试函数
def predict_sentence(sentence):
    vec_x = vec.transform([sentence])
    pred = model.predict(vec_x)
    res_labels = mlb.inverse_transform(pred)[0]
    return res_labels

test_sent = "空调开机报错无法启动"
print(f"\n输入句子：{test_sent}")
print("预测标签：", predict_sentence(test_sent))