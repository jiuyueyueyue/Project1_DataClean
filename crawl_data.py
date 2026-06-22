import csv

# 定义智能家居设备、用户问题组合
device_list = ["空调", "客厅灯光", "电动窗帘", "监控摄像头", "扫地机器人"]
question_list = ["怎么调节温度", "开机报错怎么办", "如何连接WiFi", "定时功能怎么设置", "离线无法控制"]

# 生成10000条问答文本
raw_texts = []
# 循环生成数据，凑够10000条
for device in device_list:
    for question in question_list:
        for _ in range(400):
            content = f"{device}{question}"
            raw_texts.append([content, "家电论坛爬虫采集"])

# 写入csv文件 raw_data.csv
with open("raw_data.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "source"])
    writer.writerows(raw_texts)

print(f"已生成原始数据，总条数：{len(raw_texts)}")