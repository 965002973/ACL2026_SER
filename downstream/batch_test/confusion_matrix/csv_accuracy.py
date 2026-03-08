import pandas as pd

# ======== 配置部分 ========
csv_path = "/data/zhaohaishu/Codes/zjx/Chinese_Emotional_Reading_Audio_Dataset/Chinese_Emotional_Reading_Audio_Dataset_classification_results.csv"  # 输入的CSV文件路径
# ==========================

# 读取CSV
df = pd.read_csv(csv_path)

# -------------------------------
# 数据清洗：提取英文标签部分
# -------------------------------
def extract_english_label(x):
    if isinstance(x, str):
        # 如果包含斜杠，如 "中立/neutral" → "neutral"
        if "/" in x:
            return x.split("/")[-1].strip().lower()
        else:
            return x.strip().lower()
    return x

df["true_emotion"] = df["true_emotion"].apply(extract_english_label)
df["predicted_emotion"] = df["predicted_emotion"].apply(extract_english_label)

# -------------------------------
# 计算每种情绪的分类准确率
# -------------------------------
accuracy_per_class = (
    df.groupby("true_emotion")
    .apply(lambda x: (x["true_emotion"] == x["predicted_emotion"]).mean())
    .reset_index(name="accuracy")
)

# 总体准确率
overall_acc = (df["true_emotion"] == df["predicted_emotion"]).mean()

# -------------------------------
# 保存预测错误的样本
# -------------------------------
incorrect_df = df[df["true_emotion"] != df["predicted_emotion"]]
output_incorrect_csv = "/data/zhaohaishu/Codes/zjx/Chinese_Emotional_Reading_Audio_Dataset/Chinese_misclassified_samples.csv"
incorrect_df.to_csv(output_incorrect_csv, index=False)

# -------------------------------
# 打印结果
# -------------------------------
print("📊 每个情绪的分类准确率：")
print(accuracy_per_class)
print("\n✅ 总体准确率: {:.2f}%".format(overall_acc * 100))
print(f"❌ 错误分类样本已保存到: {output_incorrect_csv}")
print(f"共 {len(incorrect_df)} 条错误样本。")
