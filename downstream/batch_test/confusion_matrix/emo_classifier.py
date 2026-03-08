import json
import csv
from funasr import AutoModel
import os

# ======== 配置部分 ========

# 模型路径
model_id = "/data/zhaohaishu/Models/emotion2vec_plus_large"

# JSON文件路径（刚刚生成的）
json_path = "/data/zhaohaishu/Codes/zjx/Chinese_Emotional_Reading_Audio_Dataset/all_data.json"

# 输出结果的CSV文件
output_csv_path = "/data/zhaohaishu/Codes/zjx/Chinese_Emotional_Reading_Audio_Dataset/Chinese_Emotional_Reading_Audio_Dataset_classification_results.csv"

# ==========================

# 加载模型
model = AutoModel(
    model=model_id,
    disable_update=True,
    trust_remote_code=False,
)

# 加载JSON数据
with open(json_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# 打开CSV文件准备写入
with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(["wav_path", "true_emotion", "predicted_emotion", "classification_confidence"])

    # 遍历每个音频文件
    for item in data_list:
        if item:
            wav_path = item["wav_path"]
            true_emotion = item["emotion"]

            # 检查文件是否存在
            if not os.path.exists(wav_path):
                print(f"⚠️ 文件不存在: {wav_path}")
                continue

            try:
                # 模型推理
                rec_result = model.generate(
                    wav_path,
                    output_dir="./outputs",
                    granularity="utterance",
                    extract_embedding=False
                )

                result = rec_result[0]
                labels = result["labels"]
                scores = result["scores"]

                # 获取预测结果
                max_index = scores.index(max(scores))
                predicted_label = labels[max_index]
                predicted_score = scores[max_index]

                max_index = scores.index(max(scores))
                raw_label = labels[max_index]

                # 只保留英文
                if "/" in raw_label:
                    predicted_label = raw_label.split("/")[-1]
                else:
                    predicted_label = raw_label

                predicted_score = scores[max_index]

                # 写入结果
                writer.writerow([wav_path, true_emotion, predicted_label, f"{predicted_score:.6f}"])

                print(f"✅ 处理完成: {os.path.basename(wav_path)} | 真: {true_emotion} | 预测: {predicted_label} ({predicted_score:.4f})")

            except Exception as e:
                print(f"❌ 处理 {wav_path} 时出错: {e}")

print(f"\n✅ 所有文件处理完毕，结果已保存至 {output_csv_path}")
