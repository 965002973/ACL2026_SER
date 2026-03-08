import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/data/zhaohaishu/Demos/TESS_4o_audio_manual_classification_results.csv")

y_true = df["true_emotion"].astype(str)
y_pred = df["predicted_emotion"].astype(str)

# ========= 2. 构建标签集合（加上 other）=========
labels = sorted(set(y_true) | set(y_pred) | {"other"} | {"<unk>"})

# ========= 3. 计算混淆矩阵 =========
cm = confusion_matrix(y_true, y_pred, labels=labels)

# # ========= 4. 画原始计数混淆矩阵并保存 =========
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm,
#             annot=True,
#             fmt="d",
#             xticklabels=labels,
#             yticklabels=labels)
# plt.xlabel("Predicted emotion")
# plt.ylabel("True emotion")
# plt.title("Confusion Matrix")
# plt.tight_layout()

# # ★ 保存图片（计数版）
# plt.savefig("confusion_matrix_counts.png", dpi=300, bbox_inches="tight")
# # 如果想看一下再关掉窗口
# plt.show()

# ========= 5. 画归一化混淆矩阵并保存 =========
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)  # 防止某些行全 0 出现 NaN

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm,
            annot=True,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel("Predicted emotion")
plt.ylabel("True emotion")
plt.title("TESS 4o_Audio Synthesized Normalized Confusion Matrix")
plt.tight_layout()

# ★ 保存图片（归一化版）
plt.savefig("/data/zhaohaishu/Demos/TESS_4o_Audio_Synthesized.png", dpi=300, bbox_inches="tight")
plt.show()