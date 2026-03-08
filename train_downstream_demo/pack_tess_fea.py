import os
from pathlib import Path
import numpy as np

FEAT_DIR = "/data/zhaohaishu/Codes/emotion2vec_upload/train_downstream_demo/features"
EMO_FILE = "/data/zhaohaishu/Codes/emotion2vec_upload/train_downstream_demo/train.emo"
OUT_PREFIX = "/data/zhaohaishu/Codes/emotion2vec_upload/train_downstream_demo/train"

feat_dir = Path(FEAT_DIR)

# 1) 递归收集所有 npy，按 stem 建字典
feat_map = {}
for p in feat_dir.rglob("*.npy"):
    feat_map[p.stem] = p   # key=03-01-06-01-02-01-12

print("found npy:", len(feat_map))

# 2) 读 emo（保持顺序）
items = []
with open(EMO_FILE) as f:
    for line in f:
        utt, lab = line.strip().split()
        items.append((utt, lab))

all_feats, lengths, kept = [], [], []
missing = 0

for utt, lab in items:
    if utt not in feat_map:
        missing += 1
        continue

    feat = np.load(feat_map[utt])
    if feat.ndim == 1:
        feat = feat[None, :]  # utterance-level -> (1, D)

    T = feat.shape[0]
    all_feats.append(feat)
    lengths.append(T)
    kept.append((utt, lab))

print("missing feats:", missing)
print("kept feats:", len(kept))

if len(all_feats) == 0:
    raise RuntimeError(
        "No features matched train.emo. "
        "Check FEAT_DIR and EMO_FILE naming."
    )

# 3) 拼大矩阵
big = np.concatenate(all_feats, axis=0)
np.save(OUT_PREFIX + ".npy", big)

with open(OUT_PREFIX + ".lengths", "w") as f:
    for T in lengths:
        f.write(str(T) + "\n")

with open(OUT_PREFIX + ".emo", "w") as f:
    for utt, lab in kept:
        f.write(f"{utt}\t{lab}\n")

print("big shape:", big.shape)
print("wrote:",
      OUT_PREFIX + ".npy",
      OUT_PREFIX + ".lengths",
      OUT_PREFIX + ".emo")
