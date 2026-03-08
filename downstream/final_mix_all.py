import numpy as np
from pathlib import Path

# IEMO = Path("/data/zhaohaishu/Codes/emotion2vec-main/IEMOCAP_downstream/train")
# RAV  = Path("/data/zhaohaishu/Codes/emotion2vec-main/RAVDESS_downstream/train")
# SAV  = Path("/data/zhaohaishu/Codes/emotion2vec-main/SAVEE_downstream/train")
# TES  = Path("/data/zhaohaishu/Codes/emotion2vec-main/TESS_downstream/train")
# MEL = Path("/data/zhaohaishu/Codes/emotion2vec-main/MELD_downstream/train")
# RAVS = Path("/data/zhaohaishu/Codes/emotion2vec-main/RAVDESS_SONG_downstream/train")
# CMU = Path("/data/zhaohaishu/Codes/emotion2vec-main/CMU_MOSEI_downstream/train")
# CRE = Path("/data/zhaohaishu/Codes/emotion2vec-main/downstream/Batch_test/crema-d/train")
# IEMO_CV2_SYN = Path("/data/zhaohaishu/Codes/emotion2vec-main/IEMOCAP_CV2_SYN_downstream/train")
# IEMO_KIMI_SYN = Path("/data/zhaohaishu/Codes/emotion2vec-main/IEMOCAP_KIMI_SYN_downstream/train")
# RAV_SYN  = Path("/data/zhaohaishu/Codes/emotion2vec-main/RAVDESS_SYN_downstream/train")
# SAV_SYN  = Path("/data/zhaohaishu/Codes/emotion2vec-main/SAVEE_SYN_downstream/train")
# TES_SYN  = Path("/data/zhaohaishu/Codes/emotion2vec-main/TESS_SYN_downstream/train")
DEMO = Path("/data/zhaohaishu/Codes/emotion2vec_upload/train_downstream_demo/train")

OUT_PREFIX = Path("/data/zhaohaishu/Codes/emotion2vec_upload/downstream/Final_mix_demo/train")
OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)

# ========= 统一标签映射（把三套都映射到同一命名） =========
UNIFY_MAP = {
    "ang": "ang",
    "hap": "hap",
    "neu": "neu",
    "sad": "sad",
    "fea": "fea",   
    "dis": "dis",
    "sur": "sur",
    "angry": "ang",
    "happy": "hap",
    "neutral": "neu",
    "sad": "sad",
    "fearful": "fea",   
    "disgust": "dis",
    "surprised": "sur",
}

def load_dann(path_prefix: Path, unify_map):
    dann_path = path_prefix.with_suffix(".dann")
    items = []
    with open(dann_path, "r", encoding="utf-8") as f:
        for line in f:
            utt, emo, domain = line.strip().split()
            emo = unify_map.get(emo, emo)
            items.append((utt, emo, domain))
    return items

def load_emo(path_prefix: Path, unify_map):
    emo_path = path_prefix.with_suffix(".emo")
    items = []
    with open(emo_path) as f:
        for line in f:
            utt, lab = line.strip().split()
            lab = unify_map.get(lab, lab)
            items.append((utt, lab))
    return items

def load_lengths(path_prefix: Path):
    lens_path = path_prefix.with_suffix(".lengths")
    with open(lens_path) as f:
        lengths = [int(x.strip()) for x in f if x.strip()]
    return lengths

def load_big_npy(path_prefix: Path):
    return np.load(path_prefix.with_suffix(".npy"))

datasets = [
    # ("IEMOCAP", IEMO),
    # ("RAVDESS", RAV),
    # ("SAVEE",   SAV),
    # ("TESS",   TES),
    # ("MELD",  MEL),
    # ("RAVDESS_SONG",  RAVS),
    # ("CMU_MOSEI",  CMU),
    # ("CREMA_D",  CRE),
    # ("IEMOCAP_CV2_SYN", IEMO_CV2_SYN),
    # ("IEMOCAP_KIMI_SYN", IEMO_KIMI_SYN),
    # ("RAVDESS_SYN", RAV_SYN),
    # ("SAVEE_SYN",   SAV_SYN),
    # ("TESS_SYN",   TES_SYN),
    ("DEMO", DEMO)
]

all_items = []
all_lengths = []
all_big = []
all_dann = []

for name, pref in datasets:
    items = load_emo(pref, UNIFY_MAP)
    lengths = load_lengths(pref)
    big = load_big_npy(pref)
    dann = load_dann(pref, UNIFY_MAP)

    assert len(items) == len(lengths), f"{name}: emo lines != lengths lines"
    assert big.shape[0] == sum(lengths), f"{name}: npy frames != sum(lengths)"

    all_items.extend(items)
    all_lengths.extend(lengths)
    all_big.append(big)
    all_dann.extend(dann)

print("total utterances:", len(all_items))
print("total frames:", sum(all_lengths))

mix_big = np.concatenate(all_big, axis=0)
np.save(OUT_PREFIX.with_suffix(".npy"), mix_big)

with open(OUT_PREFIX.with_suffix(".lengths"), "w") as f:
    for L in all_lengths:
        f.write(str(L) + "\n")

with open(OUT_PREFIX.with_suffix(".emo"), "w") as f:
    for utt, lab in all_items:
        f.write(f"{utt}\t{lab}\n")

with open(OUT_PREFIX.with_suffix(".dann"), "w") as f:
    for utt, lab1, lab2 in all_dann:
        f.write(f"{utt}\t{lab1}\t{lab2}\n")

print("wrote:",
      OUT_PREFIX.with_suffix(".npy"),
      OUT_PREFIX.with_suffix(".lengths"),
      OUT_PREFIX.with_suffix(".emo"),
      OUT_PREFIX.with_suffix(".dann")
      )

print("mix_big shape:", mix_big.shape)
