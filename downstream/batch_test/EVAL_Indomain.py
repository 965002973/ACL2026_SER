import sys
from pathlib import Path
import numpy as np
import torch


DOWNSTREAM_DIR = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(DOWNSTREAM_DIR))

from mixdata import load_ssl_features, train_valid_test_dataloader
from model import BaseModel
from utils import validate_and_test

# ====== 你要改的路径 ======
FEAT_PREFIX = "/data/zhaohaishu/Codes/emotion2vec-main/downstream/train"
HEAD_CKPT   = "/data/zhaohaishu/Codes/emotion2vec-main/downstream/outputs/2025-12-01/07-25-14/Human_train_92.pth"

# ====== 必须和训练一致 ======
SEED = 42
BATCH_SIZE = 64
VAL_RATIO = 0.1
TEST_RATIO = 0.1

label_dict = {
    "ang": 0, "hap": 1, "neu": 2, "sad": 3,
    "fea": 4, "dis": 5, "sur": 6,
}
id2label = {v:k for k,v in label_dict.items()}


def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    wa = (y_true == y_pred).mean()

    per_class = {}
    recalls, f1s = [], []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "support": int(support),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
        }
        recalls.append(recall)
        f1s.append(f1)

    ua = float(np.mean(recalls))
    f1_macro = float(np.mean(f1s))
    return wa, ua, f1_macro, per_class


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_ssl_features(FEAT_PREFIX, label_dict)

    train_loader, val_loader, test_loader = train_valid_test_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    model = BaseModel(input_dim=768, output_dim=len(label_dict)).to(device)
    model.load_state_dict(torch.load(HEAD_CKPT, map_location=device), strict=True)
    model.eval()

    # 4) 逐 batch 收集预测
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["net_input"]["feats"].to(device)              # (B,T,768)
            padding_mask = batch["net_input"]["padding_mask"].to(device)
            labels = batch["labels"].cpu().numpy().tolist()

            try:
                logits = model(feats, padding_mask)
            except TypeError:
                logits = model(feats)

            preds = logits.argmax(dim=-1).cpu().numpy().tolist()

            y_true.extend(labels)
            y_pred.extend(preds)

    wa, ua, f1, per_class = compute_metrics(y_true, y_pred, num_classes=len(label_dict))

    print("=== Intra-domain Test (Final_mix split) ===")
    print(f"WA (acc): {wa*100:.2f}%")
    print(f"UA (mean recall): {ua*100:.2f}%")
    print(f"F1 (macro): {f1*100:.2f}%")

    print("\nPer-class results:")
    for cid, stats in per_class.items():
        name = id2label[cid]
        sup = stats["support"]
        rec = stats["recall"] * 100
        pre = stats["precision"] * 100
        f1c = stats["f1"] * 100
        print(f"{name:>3} | support={sup:5d} | recall(acc)={rec:6.2f}% | precision={pre:6.2f}% | f1={f1c:6.2f}%")


if __name__ == "__main__":
    main()
