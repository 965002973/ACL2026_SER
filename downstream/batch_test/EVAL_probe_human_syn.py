import sys
from pathlib import Path
DOWNSTREAM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOWNSTREAM_DIR))

import json
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn.functional as F
import fairseq

from model import BaseModel

JSON_FILE = "/data/zhaohaishu/Datasets/GPT_test_data/Crema_D_probe/probe_train_V2/probe_test_V2.json"
EMO2VEC_MODEL_DIR = "/data/zhaohaishu/Codes/emotion2vec-main/upstream"
EMO2VEC_CKPT = "/data/zhaohaishu/Models/emotion2vec_base/emotion2vec_base.pt"
HEAD_CKPT = "/data/zhaohaishu/Codes/emotion2vec-main/downstream/outputs/2025-12-30/probe_train_human_syn/model_best.pth"

TARGET_SR = 16000
GRANULARITY = "frame"  # 必须和训练一致

label_dict = {
        "human": 0,
        "syn": 1,
    }

id2label = {v: k for k, v in label_dict.items()}

MAP = {
    "human": "human",
    "syn" : "syn",
}


@dataclass
class UserDirModule:
    user_dir: str


def load_json_items(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list)
    return data


def parse_item(item):
    wav_path = item["test_path"] 
    emo = item["data_type"].lower().strip()
    if emo not in MAP:
        raise ValueError(f"Unknown emotion: {emo}")
    label = label_dict[MAP[emo]]
    return wav_path, label


def load_emotion2vec(model_dir, ckpt):
    fairseq.utils.import_user_module(UserDirModule(model_dir))
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
    model = models[0].eval().cuda()
    return model, task


def read_wav_auto(path, target_sr=16000):
    wav, sr = sf.read(path, always_2d=False)
    info = sf.info(path)

    if info.channels > 1:
        wav = wav.mean(axis=1) if wav.ndim == 2 else librosa.to_mono(wav.T)

    if sr != target_sr:
        wav = librosa.resample(
            wav.astype(np.float32),
            orig_sr=sr,
            target_sr=target_sr
        )

    return wav


def extract_feats(wav_path, emo2vec, task, granularity="frame"):
    wav = read_wav_auto(wav_path, TARGET_SR)

    with torch.no_grad():
        x = torch.from_numpy(wav).float().cuda()
        if getattr(task.cfg, "normalize", False):
            x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)

        feats = emo2vec.extract_features(x, padding_mask=None)["x"].squeeze(0)

        if granularity == "utterance":
            feats = feats.mean(dim=0, keepdim=True)

    return feats


# ====== 只保留 Balanced Acc + Macro F1 ======
def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    recalls, f1s = [], {}
    per_class = {}

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        recalls.append(recall)
        f1s[c] = f1

        per_class[c] = {
            "support": int(support),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
        }

    balanced_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(list(f1s.values())))

    return balanced_acc, macro_f1, per_class


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_json_items(JSON_FILE)
    pairs = [parse_item(it) for it in items]
    print("num test items:", len(pairs))

    emo2vec, task = load_emotion2vec(EMO2VEC_MODEL_DIR, EMO2VEC_CKPT)

    head = BaseModel(input_dim=768, output_dim=len(label_dict)).to(device)
    head.load_state_dict(torch.load(HEAD_CKPT, map_location=device), strict=True)
    head.eval()

    y_true, y_pred = [], []

    for wav_path, gt_id in pairs:
        if not Path(wav_path).exists():
            print("[skip missing]", wav_path)
            continue

        feats = extract_feats(wav_path, emo2vec, task, granularity=GRANULARITY)
        feats = feats.unsqueeze(0)
        padding_mask = torch.zeros((1, feats.shape[1]), dtype=torch.bool, device=device)

        with torch.no_grad():
            try:
                logits = head(feats, padding_mask)
            except TypeError:
                logits = head(feats)
            pred_id = logits.argmax(dim=-1).item()

        y_true.append(gt_id)
        y_pred.append(pred_id)

        print(Path(wav_path).name, "pred=", id2label[pred_id], "gt=", id2label[gt_id])

    balanced_acc, macro_f1, per_class = compute_metrics(
        y_true, y_pred, num_classes=len(label_dict)
    )

    print(f"\nBalanced Accuracy (UA): {balanced_acc*100:.2f}%")
    print(f"Macro F1: {macro_f1*100:.2f}%")

    print("\nPer-class results:")
    for cid, stats in per_class.items():
        name = id2label[cid]
        print(
            f"{name:>10} | support={stats['support']:4d} | "
            f"recall={stats['recall']*100:6.2f}% | "
            f"precision={stats['precision']*100:6.2f}% | "
            f"f1={stats['f1']*100:6.2f}%"
        )


if __name__ == "__main__":
    main()
