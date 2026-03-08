import sys
from pathlib import Path
DOWNSTREAM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOWNSTREAM_DIR))

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn.functional as F
import fairseq

from model import EmotionDANN

JSON_FILE = "/data/zhaohaishu/Codes/emotion2vec_upload/downstream/batch_test/test_demo/tess_test.json"  
EMO2VEC_MODEL_DIR = "/data/zhaohaishu/Codes/emotion2vec-main/upstream"
EMO2VEC_CKPT = "/data/zhaohaishu/Models/emotion2vec_base/emotion2vec_base.pt"
HEAD_CKPT = "/data/zhaohaishu/Codes/emotion2vec-main/downstream/outputs/DANN_and_probe/05-04-50-adversarial_syn/model_best.pth"
TARGET_SR = 16000
GRANULARITY = "frame"  

# 训练时的 7 类
label_dict = {
    "ang": 0, "hap": 1, "neu": 2, "sad": 3,
    "fea": 4, "dis": 5, "sur": 6,
}
id2label = {v: k for k, v in label_dict.items()}

MAP = {
    "neu": "neu",
    "hap": "hap",
    "sad": "sad",
    "ang": "ang",      
    "fea": "fea",
    "dis": "dis",
    "sur": "sur",
    "neutral": "neu",
    "happy": "hap",
    "joy": "hap",
    "sadness": "sad",
    "angry": "ang",    
    "anger": "ang",       
    "fear": "fea",
    "fearful": "fea",
    "disgust": "dis",
    "disgusted": "dis",
    "surprised": "sur",
    "surprise": "sur",
}

@dataclass
class UserDirModule:
    user_dir: str


def load_json_items(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list), "你的json是 list[dict]，这里应该就是list"
    return data


def parse_item(item):
    wav_path = item["test_path"]
    emo_str = item["emotion"].lower().strip()

    if emo_str not in MAP:
        raise ValueError(f"Unknown TESS emotion: {emo_str}")

    short = MAP[emo_str]
    label_id = label_dict[short]
    return wav_path, label_id


def load_emotion2vec(model_dir, ckpt):
    fairseq.utils.import_user_module(UserDirModule(model_dir))
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
    model = models[0].eval().cuda()
    return model, task


def read_wav_auto(path, target_sr=16000):
    wav, sr = sf.read(path, always_2d=False)
    info = sf.info(path)

    # 多声道->单声道
    if info.channels > 1:
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        else:
            wav = librosa.to_mono(wav.T)

    # 重采样
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    return wav


def extract_feats(wav_path, emo2vec, task, granularity="frame"):
    wav = read_wav_auto(wav_path, TARGET_SR)

    with torch.no_grad():
        x = torch.from_numpy(wav).float().cuda()
        if getattr(task.cfg, "normalize", False):
            x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)

        feats = emo2vec.extract_features(x, padding_mask=None)["x"].squeeze(0)  # (T,768)

        if granularity == "utterance":
            feats = feats.mean(dim=0, keepdim=True)  # (1,768)

    return feats


def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # overall accuracy
    wa = (y_true == y_pred).mean()

    per_class = {}
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # 每类准确度(召回)
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

    ua = float(np.mean(recalls))       # mean recall
    f1_macro = float(np.mean(f1s))     # macro f1

    return wa, ua, f1_macro, per_class

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_json_items(JSON_FILE)
    pairs = [parse_item(it) for it in items]
    print("num test items:", len(pairs))

    emo2vec, task = load_emotion2vec(EMO2VEC_MODEL_DIR, EMO2VEC_CKPT)

    # num_domains 要和你训练时一致（通常 2：Real/Synth）
    head = EmotionDANN(
        input_dim=768,
        num_emotions=len(label_dict),
        num_domains=2
    ).to(device)

    ckpt = torch.load(HEAD_CKPT, map_location=device)
    head.load_state_dict(ckpt, strict=True)  # 若你结构有变可先 strict=False
    head.eval()

    y_true, y_pred = [], []

    for wav_path, gt_id in pairs:
        if not Path(wav_path).exists():
            print("[skip missing]", wav_path)
            continue

        feats = extract_feats(wav_path, emo2vec, task, granularity=GRANULARITY)  # (T,768) or (1,768)

        # 保证输入是 (B,T,768)
        if feats.dim() == 2:      # (T,768)
            feats = feats.unsqueeze(0)  # (1,T,768)
        elif feats.dim() == 3:    # already (1,T,768)
            pass
        else:
            raise ValueError(f"Unexpected feats shape: {feats.shape}")

        padding_mask = torch.zeros((feats.shape[0], feats.shape[1]), dtype=torch.bool, device=device)

        with torch.no_grad():
            # forward 返回两个分支
            emotion_logits, domain_logits = head(feats.to(device), padding_mask, alpha=0.0)

            pred_id = emotion_logits.argmax(dim=-1).item()

        y_true.append(gt_id)
        y_pred.append(pred_id)

        print(Path(wav_path).name, "pred=", id2label[pred_id], "gt=", id2label[gt_id])

    wa, ua, f1, per_class = compute_metrics(y_true, y_pred, num_classes=len(label_dict))
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
        print(f"{name:>3} | support={sup:4d} | recall(acc)={rec:6.2f}% | precision={pre:6.2f}% | f1={f1c:6.2f}%")

if __name__ == "__main__":
    main()