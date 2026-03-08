#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Any, Dict, List
from pathlib import Path
import torch
import torch.nn.functional as F
import fairseq
import argparse
import tempfile
import subprocess
from dataclasses import dataclass
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


# ----------------- emotion2vec loader -----------------
@dataclass
class UserDirModule:
    user_dir: str


def load_emotion2vec(model_dir: str, checkpoint_dir: str):
    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = models[0]
    model.eval()
    model.cuda()
    return model, task


# ----------------- audio IO (wav/mp4, auto resample) -----------------
def extract_wav_from_mp4(mp4_path: Path, target_sr: int):
    tmp_wav = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        tmp_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp_wav


def read_wav_auto_resample(wav_path: Path, target_sr: int):
    wav, sr = sf.read(str(wav_path), always_2d=False)
    info = sf.info(str(wav_path))
    ch = info.channels

    # to mono
    if ch > 1:
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        else:
            wav = librosa.to_mono(wav.T)

    # resample
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return wav, sr


def read_audio_auto_resample(audio_path: Path, target_sr: int):
    suf = audio_path.suffix.lower()
    if suf == ".wav":
        return read_wav_auto_resample(audio_path, target_sr)
    if suf == ".mp4":
        tmp_wav = extract_wav_from_mp4(audio_path, target_sr)
        try:
            wav, sr = read_wav_auto_resample(Path(tmp_wav), target_sr)
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        return wav, sr
    raise ValueError(f"Unsupported audio format: {suf}")


# ----------------- feature extraction -----------------
@torch.no_grad()
def extract_utterance_embedding(model, task, audio_path: Path, target_sr: int = 16000):
    wav, sr = read_audio_auto_resample(audio_path, target_sr=target_sr)

    source = torch.from_numpy(wav).float().cuda()
    if getattr(task.cfg, "normalize", False):
        source = F.layer_norm(source, source.shape)
    source = source.view(1, -1)

    feats = model.extract_features(source, padding_mask=None)
    x = feats["x"].squeeze(0)  # (T, D)

    # utterance embedding: mean pooling
    emb = x.mean(dim=0).cpu().numpy()  # (D,)
    return emb.astype(np.float32)


# ----------------- utils -----------------
def load_json_list(p: str):
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("json 顶层必须是 list")
    return data


def normalize_records(obj):
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ["data", "items", "samples", "utterances", "records", "audios", "metadata"]:
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        if "wav_path" in obj:
            return [obj]
    return []

# ----------------- 解析标签 -----------------
def parse_dataset_id(ds_id: str):
    """
    兼容你的命名：NameOfDataset-Human/SYN-XXX-TRAIN/TEST...
    也兼容你目前的下划线写法：TESS_4o_TTS_SYN_TEST 这类
    """
    tokens = ds_id.split('_')

    # dataset_name: 尽量取最前面的那段（你也可以按你的真实规范更严格拆）
    dataset_name = tokens[0]

    # source: Human / SYN
    if tokens[1].lower() == "human":
        source = "Human" 
        model = 'Human'
    elif tokens[1].lower() == "syn":
        source = "SYN"
        model = tokens[2]
    else:
        source = "UNK"
        model = "UNK"

    # vocoder_group：按你给的规则
    sm = model.lower()
    if source == "Human":
        vocoder_group = "Human"
    elif "kimicosy" in sm:
        vocoder_group = "KIMICOSY"
    elif "kimi" in sm or "index" in sm:
        vocoder_group = "BigVGAN"
    elif "cosy" in sm or "glm" in sm:
        vocoder_group = "HifiGAN"
    elif "4o-tts" in sm or "4o-audio" in sm:
        vocoder_group = "gpt-4o"
    else:
        vocoder_group = "Other"

    return {
        "dataset_name": dataset_name,
        "source": source,       
        "synth_model": model,    
        "vocoder_group": vocoder_group,
    }

# ----------------- t-SNE降维与可视化 -----------------
def visualize_tsne(all_encoded_data, all_emotions, all_dataset_name, all_source, all_model_name, all_vocoder, category, fig_path):
    """
    使用t-SNE对编码后的数据降维，并根据指定的分类方式进行可视化
    """
    # 将数据降至2维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(np.vstack(all_encoded_data))

    category_map = {
        'emotion': all_emotions,
        'dataset_name': all_dataset_name,
        'synthesis': all_source,
        'model_name': all_model_name,
        'vocoder': all_vocoder
    }
    # choices = [emotion, dataset_name, synthesis, model_name, vocoder, emotion&synthesis]
    if "&" in category:
        c1, c2 = category.split('&')
        categories1 = category_map.get(c1)
        categories2 = category_map.get(c2)
        if categories1 == None or categories2 == None:
            print("WRONG Category")
            return
        
        emotion_labels = list(set(categories1))  # 获取唯一的情绪标签
        emotion_to_num = {emotion: idx for idx, emotion in enumerate(emotion_labels)}  # 将情绪标签映射为数字
        categories_numeric = [emotion_to_num[emotion] for emotion in categories1] # 将情绪标签转换为数字

        palette = sns.color_palette("husl", len(emotion_labels))
        emotion_color_map = {emo: palette[i % len(palette)] for i, emo in enumerate(emotion_labels)}
        
        category_to_marker = {
            "Human": 'o',
            "SYN": 'x'
        }

        plt.figure(figsize=(10, 8))
        for i in range(len(reduced_data)):
            plt.scatter(
                reduced_data[i, 0], reduced_data[i, 1],
                c=[emotion_color_map[categories1[i]]],  # 使用颜色映射
                marker=category_to_marker[categories2[i]],  # 使用形状映射
                alpha=0.7,
                s=20    # 符号大小
            )
        handles = []
        labels_map = []
        for emo in emotion_labels:
            # 获取该情绪对应的颜色
            color = emotion_color_map[emo]
            
            # 创建一个假的图例句柄：强制形状为 'o' (圆)，颜色对应情绪
            # Line2D(x, y, marker=..., color='w', markerfacecolor=..., label=...)
            handle = Line2D([0], [0], marker='o', color='w', label=emo,
                            markerfacecolor=color, markersize=10)
            handles.append(handle)
            labels_map.append(emo)
        
    else:
        categories = category_map.get(category)
        if categories == None:
            print("WRONG Category")
            return
        
        emotion_labels = list(set(categories))  # 获取唯一的情绪标签
        if 'Human' in emotion_labels:
            emotion_labels.remove('Human')
            emotion_labels.insert(0, 'Human')
        emotion_to_num = {emotion: idx for idx, emotion in enumerate(emotion_labels)}  # 将情绪标签映射为数字
        categories_numeric = [emotion_to_num[emotion] for emotion in categories] # 将情绪标签转换为数字

        palette = sns.color_palette("husl", len(emotion_labels))
        emotion_color_map = {emo: palette[i % len(palette)] for i, emo in enumerate(emotion_labels)}

        plt.figure(figsize=(10, 8))
        for cat in emotion_labels:
            # 找出属于该类别的索引
            indices = [i for i, x in enumerate(categories) if x == cat]
            if not indices: continue
            
            # 提取数据
            data_subset = reduced_data[indices]
            color = emotion_color_map[cat]
            
            plt.scatter(
                data_subset[:, 0], data_subset[:, 1], 
                c=[color], 
                label=cat,
                alpha=0.8 if cat=='human' else 0.7, 
                s=10
            )
        
        handles = []
        labels_map = []
        for emo in emotion_labels:
            # 获取该情绪对应的颜色
            color = emotion_color_map[emo]
            
            # 创建一个假的图例句柄：强制形状为 'o' (圆)，颜色对应情绪
            # Line2D(x, y, marker=..., color='w', markerfacecolor=..., label=...)
            handle = Line2D([0], [0], marker='o', color='w', label=emo,
                            markerfacecolor=color, markersize=10)
            handles.append(handle)
            labels_map.append(emo)
    
    plt.legend(handles, labels_map)
    # plt.colorbar(scatter)
    
    plt.title(f"t-SNE Visualization of Audio Features ({category})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    
    plt.savefig(fig_path, dpi=300)
    print(f"[DONE] saved: {fig_path}")

# ----------------- 主要逻辑 -----------------
def main():
    DATASETS_JSON = "/data/zhaohaishu/Datasets/dataset.json"
    # 数据集使用
    # choices = ["IEMOCAP_Human_TEST", "MELD_Human_TEST", "RAVDESS_Human_TEST", "SONG_Human_TEST", "SAVEE_Human_TEST", "TESS_Human_TEST", 
    #            "IEMOCAP_SYN_KIMICOSY_TRAIN", "IEMOCAP_SYN_KIMICOSY_TEST", "TESS_SYN_COSY_TRAIN", "TESS_SYN_COSY_TEST", "TESS_SYN_KIMI", 
    #            "TESS_SYN_GLM", "TESS_SYN_4o-TTS", "TESS_SYN_4o-Audio", "CREMA-D_Human_TRAIN", "CREMA-D_Human_TEST", "CREMA-D_Human_ALL", 
    #            "CREMA-D_SYN_COSY", "CREMA-D_SYN_KIMI", "CREMA-D_SYN_INDEX", "CREMA-D_SYN_GLM", "CREMA-D_SYN_4o-TTS", "CREMA-D_SYN_4o-Audio"]  
    SELECTED_DATASETS = ["TESS_Human_ALL", "TESS_SYN_COSY_TEST", "TESS_SYN_4o-TTS",
                         "CREMA-D_Human_TEST", "CREMA-D_SYN_COSY", "CREMA-D_SYN_KIMI", "CREMA-D_SYN_INDEX", 
                         "CREMA-D_SYN_GLM", "CREMA-D_SYN_4o-TTS", "CREMA-D_SYN_4o-Audio"]  # Specify the datasets to analyze

    # , "TESS_SYN_GLM", "TESS_SYN_4o-Audio", "TESS_SYN_KIMI"


    # SELECTED_DATASETS = ["IEMOCAP_Human_TEST", "MELD_Human_TEST", "RAVDESS_Human_TEST", "SONG_Human_TEST", "SAVEE_Human_TEST", 
    #                      "TESS_Human_TEST", "CREMA-D_Human_TEST"]
    # SELECTED_DATASETS = ["IEMOCAP_Human_TEST", "MELD_Human_TEST", "RAVDESS_Human_TEST", "SONG_Human_TEST", "SAVEE_Human_TEST", 
    #                      "TESS_Human_TEST", "CREMA-D_Human_TEST", "CHinese_Multi_Speaker"]
    # SELECTED_DATASETS = ["TESS_Human_TEST", "TESS_SYN_4o-TTS"]
    # # , "TESS_SYN_COSY_TEST", "TESS_SYN_KIMI", "TESS_SYN_GLM", "TESS_SYN_4o-Audio"
    # SELECTED_DATASETS = ["CREMA-D_Human_TEST", "", "CREMA-D_SYN_COSY", "CREMA-D_SYN_KIMI", "CREMA-D_SYN_INDEX", "CREMA-D_SYN_GLM", 
    #                      "CREMA-D_SYN_4o-TTS", "CREMA-D_SYN_4o-Audio", "RAVDESS_Human_TEST"]

    # 可视化分类
    # choices = [emotion, dataset_name, synthesis, model_name, vocoder, emotion&synthesis]
    category = "model_name"
    file_path = '/data/zhaohaishu/Codes/emotion2vec-main/embedding_visualization/gy_visualize'
    fig_path = file_path + '/demo.pdf'
    # Load Emotion2Vec model
    model_dir = "/data/zhaohaishu/Codes/emotion2vec-main/upstream"  
    checkpoint_dir = "/data/zhaohaishu/Models/emotion2vec_base/emotion2vec_base.pt" 
    model, task = load_emotion2vec(model_dir, checkpoint_dir)

    datasets_obj = load_json_list(DATASETS_JSON)
    all_encoded_data = []
    all_emotions = []
    all_dataset_name = []
    all_source = []      
    all_model_name = []
    all_vocoder = []

    # 处理选定的数据集
    for i, x in enumerate(datasets_obj):
        if x.get("id") not in SELECTED_DATASETS:
            continue

        ds_id = x.get("id")
        meta_path = x.get("json")
        if not ds_id or not meta_path:
            print("ID", x, "NOT EXIST")
            continue

        if not os.path.exists(meta_path):
            print("Path", meta_path, "NOT EXIST")
            continue

        try:
            meta_obj = load_json_list(meta_path)
        except Exception as e:
            continue

        records = normalize_records(meta_obj)
        if len(records) == 0:
            continue

        for ridx, r in enumerate(records):
            wav_path = r.get("response_wav_path") or r.get("wav_path") or r.get("path") or r.get("tts_wav_path") or r.get("test_path")
            if not wav_path:
                continue

            wav_path = str(wav_path)

            # 提取音频特征
            try:
                emb = extract_utterance_embedding(model, task, Path(wav_path), target_sr=16000)
                all_encoded_data.append(emb)
            except Exception as e:
                continue

            # 获取情绪和标签
            emotion = r.get("emotion")
            if emotion:
                all_emotions.append(emotion)
                labels = parse_dataset_id(ds_id)
                all_dataset_name.append(labels["dataset_name"])
                all_source.append(labels["source"])
                all_model_name.append(labels["synth_model"])
                all_vocoder.append(labels["vocoder_group"])
    
    print("Length:", len(all_encoded_data))
    if len(all_encoded_data) == 0:
        print("NO Dataset")
        return 
    # 可视化数据
    visualize_tsne(all_encoded_data, all_emotions, all_dataset_name, all_source, all_model_name, all_vocoder, category, fig_path)

if __name__ == "__main__":
    main()

