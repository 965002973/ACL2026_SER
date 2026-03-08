import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import os
import subprocess
import tempfile

import torch
import torch.nn.functional as F
import fairseq


def get_parser():
    parser = argparse.ArgumentParser(
        description="extract emotion2vec features for downstream tasks (batch, auto-resample, mp4 supported)"
    )
    parser.add_argument('--source_dir', help='directory of source audio files (wav/mp4)', required=True)
    parser.add_argument('--target_dir', help='directory to save target npy files', required=True)
    parser.add_argument('--model_dir', type=str, help='pretrained model user dir', required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', required=True)
    parser.add_argument('--granularity', type=str, choices=['frame', 'utterance'],
                        help='which granularity to use, frame or utterance', required=True)

    parser.add_argument('--recursive', action='store_true',
                        help='recursively search audio files under source_dir')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing npy files')

    parser.add_argument('--target_sr', type=int, default=16000,
                        help='target sample rate (default: 16000)')
    
    parser.add_argument('--ext', type=str, default="mp4",
                        help='input extension: mp4 or wav or "all" (default: mp4)')

    return parser


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
    # 静默执行
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp_wav


def read_wav_auto_resample(wav_path: Path, target_sr: int):
    wav, sr = sf.read(str(wav_path), always_2d=False)
    info = sf.info(str(wav_path))
    channel = info.channels

    if channel > 1:
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        else:
            wav = librosa.to_mono(wav.T)

    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return wav, sr


def read_audio_auto_resample(audio_path: Path, target_sr: int):
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        return read_wav_auto_resample(audio_path, target_sr=target_sr)

    if suffix == ".mp4":
        tmp_wav = extract_wav_from_mp4(audio_path, target_sr=target_sr)
        try:
            wav, sr = read_wav_auto_resample(Path(tmp_wav), target_sr=target_sr)
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        return wav, sr

    raise ValueError(f"Unsupported audio format: {suffix}")


def extract_one(model, task, audio_path: Path, granularity: str, target_sr: int):
    wav, sr = read_audio_auto_resample(audio_path, target_sr=target_sr)

    with torch.no_grad():
        source = torch.from_numpy(wav).float().cuda()
        if getattr(task.cfg, "normalize", False):
            source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)

        feats = model.extract_features(source, padding_mask=None)
        feats = feats["x"].squeeze(0).cpu().numpy()  # (T, D)

        if granularity == "utterance":
            feats = np.mean(feats, axis=0)  # (D,)

        return feats


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.ext.lower() == "all":
        patterns = ["**/*.mp4", "**/*.wav"] if args.recursive else ["*.mp4", "*.wav"]
        audio_files = []
        for pat in patterns:
            audio_files.extend(list(source_dir.glob(pat)))
        audio_files = sorted(audio_files)
    else:
        ext = args.ext.lower().lstrip(".")
        pattern = f"**/*.{ext}" if args.recursive else f"*.{ext}"
        audio_files = sorted(source_dir.glob(pattern))

    if len(audio_files) == 0:
        raise RuntimeError(f"No audio files found in {source_dir} (ext={args.ext}, recursive={args.recursive})")

    model, task = load_emotion2vec(args.model_dir, args.checkpoint_dir)

    for audio_path in audio_files:
        rel = audio_path.relative_to(source_dir)
        out_path = (target_dir / rel).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} exists")
            continue

        try:
            feats = extract_one(
                model, task, audio_path,
                granularity=args.granularity,
                target_sr=args.target_sr
            )
            print(type(feats), feats.shape)
            np.save(str(out_path), feats)
            print(f"[ok] {audio_path} -> {out_path}")
        except Exception as e:
            print(f"[error] {audio_path}: {e}")


if __name__ == '__main__':
    main()
