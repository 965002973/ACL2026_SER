export CUDA_VISIBLE_DEVICES=7

python /data/zhaohaishu/Codes/emotion2vec-main/scripts/extract_features_batch.py \
  --source_dir="/data/zhaohaishu/Codes/AI-Synthesized-Voice-Generalization-main/training/test" \
  --target_dir="/data/zhaohaishu/Codes/AI-Synthesized-Voice-Generalization-main/training/test/features" \
  --model_dir="/data/zhaohaishu/Codes/emotion2vec-main/upstream" \
  --checkpoint_dir="/data/zhaohaishu/Models/emotion2vec_base/emotion2vec_base.pt" \
  --granularity="frame" \
  --recursive \
  --target_sr=16000 \
  --ext wav

# export CUDA_VISIBLE_DEVICES=0

# python /data/zhaohaishu/Codes/emotion2vec-main/scripts/extract_features_batch.py \
#   --source_dir /data/zhaohaishu/Dataset_syn/MELD/train_splits \
#   --target_dir /data/zhaohaishu/Codes/emotion2vec-main/MELD_downstream/features \
#   --model_dir /data/zhaohaishu/Codes/emotion2vec-main/upstream \
#   --checkpoint_dir /data/zhaohaishu/Models/emotion2vec_base/emotion2vec_base.pt \
#   --granularity frame \
#   --recursive \
#   --ext mp4


#!/bin/bash
# set -euo pipefail

# SRC_BASE="/data/haoaokai/gao/Another/datasets/25/12/17"
# TGT_BASE="/data/zhaohaishu/Codes/emotion2vec-main/downstream/generate_indextts2"
# MODEL_DIR="/data/zhaohaishu/Codes/emotion2vec-main/upstream"
# CKPT="/data/zhaohaishu/Models/emotion2vec_base/emotion2vec_base.pt"

# run_one () {
#   local gpu="$1"
#   local shard="$2"
#   echo "[GPU ${gpu}] start shard_${shard}"

#   CUDA_VISIBLE_DEVICES="${gpu}" \
#   python /data/zhaohaishu/Codes/emotion2vec-main/scripts/extract_features_batch.py \
#     --source_dir="${SRC_BASE}/shard_${shard}" \
#     --target_dir="${TGT_BASE}/shard_${shard}/features" \
#     --model_dir="${MODEL_DIR}" \
#     --checkpoint_dir="${CKPT}" \
#     --granularity="frame" \
#     --recursive \
#     --target_sr=16000 \
#     --ext wav

#   echo "[GPU ${gpu}] done  shard_${shard}"
# }

# for s in 00 04 08 12 16; do run_one 4 "$s" & done
# for s in 01 05 09 13 17; do run_one 5 "$s" & done
# for s in 02 06 10 14 18; do run_one 6 "$s" & done
# for s in 03 07 11 15 19; do run_one 7 "$s" & done

# wait
# echo "All shards done."
