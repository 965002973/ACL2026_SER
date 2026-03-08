#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# 1) 进入 Final_mix（你的训练入口 & 数据都在这里）
cd /data/zhaohaishu/Codes/emotion2vec-main/downstream

# 2) 合并后的打包数据前缀（不要带后缀）
feat_path=/data/zhaohaishu/Codes/emotion2vec_upload/downstream/Final_mix_demo/train

python /data/zhaohaishu/Codes/emotion2vec_upload/downstream/final_train.py \
  dataset.feat_path=$feat_path \
  dataset.batch_size=128 \
  optimization.epoch=100 \
  optimization.lr=5e-4 \
  dataset.val_ratio=0.11 \
  dataset.test_ratio=0.0 \
  dataset.eval_is_test=false

