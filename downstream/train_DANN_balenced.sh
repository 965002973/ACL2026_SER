#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

cd /data/zhaohaishu/Codes/emotion2vec-main/downstream

# 2) 合并后的打包数据前缀（不要带后缀）
feat_path=/data/zhaohaishu/Codes/emotion2vec-main/downstream/Final_mix_all_DANN/train

python train_Domain_Adversarial_balenced_01.py \
  dataset.feat_path=$feat_path \
  dataset.batch_size=128 \
  optimization.epoch=100 \
  optimization.lr=5e-4 \
  optimization.max_alpha=1 \
  dataset.val_ratio=0.1 \
  dataset.test_ratio=0.0 \
  dataset.eval_is_test=false\
  model.adversarial_domain=model_dict


# adversarial_domain Choices: [syn_dict, vocoder_dict, model_dict]