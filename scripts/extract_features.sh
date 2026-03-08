#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python /data/zhaohaishu/Codes/emotion2vec-main/scripts/extract_features.py  \
    --source_file='/data/zhaohaishu/Dataset_syn/MELD/train_splits/dia125_utt3.mp4' \
    --target_file='//data/zhaohaishu/Codes/emotion2vec-main/MELD_downstream/features/dia125_utt3.npy' \
    --model_dir='/data/zhaohaishu/Codes/emotion2vec-main/upstream' \
    --checkpoint_dir='/data/zhaohaishu/Models/emotion2vec_base/emotion2vec_base.pt' \
    --granularity='utterance' \
