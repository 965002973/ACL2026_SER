# #!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# cd /data/zhaohaishu/Codes/emotion2vec-main
# feat_path=/data/zhaohaishu/Codes/emotion2vec-main/MIX_downstream/train

# python /data/zhaohaishu/Codes/emotion2vec-main/downstream/Final_mix/final_train.py
#     dataset.feat_path=$feat_path \
#     model._name=BaseModel \
#     dataset.batch_size=128 \
#     optimization.epoch=100 \
#     optimization.lr=5e-4 \
#     dataset.eval_is_test=false \
#     dataset.val_ratio=0.1 \
#     dataset.test_ratio=0.1

#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# 1) 进入 Final_mix（你的训练入口 & 数据都在这里）
cd /data/zhaohaishu/Codes/emotion2vec-main/downstream

# 2) 合并后的打包数据前缀（不要带后缀）
feat_path=/data/zhaohaishu/Codes/emotion2vec-main/Probe_emotion_V2/train

python /data/zhaohaishu/Codes/emotion2vec-main/downstream/probe_train_emotion.py \
  dataset.feat_path=$feat_path \
  dataset.batch_size=128 \
  optimization.epoch=100 \
  optimization.lr=5e-4 \
  dataset.val_ratio=0.11 \
  dataset.test_ratio=0.0 \
  dataset.eval_is_test=false

