#!/bin/bash

python /data/zhaohaishu/Codes/emotion2vec-main/scripts/json_to_emo.py \
  --in_json /data/zhaohaishu/Codes/emotion2vec_upload/train_tess_downstream_demo/tess_json.json \
  --out_emo /data/zhaohaishu/Codes/emotion2vec_upload/train_tess_downstream_demo/train.emo \
  --path_key wav_path \
  --emo_key emotion

