#!/bin/bash

# 定义参数列表
alphas=(1 0.1 0.01)
lrs=(5e-4 1e-4)

# 定义你想用的显卡ID列表
gpus=(2 3 4 5 6)
num_gpus=${#gpus[@]} # 获取显卡数量

cd /data/zhaohaishu/Codes/emotion2vec-main/downstream
feat_path=/data/zhaohaishu/Codes/emotion2vec-main/downstream/Final_mix_all_DANN/train
current_date=$(date +%Y-%m-%d)
counter=0

# 双重循环遍历所有组合
for alpha in "${alphas[@]}"; do
  for lr in "${lrs[@]}"; do
    
    if [[ "$alpha" == "1" && "$lr" == "5e-4" ]]; then
        echo ">>> 跳过配置: alpha=$alpha, lr=$lr"
        continue  
    fi

    # 2. 计算当前任务该分配给哪张卡 (取余数算法)
    gpu_index=$((counter % num_gpus))
    current_gpu=${gpus[$gpu_index]}

    exp_name="alpha_${alpha}_lr_${lr}"
    save_dir="outputs/${current_date}/${exp_name}"

    echo "启动任务 $counter: GPU=$current_gpu | alpha=$alpha | lr=$lr"

    CUDA_VISIBLE_DEVICES=$current_gpu python train_Domain_Adversarial.py \
      dataset.feat_path=$feat_path \
      dataset.batch_size=128 \
      optimization.epoch=100 \
      optimization.lr=$lr \
      optimization.max_alpha=$alpha \
      dataset.val_ratio=0.1 \
      dataset.test_ratio=0.0 \
      dataset.eval_is_test=false\
      model.adversarial_domain=syn_dict \
      hydra.job.chdir=True \
      hydra.run.dir=$save_dir &

    counter=$((counter+1))
    
    sleep 200
      
  done
done

wait
echo "所有任务已完成！"