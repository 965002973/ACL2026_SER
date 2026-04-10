<h1 align="center"> On the Emotion Understanding of Synthesized Speech</h1>
<h4 align="center"> Yuan Ge, Haishu Zhao, Aokai Hao, Junxiang Zhang, Bei Li, Xiaoqian Liu, Chenglong Wang, Jianjin Wang, Bingsen Zhou, Bingyu Liu, Jingbo Zhu, Zhengtao Yu, Tong Xiao</h4>

<p align="center">
  📄 <a href="https://arxiv.org/pdf/2603.16483">Paper</a> |  
  🤗 <a href="https://huggingface.co/datasets/ak0255/Synthesis_SER">Dataset </a>
</p>

<h3 align="center"> Synthesized SER training and evaluation demo </h3>

## News💡

- [2026.04] Our paper is accepted by ACL 2026 (Main Conference)!🎉🎉🎉 (**_Top 5%_** of accepted papers, seminal paper)
- [2026.03] The full dataset have been publicly released.
- [2026.03] Demo code, test dataset, have been publicly released.
- [2026.03] We release our paper. If you have any questions about our project, please send email to geyuanqaq@gmail.com.

# Install requirements
```bash
pip install -r requirements.txt
```

# Train Demo
We provide training demo for TESS dataset in the `train_downstream_demo` folder. You can modify the scripts to train your downstream model on other datasets.

## Training Data Preprocessing
1. Add your training dataset

Convert the training `.wav` or `.mp4` files into `.npy` format, run:
```python
python /your/path/to/scripts/extract_features_batch.sh
```
2. Obtain the `.emo` file

Convert the `json` file of training data into `.emo` format, run:
```python
python /your/path/to/scripts/json_to_emo.sh
```
3. Obtain the `.dann` and `.length`

After obtaining the `.npy` features and the corresponding `.emo` files, run:
```python
python /your/path/to/train_downstream_demo/pack_tess_fea.py
```

If you want to train a DANN model, additionally run:
```python 
python /your/path/to/train_downstream_demo/add_dann_para.py
```
4. Mix all training datasets:

Mix all datasets for downstream training, run:
```python 
python /your/path/to/downstream/final_mix_all.py
```
You will obtain the data format like `/downstream/Final_mix_demo` for the subsequent training.

## Training your downstream model
We provide a training data format demo for TESS dataset in `/downstream/Final_mix_demo` folder. 
Based on this format, you can reproduce the experimental results with the same datasets presented in the paper.

1. Train a SER model, run:
```python
python /your/path/to/downstream/final_train.sh
```
2. Train a DANN model, run:
```python
python /your/path/to/downstream/train_DANN_balenced.sh
```
3. Probe experiments, run:
```python
python /your/path/to/downstream/probe_train.sh
```
The training results will be saved in `/downstream/outputs` folder

# Evaluation Demo
We provide a evaluatino demo in `/downstream/batch_test` folder.

Testing a SER model, run:
```python
python /your/path/to/downstream/batch_test/EVAL.py
```

DANN model evaluation, run:
```python
python /your/path/to/downstream/batch_test/EVAL_DANN.py
```

Probe experiments evaluation, run:
```python
python /your/path/to/downstream/batch_test/EVAL_probe_emotion.py
```

# Visualization
If you want to observe the distribution of the features after T-SNE dimensionality reduction, run:
```python
python /your/path/to/downstream/visualize/visualize.py
```


## Citation 

If you find our paper useful, please consider citing:
```bibtex
@article{ge2026emotion,
  title={On the Emotion Understanding of Synthesized Speech},
  author={Ge, Yuan and Zhao, Haishu and Hao, Aokai and Zhang, Junxiang and Li, Bei and Liu, Xiaoqian and Wang, Chenglong and Wang, Jianjin and Zhou, Bingsen and Liu, Bingyu and others},
  journal={arXiv preprint arXiv:2603.16483},
  year={2026}
}
```



