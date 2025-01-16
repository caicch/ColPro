# Empowering Large Language Model for Continual Video Question Answering with Collaborative Prompting

This is the implementation of ColPro (EMNLP 2024 [here](https://aclanthology.org/2024.emnlp-main.227/))

## Setup
To install requirements, run:
```
git clone
cd ColPro
mkdir pretrained
mkdir data
conda create -n colpro python=3.8
conda activate colpro
sh setup.sh
```

## Dataset & LLaMA Preparation

The preprocessed datasets (split NExT-QA, DramaQA) are stored in datasets folder. You can vit pre-training weight (clipvitl14.pth for NExT-QA, DramaQA) form [here](https://drive.google.com/drive/folders/1XtMZMNW3CRmzvpEVYj29iaUgDFcPUroe?usp=drive_link). Put them in ```./data```. Also, you can download original LLaMA at [here](https://github.com/facebookresearch/llama/tree/llama_v1), and put the checkpoint in ```./pretrained```. 

```
./pretrained
   └─ llama
       |─ 7B
       |   |─ consolidated.00.pth
       |   └─ params.json
       └─ tokenizer.model
./data
   |─ nextqa
   |   |─ split_data
   |   |  |─ train_CH.csv
   |   |  |─ train_CW.csv
   |   |   ..
   |   └─ clipvitl14.pth
   └─ dramaqa
       :
```

## Training LLaMA-VQA (LLaMA + ColPro)

We upload the baseline code first, clean the code of the paper

### NExT-QA on a single 24G GPU

```
python train_nextQA.py --model 7B \
--max_seq_len 128 --batch_size 2 --epochs 5 --warmup_epochs 2 --bias 3.5 --tau 100. --max_feats 10 --dataset nextqa \
--blr 9e-2 --weight_decay 0.14 --output_dir ./checkpoint/nextqa --accum_iter 2
```

### DramaQA on a single 24G GPU

```
python train.py --model 7B \
--max_seq_len 384 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset dramaqa \
--blr 9e-2 --weight_decay 0.10 --output_dir ./checkpoint/dramaqa --accum_iter 8
```

## Acknowledgements

This repo is built upon [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) and [Flipped-VQA](https://github.com/mlvlab/Flipped-VQA)

## Citations

```
@inproceedings{cai-etal-2024-empowering,
    title = "Empowering Large Language Model for Continual Video Question Answering with Collaborative Prompting",
    author = "Cai, Chen and Wang, Zheng  and Gao, Jianjun and Liu, Wenyang  and Lu, Ye  and Zhang, Runzhong  and Yap, Kim-Hui",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    doi = "10.18653/v1/2024.emnlp-main.227",
    pages = "3921--3932",
}
```

