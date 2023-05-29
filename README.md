
# CF-ViT: A General Coarse-to-Fine Method for Vision Transformer (AAAI 2023 Oral)
This is Pytorch implementation of our paper "[CF-ViT: A General Coarse-to-Fine Method for Vision Transformer](https://arxiv.org/abs/2203.03821)".

## Pre-trained Models

|Backbone|# of Coarse Seage|Accuracy(threshold=1)|Checkpoints Links|Logs Links|
|-----|------|-----|-----|-----|
|DeiT-S| 7x7|80.8|[Google Drive](https://drive.google.com/file/d/1b8qU5lTP62Jr_YtEcj2lsJAWkVFZWDWL/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1VUndAotIgBVrd9MguFwuw3vZh-rRm0Mj/view?usp=sharing)|
|DeiT-S| 9x9|81.9| [Google Drive](https://drive.google.com/file/d/1m082crS9cWQEHOewnVrYTktXEgorLCNM/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1ICZ9M3zgLv6_H3lABB8q_0FvGt3oOjXp/view?usp=sharing)|
|LV-ViT-S| 7x7|83.6| [Google Drive](https://drive.google.com/file/d/1C2pjsLmG7OuiZwR-zy5HHgQIdnZsy8C3/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1F-82NHjxG2OaMtJlPUAnoj2nKpgg7ZiP/view?usp=sharing)|
|LV-ViT-S| 9x9|84.4| [Google Drive](https://drive.google.com/file/d/13BBDfWgJrC1_DU96gOVEXphdr1SIm_tH/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1ccaf0mk_8zUT2WHdxYPiaG7GSZTGyMKE/view?usp=sharing)|

- What are contained in the checkpoints:

```
**.pth
├── model: state dictionaries of the model
├── flop: a list containing the GFLOPs corresponding to exiting at each stage
├── anytime_classification: Top-1 accuracy of each stage
├── budgeted_batch_classification: results of budgeted batch classification (a two-item list, [0] and [1] correspond to the two coordinates of a curve)

```
## Requirements
- python 3.9.7
- pytorch 1.10.1
- torchvision 0.11.2
- apex 


## Data Preparation
- The ImageNet dataset should be prepared as follows:
```
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...

```

## Evaluate Pre-trained Models
- Get accuracy of each stage
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --eval-mode 0 --data_url PATH_TO_IMAGENET  --batch_size 64 --model {cf_deit_small, cf_lvvit_small} --checkpoint_path PATH_TO_CHECKPOINT  --coarse-stage-size {7,9} 

```

- Infer the model on the validation set with various threshold([0.01:1:0.01])
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --eval-mode 1 --data_url PATH_TO_IMAGENET  --batch_size 64 --model {cf_deit_small, cf_lvvit_small} --checkpoint_path PATH_TO_CHECKPOINT  --coarse-stage-size {7,9} 

```

- Infer the model on the validation set with one threshold and meature the throughput

```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --eval-mode 2 --data_url PATH_TO_IMAGENET  --batch_size 1024 --model {cf_deit_small, cf_lvvit_small} --checkpoint_path PATH_TO_CHECKPOINT  --coarse-stage-size {7,9} --threshold THRESHOLD

```

- Read the evaluation results saved in pre-trained models
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --eval-mode 3 --data_url PATH_TO_IMAGENET  --batch_size 64 --model {cf_deit_small, cf_lvvit_small} --checkpoint_path PATH_TO_CHECKPOINT  --coarse-stage-size {7,9} 

```

## Train
- Train CF-ViT(DeiT-S) on ImageNet 
```
python -m torch.distributed.launch --nproc_per_node=4 main_deit.py  --model cf_deit_small --batch-size 256 --data-path PATH_TO_IMAGENET --coarse-stage-size {7,9} --dist-eval --output PATH_TO_LOG

```

- Train CF-ViT(LV-ViT-S) on ImageNet 
```
python -m torch.distributed.launch --nproc_per_node=4 main_lvvit.py PATH_TO_IMAGENET --model cf_lvvit_small -b 256 --apex-amp --drop-path 0.1 --token-label --token-label-data PATH_TO_TOKENLABEL --model-ema --eval-metric top1_f --coarse-stage-size {7,9} --output PATH_TO_LOG

```



## Visualization
- Visualization of images correctly classified at coarse stage and fine stage.
```
python visualize.py --model cf_deit_small --resume  PATH_TO_CHECKPOINT --output_dir PATH_TP_SAVE --data-path PATH_TO_IMAGENET --batch-size 64 

```
- Other drawing code can be found in draw_picture.ipynb



## Acknowledgment
Our code of LV-ViT is from [here](https://github.com/zihangJiang/TokenLabeling). Our code of DeiT is from [here](https://github.com/facebookresearch/deitzhe). The visualization code is modified from [Evo-ViT](https://github.com/YifanXu74/Evo-ViT). The dynamic inference with early-exit code is modified from [DVT](https://github.com/blackfeather-wang/Dynamic-Vision-Transformer). Thanks to these authors. 

## Citation
```
@inproceedings{CFViT,
  title={CF-ViT: A General Coarse-to-Fine Method for Vision Transformer},
  author={Mengzhao Chen and Mingbao Lin and Ke Li and Yunhang Shen and Yongjian Wu and Fei Chao and Rongrong Ji},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37}
}
```
