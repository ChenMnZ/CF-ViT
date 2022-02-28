
## Pre-trained Models

|Backbone|# of Coarse Seage|Links|
|-----|------|-----|-----|
|DeiT-S| 7x7|[Google Drive]()|
|DeiT-S| 9x9| [Google Drive]()|
|LV-ViT-S| 7x7| [Google Drive]()|
|LV-ViT-S| 9x9| [Google Drive]()|

- What are contained in the checkpoints:

```
**.pth
├── model: state dictionaries of the model
├── flop: a list containing the GFLOPs corresponding to exiting at each exit
├── anytime_classification: Top-1 accuracy of each exit
├── dynamic_threshold: the confidence thresholds used in budgeted batch classification
├── budgeted_batch_classification: results of budgeted batch classification (a two-item list, [0] and [1] correspond to the two coordinates of a curve)

```
## Requirements

## Data Preparation
- The ImageNet dataset should be prepared as follows:
```
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...

```

## Evaluate Pre-trained Models
The dynamic inference with early-exit code is modified from [DVT](https://github.com/blackfeather-wang/Dynamic-Vision-Transformer/blob/main/README.md).
- infer the model on the validation set without early exit
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --mode 0 --data_url PATH_TO_IMAGENET  --batch_size 64 --model {cf_deit_small, cf_lvvit_small} --checkpoint_path PATH_TO_CHECKPOINT  --coarse-stage-size {7,9} 
```
- infer the model on the validation set with early exit
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --mode 1 --data_url PATH_TO_IMAGENET  --batch_size 64 --model {cf_deit_small, cf_lvvit_small} --checkpoint_path PATH_TO_CHECKPOINT  --coarse-stage-size {7,9} 
```
- Read the evaluation results saved in pre-trained models
```
CUDA_VISIBLE_DEVICES=0 python dynamic_inference.py --mode 2 --data_url PATH_TO_IMAGENET  --batch_size 64 --model {cf_deit_small, cf_lvvit_small} --checkpoint_path PATH_TO_CHECKPOINT  --coarse-stage-size {7,9} 
```

## Testing inference throughput
python main_lvvit.py /media/DATASET/ImageNet --model cf_lvvit_s -b 256 --apex-amp --drop-path 0.1 --token-label --token-label-data /media/DATASET/label_top5_train_nfnet --model-ema --eval-metric top1_f --coarse-stage-size 7 --resume /home/cmz/cf-vit/checkpoints/cf-lvvit-s-7x7-83.5.pth --eval-throughput


## Train
- Train CF-ViT(DeiT-S) on ImageNet 
```
python -m torch.distributed.launch --nproc_per_node=4 main_deit.py  --model cf_deit_small --batch-size 256 --data-path PATH_TO_IMAGENET --coarse-stage-size 7 --dist-eval --output PATH_TO_LOG
```
```
python -m torch.distributed.launch --nproc_per_node=4 main_deit.py  --model cf_deit_small --batch-size 256 --data-path PATH_TO_IMAGENET --coarse-stage-size 9 --dist-eval --output PATH_TO_LOG
```
- Train CF-ViT(LV-ViT-S) on ImageNet 
```
python -m torch.distributed.launch --nproc_per_node=4 main_lvvit.py PATH_TO_IMAGENET --model cf_lvvit_small -b 256 --apex-amp --drop-path 0.1 --token-label --token-label-data PATH_TO_TOKENLABEL --model-ema --eval-metric top1_f --coarse-stage-size 7 --output PATH_TO_LOG
```
```
python -m torch.distributed.launch --nproc_per_node=4 main_lvvit.py PATH_TO_IMAGENET --model cf_lvvit_small -b 256 --apex-amp --drop-path 0.1 --token-label --token-label-data PATH_TO_TOKENLABEL --model-ema --eval-metric top1_f --coarse-stage-size 9 --output PATH_TO_LOG
```


## Visualization
The visualization code is modified from [Evo-ViT](https://github.com/YifanXu74/Evo-ViT).
'''
python visualize.py --model cf_deit_small --resume  PATH_TO_CHECKPOINT --output_dir PATH_TP_SAVE --data-path PATH_TO_IMAGENET --batch-size 64 
'''



## Acknowledgment
Our code of LV-ViT is from [here](https://github.com/zihangJiang/TokenLabeling). Our code of DeiT is from [here](https://github.com/facebookresearch/deitzhe). 
