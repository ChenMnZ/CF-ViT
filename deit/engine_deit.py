# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from deit.losses import DistillationLoss
import utils
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, input_size_list:list=[112,160,224]):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # for each batch, update the mixup para1
        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        samples_list = []
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)  
        for i in range(0,len(input_size_list)-1):
            resized_img = F.interpolate(samples, (input_size_list[i], input_size_list[i]), mode='bilinear', align_corners=True)
            resized_img = torch.squeeze(resized_img)
            samples_list.append(resized_img)  
        samples_list.append(samples)
        with torch.cuda.amp.autocast():
            results = model(samples_list)
            loss_ce = criterion(samples, results[1], targets) 
            max_output_detach_softmax = F.softmax(results[1].detach(), dim=1)
            kl_layer = torch.nn.KLDivLoss(reduction='batchmean')
            log_softmax_result0 = F.log_softmax(results[0], dim=1)
            loss_kl = kl_layer(log_softmax_result0, max_output_detach_softmax)
            loss = loss_ce + loss_kl
        optimizer.zero_grad()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device,input_size_list):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        target = target.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        images_list = []
        for i in range(0,len(input_size_list)-1):
            resized_img = F.interpolate(images, (input_size_list[i], input_size_list[i]), mode='bilinear', align_corners=True)
            resized_img = torch.squeeze(resized_img)
            images_list.append(resized_img)  
        images_list.append(images)  
        with torch.cuda.amp.autocast():
            results = model(images_list)
        batch_size = images.shape[0]

        for index, image_size in enumerate(input_size_list):
            acc1, acc5 = accuracy(results[index], target, topk=(1, 5))
            metric_logger.meters[f'{input_size_list[index]}_acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters[f'{input_size_list[index]}_acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for input_size in input_size_list:
        print('size:{size}: * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} '
            .format(size=input_size, top1=metric_logger.meters[f'{input_size}_acc1'], top5=metric_logger.meters[f'{input_size}_acc5']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



