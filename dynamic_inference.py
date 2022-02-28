import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.multiprocessing
import torch.nn.functional as F
from timm.data.auto_augment import invert
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import *

import math
import argparse
import deit.models_deit 
import lvvit.models
from timm.models import create_model

import pdb



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Inference code for CFViT')

parser.add_argument('--data_url', default='./data', type=str,
                    help='path to the dataset (ImageNet)')

parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')

parser.add_argument('--threshold', default=0.5, type=float,
                    help='the threshhold of coarse stage')

parser.add_argument('--model', default='DVT_T2t_vit_12', type=str,
                    help='model name')


parser.add_argument('--coarse-stage-size', default=7, type=int, help='the length of coarse splitting')



parser.add_argument('--checkpoint_path', default='', type=str,
                    help='path to the pre-train model (default: none)')

parser.add_argument('--eval-mode', default=1, type=int,
                    help='mode 0 : inference without early exit\
                          mode 1 : infer the model on the validation set with various threshold\
                          mode 2 : print the dynamic inference results')

args = parser.parse_args()


def main():
    args.input_size_list = [16*args.coarse_stage_size, 16*2*args.coarse_stage_size]
    args.input_size = max(args.input_size_list)
    # load pretrained model
    checkpoint = torch.load(args.checkpoint_path)
    flops = checkpoint['flop']
    if args.eval_mode == 2:
        anytime_classification = checkpoint['anytime_classification']
        budgeted_batch_classification = checkpoint['budgeted_batch_classification']
        print('flops :', flops)
        print('anytime_classification :', anytime_classification)
        print('budgeted_batch_classification :', budgeted_batch_classification)
        pdb.set_trace()
        return

    model = create_model(
    args.model,
    pretrained=False,
    img_size_list = args.input_size_list,
    num_classes=1000,
    drop_rate=0.0,
    drop_connect_rate=None,  
    drop_path_rate=0.1,
    drop_block_rate=None,
    global_pool=None,
    bn_tf=False,
    bn_momentum=None,
    bn_eps=None,
    checkpoint_path='')

    valdir = args.data_url + 'val/'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    crop_pac = 0.875 
    if "lvvit" in args.model:
        crop_pac = 0.9

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(args.input_size/crop_pac),interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            normalize])),
        batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=False)
    
    model = model.cuda()

    model.load_state_dict(checkpoint['model'])

    # if "deit" in args.model:
    #     model.load_state_dict(checkpoint['model'])
    # else:
    #     model.load_state_dict(checkpoint['state_dict'])

    model.apply(lambda m: setattr(m,'informative_selection', True))
    

    budgeted_batch_flops_list = []
    budgeted_batch_acc_list = []

    print('generate logits on test samples...')
    test_logits, test_targets, anytime_classification = generate_logits(model, val_loader, args.input_size_list)
    print('flops :', flops)
    print('anytime_classification :', anytime_classification)
    # pdb.set_trace()

    # pdb.set_trace()
    if args.eval_mode == 0:
        return
    for p in range(0, 100):

        print('inference: {}/100'.format(p))
        threshold = [0.01 * p,-1]
        
        acc_step, flops_step = dynamic_evaluate(test_logits, test_targets, flops, threshold)
        
        budgeted_batch_acc_list.append(acc_step)
        budgeted_batch_flops_list.append(flops_step)
    
    budgeted_batch_classification = [budgeted_batch_flops_list, budgeted_batch_acc_list]

    print('budgeted_batch_classification :', budgeted_batch_classification)
    checkpoint['anytime_classification'] = anytime_classification
    checkpoint['budgeted_batch_classification'] = budgeted_batch_classification
    pdb.set_trace()
    torch.save(checkpoint, args.checkpoint_path)



def generate_logits(model, dataloader, input_size_list):

    logits_list = []
    targets_list = []

    top1 = [AverageMeter() for _ in range(2)]
    model.eval()
    input_size_list = sorted(input_size_list)
    for i, (x, target) in enumerate(dataloader):

        logits_temp = torch.zeros(2, x.size(0), 1000).cuda()

        target_var = target.cuda()
        input_var = x.cuda()
        images_list = []
        for i in range(0,len(input_size_list)-1):
            resized_img = F.interpolate(input_var, (input_size_list[i], input_size_list[i]), mode='bilinear', align_corners=True)
            resized_img = torch.squeeze(resized_img)
            images_list.append(resized_img)  
        images_list.append(input_var)  
        with torch.no_grad():
            results = model(images_list)
            coarse_output, fine_output = results[0], results[1]
                
            logits_temp[0] = F.softmax(coarse_output, 1)
            logits_temp[1] = F.softmax(fine_output, 1)

            acc = accuracy(coarse_output, target_var, topk=(1,))
            top1[0].update(acc.sum(0).mul_(100.0 / x.size(0)).data.item(), x.size(0))
            acc = accuracy(fine_output, target_var, topk=(1,))
            top1[1].update(acc.sum(0).mul_(100.0 / x.size(0)).data.item(), x.size(0))
            
        logits_list.append(logits_temp)
        targets_list.append(target_var)

        anytime_classification = []

        for index in range(2):
            anytime_classification.append(top1[index].ave)

    return torch.cat(logits_list, 1), torch.cat(targets_list, 0), anytime_classification



   

def dynamic_evaluate(logits, targets, flops, T):

    n_stage, n_sample, c = logits.size()
    max_preds, argmax_preds = logits.max(dim=2, keepdim=False)
    _, sorted_idx = max_preds.sort(dim=1, descending=True)
    acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
    acc, expected_flops = 0, 0
    for i in range(n_sample):
        gold_label = targets[i]
        for k in range(n_stage):
            if max_preds[k][i].item() >= T[k]:  # force the sample to exit at k
                if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                    acc += 1
                    acc_rec[k] += 1
                exp[k] += 1
                break
    acc_all = 0
    for k in range(n_stage):
        _t = 1.0 * exp[k] / n_sample
        expected_flops += _t * flops[k]
        acc_all += acc_rec[k]

    return acc * 100.0 / n_sample, expected_flops.item()


if __name__ == '__main__':
    main()