import yaml
import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset.test import SemiDataset
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, color_map
from PIL import Image
import cv2
import random
import logging
import torch.nn.functional as F
from collections import OrderedDict
from model.semseg.upernet_mdf import UperNet
import numpy as np
from util.test_utils import remove_module_prefix, save_prediction, net_process
from tqdm import tqdm



def evaluate(model, loader, mode, cfg, visual=False):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

    with torch.no_grad():
        for img_np, img, mask, id in tqdm(loader, desc=f"Evaluating {cfg['dataset']}", ncols=100):
            x = img.cuda()
            if mode == 'slide_window':
                b, _, h, w = x.shape    
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()
                size = cfg['crop_size']
                step = 256
                b = 0
                a = 0
                while (a <= int(h / step)):
                    while (b <= int(w / step)):
                        sub_input = x[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)]
                        pre = net_process(model, sub_input, cfg, flip=False)
                        final[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)] += pre
                        b += 1
                    b = 0
                    a += 1
                pred = final.argmax(dim=1)

            elif mode == 'resize':
                original_shape = x.shape[-2:]
                resized_x = F.interpolate(x, size=cfg['crop_size'], mode='bilinear', align_corners=True)
                resized_o = net_process(model, resized_x, cfg, flip=False)  
                o = F.interpolate(resized_o, size=original_shape, mode='bilinear', align_corners=True)
                pred = o.argmax(dim=1)
            
            else:
                pred = net_process(model, x, cfg, flip=True).argmax(dim=1)
 
            mask = np.array(mask, dtype=np.int32)

            if visual:
                save_prediction(pred, id, cfg['dataset'])
                
                if cfg['dataset'] == 'loveda':
                    continue
            
            intersection, union, target, predict = intersectionAndUnion(pred.cpu().numpy(), mask, cfg['nclass'], cfg['ignore_index'])
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            predict_meter.update(predict)

        if cfg['dataset'] == 'loveda':
            return None, None, None, None, None, None

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10) * 100.0
        precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10) * 100.0
        F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)

        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        mF1 = np.mean(F1_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default='.configs/rsseg.yaml')
    parser.add_argument('--dataset', type=str, default='potsdam', choices=['vaihingen', 'potsdam', 'openearthmap', 'loveda'])
    parser.add_argument('--ckpt-path', type=str, default='./checkpoint/s5_vit_b_moe_mdf.pth')
    parser.add_argument('--backbone', type=str, default='vit_b_moe', required=False)
    parser.add_argument('--init_backbone', type=str, default='none', required=False)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--experts', type=int, default=4)
    parser.add_argument('--tasks', nargs='+', default=['vaihingen', 'potsdam', 'openearthmap', 'loveda']  , type=str, help='List of dataset names')
    parser.add_argument('--part', type=int, default=256)
    parser.add_argument('--visual', default=False)

    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg_dataset = cfg[args.dataset]
    model = UperNet(args, cfg)
    ckpt = torch.load(args.ckpt_path)['model']
    ckpt = remove_module_prefix(ckpt)
    model.load_state_dict(ckpt)
    model.cuda()
    print('Total params: {:.1f}M\n'.format(count_params(model)))
    valset = SemiDataset(cfg_dataset['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=8, drop_last=False)
    
    if args.dataset == 'openearthmap':
        eval_mode = 'resize'
    elif args.dataset == 'loveda':
        eval_mode = 'slide_window'
    else:
        eval_mode = 'original'

    mIoU, mAcc, mF1, allAcc, iou_class, F1_class = evaluate(model, valloader, eval_mode, cfg_dataset, visual=args.visual)

    if args.dataset == 'loveda':
        print("Loveda dataset: predictions saved, skipping metric calculation.")
    
    else:
        print(f"***** Evaluation {eval_mode} for dataset {cfg_dataset['dataset']} *****")

        for cls_idx, F1 in enumerate(F1_class):
            print(f"Class [{cls_idx} {CLASSES[cfg_dataset['dataset']][cls_idx]}] F1: {F1:.2f}")
        print(f"Mean F1: {mF1:.2f}\n")
        
        for cls_idx, IoU in enumerate(iou_class):
            print(f"Class [{cls_idx} {CLASSES[cfg_dataset['dataset']][cls_idx]}] IoU: {IoU:.2f}")
        print(f"Mean IoU: {mIoU:.2f}\n")
    
    
if __name__ == '__main__':
    main()
