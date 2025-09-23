import argparse
import logging
import os
import pprint
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.distributed as dist
import numpy as np
import random
from dataset.finetune import SemiDataset, ValDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, intersectionAndUnion, intersectionAndUnionGPU
from util.dist_helper import setup_distributed
from model.semseg.upernet_mdf import UperNet
import torch.nn.functional as F
from util.train_utils import (DictAverageMeter, confidence_weighted_loss)
from model.util.utils import *

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--backbone', type=str, default='vit_b', required=True)
parser.add_argument('--init_backbone', type=str, default='none', required=True)
parser.add_argument('--vaihingen-id-path', type=str, required=True)
parser.add_argument('--potsdam-id-path', type=str, default=None)
parser.add_argument('--openearthmap-id-path', type=str, default=None)
parser.add_argument('--loveda-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--experts', default=4, type=int)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--part', type=int, default=256)
parser.add_argument('--tasks', nargs='+', default=['vaihingen', 'potsdam', 'openearthmap', 'loveda'], type=str, help='List of dataset names')
parser.add_argument('--interval', default=1, type=int, help='valid interval')
parser.add_argument('--load_network', default=True)
parser.add_argument('--resume', type=str, help='resume name')


@torch.no_grad()
def validation(cfg, model, valid_loader, dataset=None):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()
    model.eval()
    for (x, y) in valid_loader:
        x = x.cuda()
        if cfg[dataset]['eval_mode'] == 'slide_window':
            b, _, h, w = x.shape    
            final = torch.zeros(b, cfg['nclass'], h, w).cuda() 
            size = cfg['crop_size']
            step = int(cfg['crop_size'] * 2 / 3) 
            b = 0
            a = 0
            while (a <= int(h / step)):
                while (b <= int(w / step)):
                    sub_input = x[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)]
                    mask = model(sub_input, dataset) 
                    final[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)] += mask
                    b += 1
                b = 0
                a += 1
            o = final.argmax(dim=1)
        elif cfg[dataset]['eval_mode'] == 'resize':
            original_shape = x.shape[-2:] 
            resized_x = F.interpolate(x, size=cfg['crop_size'], mode='bilinear', align_corners=True)
            resized_o = model(resized_x, dataset)   
            o = F.interpolate(resized_o, size=original_shape, mode='bilinear', align_corners=True)
            o = o.argmax(dim=1)
        else:
            o = model(x, dataset)
            o = o.max(1)[1]
        intersection, union, target, predict = \
            intersectionAndUnion(o.cpu().numpy(), y.numpy(), cfg[dataset]['nclass'], cfg[dataset]['ignore_index'])
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()
        reduced_predict = torch.from_numpy(predict).cuda()
        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)
        dist.all_reduce(reduced_predict)
        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())
        predict_meter.update(reduced_predict.cpu().numpy())
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10)
    F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)
    
    mIoU = np.nanmean(iou_class) * 100.0
    mAcc = np.nanmean(accuracy_class) * 100.0
    mF1 = np.nanmean(F1_class) * 100.0
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


def main():
    
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True
    model = UperNet(args, cfg)

    if args.load_network:
        model, ckpt_dict = load_network(model, args.resume, rank=rank, logger=logger)
    
    if "moe" in args.backbone:
        logger.info("=> copying FFN weights to FFNMOE weights")
        load_moe_ffn_weights(model, ckpt_dict, part_features=args.part)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    from mmengine.optim import build_optim_wrapper
    optim_wrapper = dict(
        optimizer=dict(
        type='AdamW', lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.05),
        paramwise_cfg=dict(
            num_layers=12, 
            layer_decay_rate=0.9,
            )
            )
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, cfg['epochs'], eta_min=0, last_epoch=-1)
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    if "vit" in args.backbone:
        model._set_static_graph()

    criterions = {name: nn.CrossEntropyLoss(**cfg[name]['criterion']['kwargs']).cuda(local_rank)
                  for name in args.tasks}

    train_sets = {'potsdam': SemiDataset('potsdam', cfg['data_root'], 'train_l',
                            size=cfg['crop_size'], ignore_value=cfg['potsdam']['ignore_index'],
                            id_path=args.potsdam_id_path),
                  'vaihingen': SemiDataset('vaihingen', cfg['data_root'], 'train_l',
                                        size=cfg['crop_size'], ignore_value=cfg['vaihingen']['ignore_index'],
                                        id_path=args.vaihingen_id_path),
                  'openearthmap': SemiDataset('openearthmap', cfg['data_root'], 'train_l',
                                            size=cfg['crop_size'], ignore_value=cfg['openearthmap']['ignore_index'],
                                            id_path=args.openearthmap_id_path),
                  'loveda': SemiDataset('loveda', cfg['data_root'], 'train_l',
                                    size=cfg['crop_size'], ignore_value=cfg['loveda']['ignore_index'],
                                    id_path=args.loveda_id_path)}

    max_len = max(len(ds.ids) for ds in train_sets.values())

    for name in list(train_sets.keys()):
        train_sets[name] = SemiDataset(
            name, cfg['data_root'], 'train_l',
            size=cfg['crop_size'],
            ignore_value=cfg[name]['ignore_index'],
            id_path=getattr(args, f"{name.lower()}_id_path"),
            nsample=max_len)
                  
    train_loaders = {}
   
    for name, dataset in train_sets.items():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(
            dataset,
            batch_size=cfg['batch_size'],
            shuffle=(sampler is None),
            pin_memory=True,
            num_workers=8,
            drop_last=True,
            sampler=sampler
        )
        train_loaders[name] = loader

    val_loaders = {}
   
    for name in args.tasks:
        dataset = ValDataset(name, cfg['data_root'], 'val')
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        batch_size = 1 if name == 'openearthmap' else 8
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=8,
            drop_last=False,
            sampler=sampler
        )
        val_loaders[name] = loader

    previous_best_avg_mIoU = 0.0
    epoch = -1
    scaler = torch.cuda.amp.GradScaler()
    amp = cfg['amp']

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_avg_mIoU = checkpoint['previous_best_avg_mIoU']
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        log_avg = DictAverageMeter()
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_avg_mIoU))

        for loader in train_loaders.values():
            loader.sampler.set_epoch(epoch)

        loaders = [train_loaders[name] for name in args.tasks]
        loader = zip(*loaders)

        for i, batches in enumerate(loader):
            imgs, masks = {}, {}
            for (name, (img, mask)) in zip(args.tasks, batches):
                imgs[name], masks[name] = img.cuda(), mask.cuda()

            with torch.cuda.amp.autocast(enabled=amp):
                model.train()
                losses = {}
                for name in args.tasks:
                    pred = model(imgs[name], name)
                    losses[name] = criterions[name](pred, masks[name])
                total_loss = sum(losses.values())

            torch.distributed.barrier()
            optimizer.zero_grad()
            loss = scaler.scale(total_loss)
            loss.backward()
            scaler.step(optimizer)
            scaler.update()

            log_dict = {'Total loss': total_loss}
            for k, v in losses.items():
                log_dict[f'Loss {k}'] = v
            log_avg.update(log_dict)

            if (i % (max(2, len(train_loaders['potsdam']) // 8)) == 0) and (rank == 0):
                logger.info('===========> Iteration: {:}/{:}, Epoch: {:}/{:}, log_avg: {}'
                            .format(i, len(train_loaders['potsdam']), epoch, cfg['epochs'], str(log_avg)))
        scheduler.step()
        if (epoch + 1) % args.interval == 0:
            miou_dict = {}
            for name in ['vaihingen', 'potsdam', 'openearthmap']:
                valloader = val_loaders[name]
                start_time = time.time()
                mIoU, mAcc, mF1, allAcc, iou_class, F1_class = validation(cfg, model, valloader, dataset=name)
                end_time = time.time()

                if rank == 0:
                    logger.info(f'\n========== Evaluation on {name.upper()} ==========')
                    logger.info('Validation Results - Epoch [{}/{}] on {}:'.format(epoch + 1, cfg['epochs'], name))
                    logger.info('mIoU: {:.4f}, mAcc: {:.4f}, mF1: {:.4f}, allAcc: {:.4f}, Time: {:.2f}s'.format(
                        mIoU, mAcc, mF1, allAcc, end_time - start_time))
                miou_dict[name] = mIoU

            avg_mIoU = sum(miou_dict.values()) / len(miou_dict)
            is_best = avg_mIoU > previous_best_avg_mIoU

            if rank == 0:
                logger.info('\n========== Summary for Epoch [{}/{}] =========='.format(epoch + 1, cfg['epochs']))
                for name in miou_dict:
                    logger.info('Dataset: {:<15} mIoU: {:.4f}'.format(name, miou_dict[name]))
                logger.info('Average mIoU across datasets: {:.4f} (Best so far: {:.4f})\n'.format(
                    avg_mIoU, previous_best_avg_mIoU))

                checkpoint = {'model': model.state_dict(),
                              'previous_best_avg_mIoU': previous_best_avg_mIoU}

                latest_path = os.path.join(args.save_path, 'latest.pth')
                torch.save(checkpoint, latest_path)
                logger.info(f'Latest checkpoint saved to {latest_path}')

                if is_best:
                    previous_best_avg_mIoU = avg_mIoU
                    best_path = os.path.join(args.save_path, 'best.pth')
                    torch.save(checkpoint, best_path)
                    logger.info(f'>>> New Best Model! Saved to {best_path}')


if __name__ == '__main__':
    main()
