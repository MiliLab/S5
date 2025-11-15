import os
import time
import yaml
import pprint
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import random
import numpy as np
import cv2
import PIL.Image as Image

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from dataset.semi_sup import SemiDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, intersectionAndUnionGPU
from util.dist_helper import setup_distributed
from util.train_utils import DictAverageMeter, confidence_weighted_loss, cutmix_img_, cutmix_mask
from model.semseg.upernet import UperNet

from mmengine.optim import build_optim_wrapper


# -----------------------
# Utils
# -----------------------
def set_seeds(seed=2025):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def save_checkpoint(state, backbone, save_dir: Path, backbone_name: str, suffix: str):
    """统一的 checkpoint 保存函数."""
    torch.save(state, save_dir / f'{backbone_name}_s4p_upernet.pth')
    torch.save(backbone, save_dir / f'{backbone_name}_s4p.pth')


@torch.no_grad()
def validation(model, dataloader, cfg, eval_mode="resize"):
    """Validation with resize or sliding-window inference."""
    meters = {k: AverageMeter() for k in ["intersection", "union", "target", "predict"]}
    model.eval()

    for x, y in dataloader:
        x, y = x.cuda(), y.long().cuda()
        if eval_mode == 'resize':
            ori_shape = x.shape[-2:]
            out = model(F.interpolate(x, size=cfg['crop_size'], mode='bilinear', align_corners=True))
            pred = F.interpolate(out, size=ori_shape, mode='bilinear', align_corners=True).argmax(dim=1)
        else:
            pred = model(x).argmax(dim=1)

        inter, union, tgt, prd = intersectionAndUnionGPU(pred, y, cfg['nclass'], cfg['ignore_index'])
        for k, v in zip(meters.keys(), [inter, union, tgt, prd]):
            dist.all_reduce(v)
            meters[k].update(v)

    # Metrics
    iou = meters["intersection"].sum.cpu().numpy()  / (meters["union"].sum.cpu().numpy()  + 1e-10) * 100
    acc = meters["intersection"].sum.cpu().numpy()  / (meters["target"].sum.cpu().numpy()  + 1e-10) * 100
    prec = meters["intersection"].sum.cpu().numpy()  / (meters["predict"].sum.cpu().numpy()  + 1e-10) * 100
    f1 = 2 * (prec * acc) / (prec + acc + 1e-10)

    if cfg['dataset'] == 'pretrain':  # skip background
        valid_idx = slice(1, None)
    else:
        valid_idx = slice(None)

    return {
        "mIoU": np.mean(iou[valid_idx]),
        "mAcc": np.mean(acc[valid_idx]),
        "mF1": np.mean(f1[valid_idx]),
        "allAcc": meters["intersection"].sum[valid_idx].sum() /
                  (meters["target"].sum[valid_idx].sum() + 1e-10),
        "iou_class": iou,
    }


# -----------------------
# Main Training
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='swint')
    parser.add_argument('--init_backbone', type=str, default='imp')
    parser.add_argument('--decoder', type=str, default='upernet')
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    parser.add_argument('--load', type=str, default='backbone')
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = init_log('global', logging.INFO)
    logger.propagate = False

    # DDP
    rank, world_size = setup_distributed(port=args.port)
    ddp = world_size > 1
    amp = cfg.get("amp", False)

    if rank == 0:
        logger.info("Args + Cfg:\n{}".format(pprint.pformat({**cfg, **vars(args)})))
        writer = SummaryWriter(save_dir)

    # Model
    cudnn.enabled, cudnn.benchmark = True, True
    model = UperNet(args, cfg).cuda()
    if ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                                                    broadcast_buffers=False, find_unused_parameters=True)
    if args.backbone in {'vit_l', 'vit_b', 'vit_h', 'vit_l_rvsa'}:
        model._set_static_graph()

    if rank == 0:
        logger.info(f"Total params: {count_params(model):.1f}M")

    # Optimizer
    optim_cfg = dict(
        optimizer=dict(type='AdamW', lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01),
        paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                        'relative_position_bias_table': dict(decay_mult=0.),
                                        'norm': dict(decay_mult=0.)})
    )
    optimizer = build_optim_wrapper(model, optim_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, cfg['total_iters'], eta_min=0)

    # Loss
    criterion_l = (nn.CrossEntropyLoss(**cfg['criterion']['kwargs'])
                   if cfg['criterion']['name'] == 'CELoss'
                   else ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs'])).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    # Datasets & Loaders
    train_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', size=cfg['crop_size'],
                          ignore_value=cfg['ignore_index'], id_path=args.unlabeled_id_path)
    train_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', size=cfg['crop_size'],
                          ignore_value=cfg['ignore_index'], id_path=args.labeled_id_path, nsample=len(train_u.ids))

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    loader_kwargs = dict(batch_size=cfg['batch_size'], pin_memory=True, drop_last=True, num_workers=cfg['workers'])
    if ddp:
        trainloader_l = DataLoader(train_l, sampler=torch.utils.data.distributed.DistributedSampler(train_l), **loader_kwargs)
        trainloader_u = DataLoader(train_u, sampler=torch.utils.data.distributed.DistributedSampler(train_u), **loader_kwargs)
        valloader = DataLoader(valset, sampler=torch.utils.data.distributed.DistributedSampler(valset),
                               batch_size=4, pin_memory=True, num_workers=cfg['workers'])
    else:
        trainloader_l = DataLoader(train_l, shuffle=True, num_workers=4, **loader_kwargs)
        trainloader_u = DataLoader(train_u, shuffle=True, num_workers=1, **loader_kwargs)
        valloader = DataLoader(valset, batch_size=4, shuffle=False, num_workers=1)

    # Training state
    iters, epoch = -1, -1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    conf_thresh = cfg['conf_thresh']
    total_epochs = int(cfg['total_iters'] / len(trainloader_u)) + 1
    # Main loop
    for epoch in range(epoch + 1, total_epochs):
        if ddp:
            trainloader_l.sampler.set_epoch(epoch)
            trainloader_u.sampler.set_epoch(epoch)

        model.train()
        log_avg = DictAverageMeter()
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s, _, ignore_mask, cutmix_box),
                (img_u_w_mix, img_u_s_mix, _, ignore_mask_mix, _)) in enumerate(zip(trainloader_l, trainloader_u, trainloader_u)):
            
            t0 = time.time()
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s, ignore_mask, cutmix_box = img_u_w.cuda(), img_u_s.cuda(), ignore_mask.cuda(), cutmix_box.cuda()
            img_u_w_mix, img_u_s_mix, ignore_mask_mix = img_u_w_mix.cuda(), img_u_s_mix.cuda(), ignore_mask_mix.cuda() 
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            cutmix_img_(img_u_s, img_u_s_mix, cutmix_box)

            with torch.no_grad():
                model.eval()
                pred_u_w, pred_u_w_mix = model(torch.cat((img_u_w, img_u_w_mix))).split([num_ulb, num_ulb])
                conf_u_w, mask_u_w = pred_u_w.softmax(dim=1).max(dim=1)
                conf_u_w_mix, mask_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)
                mask_u_w_cutmixed = cutmix_mask(mask_u_w, mask_u_w_mix, cutmix_box)
                conf_u_w_cutmixed = cutmix_mask(conf_u_w, conf_u_w_mix, cutmix_box)
                ignore_mask_cutmixed = cutmix_mask(ignore_mask, ignore_mask_mix, cutmix_box)

            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            
            with torch.cuda.amp.autocast(enabled=amp):
                preds = model(torch.cat((img_x, img_u_s)))
                pred, pred_u_s = preds.split([num_lb, num_ulb])       
                loss_x = criterion_l(pred, mask_x)
                loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
                loss_u_s = confidence_weighted_loss(loss_u_s, conf_u_w_cutmixed, ignore_mask_cutmixed, conf_thresh=conf_thresh)
                
                mask_ratio = ((conf_u_w_cutmixed >= conf_thresh) & (ignore_mask_cutmixed != cfg['ignore_index'])).sum().item() / (ignore_mask_cutmixed != cfg['ignore_index']).sum()
                total_loss = (loss_x + loss_u_s) / 2.0

            if ddp:
                torch.distributed.barrier()

            optimizer.zero_grad()
            if amp:
                loss = scaler.scale(total_loss)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            log_avg.update({
                'iter time': time.time() - t0,
                'Total loss': total_loss,   
                'Loss x': loss_x,
                'Loss u_s': loss_u_s,
                'Mask ratio': mask_ratio,
            })

            iters += 1
            scheduler.step()
            
            if rank == 0 and iters % cfg['log_interval'] == 0:
                logger.info('===========> Iteration: {:}/{:}, Epoch: {:}/{:}, log_avg: {}'.format(iters, cfg['total_iters'], epoch, total_epochs, str(log_avg)))
            
            if (iters) % cfg['validate_interval'] == 0:
                if rank == 0:
                    logger.info('>>>>>>>>>>>>>>>> Start Evaluation of Finetune >>>>>>>>>>>>>>>>')

                start_time = time.time()
                results = validation(model, valloader, cfg, cfg['eval_mode'])
                end_time = time.time()

                if rank == 0:
                    for (cls_idx, iou) in enumerate(results['iou_class']):
                        logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                    'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                    logger.info('Last: validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs'
                    .format(epoch, total_epochs, results['mIoU'], results['mAcc'], results['mF1'], results['allAcc'], end_time-start_time))
                    
            if iters % cfg['save_interval'] == 0 and rank == 0:
                state = {"model": model.state_dict()}
                backbone = {"model": model.module.encoder.state_dict()}
                save_checkpoint(state, backbone, save_dir, args.backbone, "iter")

            if iters >= cfg['total_iters'] and rank == 0:
                logger.info(f"Training finished at {iters} iterations (target {cfg['total_iters']}).")
                return


if __name__ == '__main__':
    set_seeds()
    main()
