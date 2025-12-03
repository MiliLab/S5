import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)
from torch.utils.data import DataLoader, Dataset
from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook, RandomFPHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
import math


class RepeatDataset(Dataset):
    def __init__(self, dataset, target_len):
        self.dataset = dataset
        self.target_len = target_len
        self.orig_len = len(dataset)

    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        return self.dataset[idx % self.orig_len]

class ZippedDataLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return zip(*[iter(dl) for dl in self.loaders])

    def __len__(self):
        return min(len(dl) for dl in self.loaders)
    
    @property
    def sampler(self):
        class CombinedSampler:
            def __init__(self, samplers):
                self.samplers = samplers

            def set_epoch(self, epoch):
                for s in self.samplers:
                    if hasattr(s, 'set_epoch'):
                        s.set_epoch(epoch)

        return CombinedSampler([getattr(dl, 'sampler', None) for dl in self.loaders])

def train_detector(model,
                   dior_datasets,
                   dota_datasets,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dior_datasets = dior_datasets if isinstance(dior_datasets, (list, tuple)) else [dior_datasets]
    dota_datasets = dota_datasets if isinstance(dota_datasets, (list, tuple)) else [dota_datasets]


    dior_dataset = dior_datasets[0]
    dota_dataset = dota_datasets[0]
    
    len_dior = len(dior_dataset)
    len_dota = len(dota_dataset)
    max_len = max(len_dior, len_dota)

    if len_dior != max_len:
        dior_dataset.repeat_to_length(max_len)
        dior_dataset._set_group_flag()
    if len_dota != max_len:
        dota_dataset.repeat_to_length(max_len)

    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    dior_loader = build_dataloader(
        dior_dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False,  
        seed=cfg.seed)

    dota_loader = build_dataloader(
        dota_dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        # shuffle=False,
        dist=distributed,
        seed=cfg.seed)
    
    zipped_loader = ZippedDataLoader([dior_loader, dota_loader])

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        if eval_cfg is not None:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # register random fp hook
    if cfg.get('random_fp', False):
        runner.register_hook(RandomFPHook())

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run([zipped_loader], cfg.workflow, cfg.total_epochs)
