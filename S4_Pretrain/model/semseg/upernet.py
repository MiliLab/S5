import torch
import torch.nn as nn
from model.backbone.vit import ViT_B, ViT_L, ViT_H
from model.semseg.encoder_decoder import MTP_SS_UperNet
from model.backbone.swin import swin
import torch.nn.functional as F

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def get_backbone(args):
    
    if args.backbone == 'swin_t':
        encoder = swin(embed_dim=96, 
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=7,
                    ape=False,
                    drop_path_rate=0.3,
                    patch_norm=True
                    )
        print('################# Using Swin-T as backbone! ###################')
        if args.init_backbone == 'rsp':
            encoder.init_weights('./pretrained/rsp-swin-t-ckpt.pth')
        if args.init_backbone == 'imp':
            encoder.init_weights('./pretrained/swin_tiny_patch4_window7_224.pth')
            print('################# Initing Swin-T pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure Swin-T Pretraining! ###################')
        else:
            raise NotImplementedError
    
    elif args.backbone == 'swin_b':
        encoder = swin_b()
        print('################# Using Swin-T as backbone! ###################')
        if args.init_backbone == 'imp':
            encoder.init_weights('./pretrained/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth')
            print('################# Initing Swin-T pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure Swin-T Pretraining! ###################')
        else:
            raise NotImplementedError
        
    elif args.backbone == 'swin_l':
        encoder = swin(embed_dim=192, 
                        depths=[2, 2, 18, 2],
                        num_heads=[6, 12, 24, 48],
                        window_size=7,
                        ape=False,
                        drop_path_rate=0.3,
                        patch_norm=True
                        )
        print('################# Using Swin-L as backbone! ###################')
        if args.init_backbone == 'imp':
            encoder.init_weights('./pretrained/swin_large_patch4_window7_224_22k.pth')
            print('################# Initing Swin-T pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure Swin-T Pretraining! ###################')
        else:
            raise NotImplementedError

    elif args.backbone == 'vit_b_rvsa':
        encoder = vit_b_rvsa(args)
        print('################# Using ViT-B + RVSA as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('./pretrained/vit-b-checkpoint-1599.pth')
            print('################# Initing ViT-B + RVSA pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-B + RVSA SEP Pretraining! ###################')
        else:
            raise NotImplementedError

    elif args.backbone == 'vit_l_rvsa':
        encoder = vit_l_rvsa(args)
        print('################# Using ViT-L + RVSA as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('./pretrained/vit-l-mae-checkpoint-1599.pth')
            print('################# Initing ViT-L + RVSA pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-L + RVSA SEP Pretraining! ###################')
        else:
            raise NotImplementedError

    elif args.backbone == 'vit_h':
        encoder = ViT_H(args)
        print('################# Using ViT-H as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('./pretrained/vit-h-mae-checkpoint-1599.pth')
            print('################# Initing ViT-H pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-H SEP Pretraining! ###################')
        else:
            raise NotImplementedError

    elif args.backbone == 'vit_l':
        encoder = ViT_L(args)
        print('################# Using ViT-L as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('./pretrained/vit-l-mae-checkpoint-1599.pth')
            print('################# Initing ViT-L pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-L SEP Pretraining! ###################')
        else:
            raise NotImplementedError

    elif args.backbone == 'vit_b':
        encoder = ViT_B(args)
        print('################# Using ViT-B as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('./pretrained/vit-b-checkpoint-1599.pth')
            print('################# Initing ViT-B pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-B SEP Pretraining! ###################')
        else:
            raise NotImplementedError

    return encoder

def get_semsegdecoder(in_channels):
    semsegdecoder = MTP_SS_UperNet(
    decode_head = dict(
                type='UPerHead',
                num_classes = 1,
                in_channels=in_channels,
                ignore_index=255,
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=256,
                dropout_ratio=0.1,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                ))
    return semsegdecoder


class UperNet(torch.nn.Module):
    def __init__(self, args, cfg):
        super(UperNet, self).__init__()

        self.args = args
        self.encoder = get_backbone(args)
        print('################# Using UperNet for semseg! ######################')
        self.semsegdecoder = get_semsegdecoder(in_channels=getattr(self.encoder, 'out_channels', None))
        self.semseghead = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(256, cfg['nclass'], kernel_size=1))

    def forward(self, x):
        h, w = x.shape[-2:]
        e = self.encoder(x)
        ss = self.semsegdecoder.decode_head._forward_feature(e)
        out = self.semseghead(ss)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out












