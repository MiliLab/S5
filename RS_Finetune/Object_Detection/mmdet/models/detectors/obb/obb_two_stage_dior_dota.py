import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from .obb_base import OBBBaseDetector
from .obb_test_mixins import RotateAugRPNTestMixin
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)



@DETECTORS.register_module()
class OBBTwoStageDetectorMTD(OBBBaseDetector, RotateAugRPNTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck_dior=None,
                 neck_dota=None,
                 rpn_head_dior=None,
                 rpn_head_dota=None,
                 roi_head_dior=None,
                 roi_head_dota=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OBBTwoStageDetectorMTD, self).__init__()
        self.backbone = build_backbone(backbone)

        self.neck_dior = build_neck(neck_dior)
        self.neck_dota = build_neck(neck_dota)

        
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head_dior_ = rpn_head_dior.copy()
        rpn_head_dior_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        self.rpn_head_dior = build_head(rpn_head_dior_)

        rpn_head_dota_ = rpn_head_dota.copy()
        rpn_head_dota_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        self.rpn_head_dota = build_head(rpn_head_dota_)

        rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        roi_head_dior.update(train_cfg=rcnn_train_cfg)
        roi_head_dior.update(test_cfg=test_cfg.rcnn)
        self.roi_head_dior = build_head(roi_head_dior)

        roi_head_dota.update(train_cfg=rcnn_train_cfg)
        roi_head_dota.update(test_cfg=test_cfg.rcnn)
        self.roi_head_dota = build_head(roi_head_dota)
        self.iter_counter = 0


        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None
    

    @property
    def with_seg_head(self):
        """bool: whether the detector has seg_head"""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    @property
    def with_aux_seg_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'auxiliary_seg_head') and self.auxiliary_seg_head is not None
    
    @property
    def with_cls_head(self):
        """bool: whether the detector has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None
    
    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(OBBTwoStageDetectorMTD, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network FLOPs (only for DIOR branch)."""
        outs = ()

        # Backbone
        if 'MOE' in self.backbone.__class__.__name__:
            x = self.backbone(img, 'dior')
        else:
            x = self.backbone(img)

        # Neck
        if self.neck_dior is not None:
            x = self.neck_dior(x)

        # RPN
        if self.rpn_head_dior is not None:
            proposal_type = getattr(self.rpn_head_dior, 'bbox_type', 'hbb')
            rpn_outs = self.rpn_head_dior(x)
            outs = outs + (rpn_outs,)
        else:
            proposal_type = 'hbb'

        # Dummy proposals
        if proposal_type == 'hbb':
            proposals = torch.randn(1000, 4).to(img.device)
        elif proposal_type == 'obb':
            proposals = torch.randn(1000, 5).to(img.device)
        else:
            proposals = None

        # ROI Head
        roi_outs = self.roi_head_dior.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)

        return outs


    def train_step(self, data, optimizer):
        if isinstance(data, (tuple, list)) and len(data) == 2:
            data_dior, data_dota = data
        else:
            raise ValueError("Expected data to be a tuple/list of two dicts for DIOR and DOTA")


        with autocast(enabled=True):  # 关键
            losses_dior = self.forward_train(**data_dior)
            losses_dota = self.forward_train(**data_dota)

            loss_dior, log_vars_dior = self._parse_losses(losses_dior)
            loss_dota, log_vars_dota = self._parse_losses(losses_dota)

            total_loss = loss_dior + loss_dota
        total_log_vars = {}
        total_log_vars.update({f"dior_{k}": v for k, v in log_vars_dior.items()})
        total_log_vars.update({f"dota_{k}": v for k, v in log_vars_dota.items()})
        num_samples = len(data_dior['img_metas']) + len(data_dota['img_metas'])

        outputs = dict(loss=total_loss, log_vars=total_log_vars, num_samples=num_samples)
        return outputs
    


    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_obboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        新增参数 dataset_name 用来区分处理 dior 或 dota 数据
        """
        dataset_name = None
        dataset_name = img_metas[0]['dataset_name']

        if dataset_name == 'dior':
            if 'MOE' in self.backbone.__class__.__name__:
                x = self.backbone(img, dataset_name)
            else:
                x = self.backbone(img)    
            if self.neck_dior is not None:
                x = self.neck_dior(x)
            # RPN
            if self.rpn_head_dior is not None:
                proposal_type = getattr(self.rpn_head_dior, 'bbox_type', 'hbb')
                target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
                target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' else gt_obboxes_ignore
                proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head_dior.forward_train(
                    x, img_metas, target_bboxes, gt_labels=None,
                    gt_bboxes_ignore=target_bboxes_ignore, proposal_cfg=proposal_cfg)
            else:
                proposal_list = proposals

            # ROI Head
            roi_losses = self.roi_head_dior.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_obboxes, gt_labels,
                gt_bboxes_ignore, gt_obboxes_ignore, **kwargs)

            losses = {}
            if self.rpn_head_dior is not None:
                losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        elif dataset_name == 'dota':
            if 'MOE' in self.backbone.__class__.__name__:
                x = self.backbone(img, dataset_name)
            else:
                x = self.backbone(img)  
            if self.neck_dota is not None:
                x = self.neck_dota(x)
            # RPN
            if self.rpn_head_dota is not None:
                proposal_type = getattr(self.rpn_head_dota, 'bbox_type', 'hbb')
                target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
                target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' else gt_obboxes_ignore
                proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head_dota.forward_train(
                    x, img_metas, target_bboxes, gt_labels=None,
                    gt_bboxes_ignore=target_bboxes_ignore, proposal_cfg=proposal_cfg)
            else:
                proposal_list = proposals

            # ROI Head
            roi_losses = self.roi_head_dota.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_obboxes, gt_labels,
                gt_bboxes_ignore, gt_obboxes_ignore, **kwargs)

            losses = {}
            if self.rpn_head_dota is not None:
                losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        else:
            raise ValueError(f"Unknown dataset_name {dataset_name}, must be 'dior' or 'dota'")
    

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        dataset_name = img_metas[0]['dataset_name'] 
        
        if dataset_name not in ['dior', 'dota']:
            raise ValueError(f"Unknown dataset_name {dataset_name}, must be 'dior' or 'dota'")

        outs = {}

        if 'MOE' in self.backbone.__class__.__name__:
            x = self.backbone(img, dataset_name)
        else:
            x = self.backbone(img)

        # 使用不同的 neck 和分支
        if dataset_name == 'dior':
            if self.neck_dior is not None:
                x = self.neck_dior(x)
            rpn_head = self.rpn_head_dior
            roi_head = self.roi_head_dior
        elif dataset_name == 'dota':
            if self.neck_dota is not None:
                x = self.neck_dota(x)
            rpn_head = self.rpn_head_dota
            roi_head = self.roi_head_dota

        # RPN
        if proposals is None and rpn_head is not None:
            proposal_list = rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        # ROI HEAD
        if roi_head is not None:
            # outs['det_out'] = roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
            return roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rotate_aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)



    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg['seg'].stride
        h_crop, w_crop = self.test_cfg['seg'].crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.seg_head.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # crop_seg_logit = self.encode_decode(crop_img, img_meta)
                x = self.backbone(crop_img)
                if self.with_neck:
                    x = self.neck(x)
                crop_seg_logit = self.seg_head.forward_test(x, img_meta, test_cfg=None)
                crop_seg_logit = resize(
                    input=crop_seg_logit,
                    size=crop_img.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=False,
                warning=False)
        return preds

    def whole_inference(self, x, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.seg_head.forward_test(x, img_meta, test_cfg=None)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=False,
                warning=False)

        return seg_logit

