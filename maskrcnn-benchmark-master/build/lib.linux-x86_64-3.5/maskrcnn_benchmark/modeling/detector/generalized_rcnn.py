# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        img_size = cfg.INPUT.MIN_SIZE_TEST
        
        # achieve the grid sizes in preprocess stage
        f0 = int(img_size/4) if (img_size%4) == 0 else (int(img_size/4)+1)
        f1 = int(f0/2) if (f0%2) == 0 else (int(f0/2)+1)
        f2 = int(f1/2) if (f1%2) == 0 else (int(f1/2)+1)
        f3 = int(f2/2) if (f2%2) == 0 else (int(f2/2)+1)
        f4 = int(f3/2) if (f3%2) == 0 else (int(f3/2)+1)
        self.grid_size = ((f0,f0),(f1,f1),(f2,f2),(f3,f3),(f4,f4))
        self.num_anchors_per_grid = len(cfg.MODEL.RPN.ASPECT_RATIOS)
        self.img_shape = (img_size, img_size)
        self.device = cfg.MODEL.DEVICE
        if self.device == 'mlu':
          anchors = self.rpn.anchor_generator.forward_for_mlu(self.img_shape, self.grid_size, self.num_anchors_per_grid)
          self.Anchors_mlu = nn.Parameter(anchors)
        else:
          self.Anchors_mlu = None
          
    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        #images = to_image_list(images)
        #features = self.backbone(images.tensors)
        # modified for mlu forward
        features = self.backbone(images)
        proposals, proposal_losses = self.rpn(images, features, self.Anchors_mlu, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
