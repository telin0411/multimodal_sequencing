import csv
import glob
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm

from dataclasses import dataclass
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from skimage import io, transform
from torchvision import transforms, utils
import torchvision.models as models
from datasets.img_utils import Rescale, RandomCrop, ToTensor

# Detectron2
import numpy as np
import cv2
import torch
import detectron2
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import ImageList


class Detectron2Wrapper(torch.nn.Module):
    def __init__(self, config):
        super(Detectron2Wrapper, self).__init__()
        self.config = config
        self.cfg = get_cfg()
        config_name = config.vision_model
        config_name = config_name.split("detectron2_")[-1]
        self.cfg.merge_from_file(model_zoo.get_config_file(config_name))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
        # self.model.proposal_generator = torch.nn.Identity()
        # self.model.roi_heads = torch.nn.Identity()
        if config.include_num_img_regional_features is None:
            self.model = self.model.backbone
        else:
            self.include_num_img_regional_features = config.include_num_img_regional_features

        if config.include_full_img_features:
            self.avg_pool = torch.nn.AvgPool2d(2, stride=2)  # 1/2X
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)  # 1/2X

    def forward(self, images):
        bz = images.size(0)
        """ Construct the list of dict batch if not just using the backbone
        batch = []
        for i in range(len(images)):
            item = {
                "image": images[i],
            }
            batch.append(item)
        x = self.model(batch)
        print()
        print(x)
        print()
        raise
        """

        if False:
            self.model.proposal_generator.training = False
            self.model.roi_heads.training = False
            images = ImageList.from_tensors([images[i] for i in range(bz)])
            features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(images, features)
            instances, _ = self.model.roi_heads(images, features, proposals)
            box_features = [features[f] for f in self.model.roi_heads.in_features]
            box_features = self.model.roi_heads.box_pooler(box_features,
                [x.pred_boxes for x in instances])
            roi_pooled_features = self.model.roi_heads.box_head(box_features)
            roi_pooled_features = torch.reshape(roi_pooled_features, (bz, -1,
                self.config.vision_feature_dim))
            top_k = self.config.include_num_img_regional_features
            top_k_features = roi_pooled_features[:, :top_k, :]
            if self.config.include_full_img_features:
                full_image_features = self.avg_pool(features["p6"])
                full_image_features = torch.reshape(full_image_features, (bz, -1))
                encoded_images = full_image_features
                encoded_regional_features = top_k_features[:, 0:top_k, :]

        if self.config.include_num_img_regional_features is not None:
            self.model.proposal_generator.training = False
            self.model.roi_heads.training = False
            # self.model.eval()
            images = ImageList.from_tensors([images[i] for i in range(bz)])
            features = self.model.backbone(images.tensor)
            if self.config.include_full_img_features:
                full_image_features = self.avg_pool(features["p6"])
                full_image_features = torch.reshape(full_image_features, (bz, -1))
                roi_feat_list = []
            proposals, _ = self.model.proposal_generator(images, features)
            instances, _ = self.model.roi_heads(images, features, proposals)
            box_features = [features[f] for f in self.model.roi_heads.in_features]
            # box_features = self.model.roi_heads.box_pooler(box_features,
            #     [x.pred_boxes for x in instances])
            for i in range(len(instances)):
                instance = instances[i]
                pred_boxes = instance.pred_boxes
                box_feature = [features[f][i:i+1] for f in self.model.roi_heads.in_features]
                box_feature = self.model.roi_heads.box_pooler(box_feature,
                    [pred_boxes])
                roi_pooled_feature = self.model.roi_heads.box_head(box_feature)
                for j in range(1+self.include_num_img_regional_features-len(pred_boxes[:self.include_num_img_regional_features])):
                    roi_feat_list.append(full_image_features[i])
                for j in range(len(pred_boxes[:self.include_num_img_regional_features])):
                    roi_feat_list.append(roi_pooled_feature[j])

            roi_pooled_features = torch.stack(roi_feat_list)
            # roi_pooled_features = self.model.roi_heads.box_head(box_features)
            # print(roi_pooled_features.size())
            # print(features["p6"].size())
            # print(proposals[0].objectness_logits.size())
            # for x in instances:
            #     print(x)
            # print(instances)
            # box_lists = [proposals[i].proposal_boxes for i in range(bz)]
            # features_lists = [features[f] for f in self.model.roi_heads.box_in_features]
            # rois = self.model.roi_heads.box_pooler(features_lists, box_lists)
            # roi_pooled_features = self.model.roi_heads.box_head(rois)
            # preds = self.model.roi_heads.box_predictor(roi_pooled_features)
            # print(rois.size())
            # print(roi_pooled_features.size())
            # print(preds)
            # print(torch.sum(self.model.backbone.bottom_up.res2[0].conv1.weight))
            # print(torch.sum(self.model.roi_heads.box_head.fc1.weight))
            # print(features["p6"].size())
            # print(features["p6"])
            roi_pooled_features = torch.reshape(roi_pooled_features, (bz, -1,
                self.config.vision_feature_dim))
            top_k = self.config.include_num_img_regional_features
            top_k_features = roi_pooled_features[:, :top_k, :]
            # print(top_k_features.size())
            if self.config.include_full_img_features:
                # full_image_features = self.avg_pool(features["p6"])
                # full_image_features = torch.reshape(full_image_features, (bz, -1))
                encoded_images = full_image_features
                encoded_regional_features = top_k_features[:, 0:top_k, :]
            else:
                encoded_images = top_k_features[:, 0, :]
                encoded_regional_features = top_k_features[:, 1:top_k+1, :]
            # self.model.train()
            return encoded_images, encoded_regional_features
        else:
            # print(torch.sum(self.model.fpn_output5.weight))
            # print(torch.sum(self.model.bottom_up.res2[0].conv1.weight))
            encoded_images = self.model(images)
            # print(encoded_images.keys())
            # print(encoded_images["p2"].size())
            # print(encoded_images["p3"].size())
            # print(encoded_images["p4"].size())
            # print(encoded_images["p5"].size())
            # print(encoded_images["p6"].size());raise
            encoded_images = encoded_images["p6"]
            if self.config.include_full_img_features:
                encoded_images = self.avg_pool(encoded_images)
            encoded_images = torch.reshape(encoded_images, (bz, -1))

        return encoded_images


class Detectron2ImageTransform(object):
    def __init__(self, cfg, size):
        self.cfg = cfg
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img, (self.size, self.size)).astype(np.float32)
        img -= self.cfg.MODEL.PIXEL_MEAN
        return img


def get_multimodal_utils(args=None):
    if args is None:
        vision_model = "resnet18"
    else:
        vision_model = args.vision_model

    if "detectron2" in args.vision_model:
        model = Detectron2Wrapper(args)
        img_transform_func = transforms.Compose([
            Detectron2ImageTransform(model.cfg, 256),
            ToTensor(),
        ])
    else:

        if "resnet" in args.vision_model:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            # TODO: Verify we can use our own ToTensor() function.
            img_transform_func = transforms.Compose([
                Rescale((224, 224)),
                # Rescale((256, 256)),
                ToTensor(),
                # transforms.RandomCrop((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.2),
                # transforms.RandomVerticalFlip(p=0.2),
                normalize,
            ])

        if vision_model == "resnet18":
            model = models.resnet18(pretrained=True)
        elif vision_model == "resnet50":
            model = models.resnet50(pretrained=True)
        elif vision_model == "resnet101":
            model = models.resnet101(pretrained=True)

    """
    if args.vision_model_checkpoint is not None:
        ckpt = torch.load(args.vision_model_checkpoint)
        vision_parts = {}
        for k, v in ckpt.items():
            if "vision_model" in k:
                k_new = k.partition(
                    "{}.".format("vision_model"))[-1]
                vision_parts[k_new] = v
        model.load_state_dict(vision_parts, strict=False)
    """

    return model, img_transform_func
