# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Requires vqa-maskrcnn-benchmark to be built and installed. See Readme
# for more details.
import argparse
import glob
import os

import cv2
import numpy as np
import logging
import torch
from PIL import Image

try:
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.layers import nms
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    from maskrcnn_benchmark.structures.image_list import to_image_list
    from maskrcnn_benchmark.utils.model_serialization import load_state_dict
except:
    raise ImportError("Please run `conda activate vilbert-mt` first!")

import tqdm

from datasets.recipeqa import RecipeQAGeneralProcessor
from datasets.mpii_movie import MPIIMovieGeneralProcessor

""" Usage:
python3 -m trainers.extract_maskrcnn_features \
  --model_file pretrained_models/vqa_maskrcnn/detectron_model.pth \
  --config_file pretrained_models/vqa_maskrcnn/detectron_config.yaml \
  --task_names "recipeqa" \
"""


processors = {
    "recipeqa": RecipeQAGeneralProcessor,
    "mpii_movie": MPIIMovieGeneralProcessor,
}

logger = logging.getLogger(__name__)


class FeatureExtractor:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.detection_model = self._build_detection_model()
        self.get_images()

        # os.makedirs(self.args.output_folder, exist_ok=True)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_file", default=None, type=str, help="Detectron model file"
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        # parser.add_argument(
        #     "--output_folder", type=str, default="./output", help="Output folder"
        # )
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        parser.add_argument(
            "--partition", type=int, default=0, help="Partition to download."
        )
        parser.add_argument(
            "--task_names",
            default=None,
            type=str,
            nargs="+",
            required=True,
            help=("The name of the task, see datasets/processors.py"),
        )
        parser.add_argument(
            "--delete_files",
            action="store_true",
            help="If delete the files created accordig to the current settings.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="If verbose.",
        )
        return parser

    def get_images(self):
        examples = []
        print("-"*50)
        for task_name in self.args.task_names:
            processor = processors[task_name](min_story_length=1,
                                              max_story_length=5)

            train_examples = processor.get_train_examples()
            val_examples = processor.get_dev_examples()
            test_examples = processor.get_test_examples()
            examples_curr = train_examples + val_examples + test_examples
            examples += examples_curr

            print("Task: {}".format(task_name))
            print("Num of Train Examples: {}".format(len(train_examples)))
            print("Num of Val   Examples: {}".format(len(val_examples)))
            print("Num of Test  Examples: {}".format(len(test_examples)))
            print("Num of All   Examples: {}".format(len(examples)))
            print("-"*50)

        self.examples = examples

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        if im.shape[-1] > 3:  # Exclude alpha channel
            im = im[:, :, :3]
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        # print (output[0][0].get_field("scores"))
        # print (output[0][0].get_field("labels"))
        # print (output[0][0].fields())
        # print (output[0][0].bbox)
        # print (output[1], output[1].size())
        labels = output[0][0].get_field("labels")
        scores = output[0][0].get_field("scores")
        batch_size = len(output[0])
        n_boxes_per_image = len(output[0][0].bbox)
        # print (n_boxes_per_image)
        # print (output[0][0].get_field("scores").size())
        score_list = torch.split(output[0][0].get_field("scores"), 1, dim=0)
        # print (score_list)
        # score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[1].split(1)
        
        cur_device = score_list[0].device

        assert n_boxes_per_image == len(output[1]) == len(score_list)

        feat_list = []
        info_list = []

        for i in range(batch_size):
            bbox = output[0][i].convert(mode="xywh").bbox
            objects = labels
            cls_prob = scores
            num_boxes = len(output[0][i].bbox)
            feat_list.append(output[1])
            """
            dets = output[0][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            # for cls_ind in range(start_index, scores.shape[1]):
            for cls_ind in range(start_index, scores.shape[0]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            print (keep_boxes, self.args.num_features);raise
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            # objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)
            objects = labels
            # cls_prob = torch.max(scores[keep_boxes][start_index:], dim=1)
            cls_prob = scores
            """

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    # "num_boxes": num_boxes.item(),
                    "num_boxes": num_boxes,
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    # "cls_prob": scores[keep_boxes].cpu().numpy(),
                    "cls_prob": scores,
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []

        for image_path in image_paths:
            im, im_scale, im_info = self._image_transform(image_path)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")

        with torch.no_grad():
            output = self.detection_model(current_img_list)
            bbox_curr = [output[0]]
            _, feats = self.detection_model(current_img_list, targets=bbox_curr, get_feats=True)
            output = (output, feats)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0].split("_maskrcnn")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = file_base_name + ".npy"

        np.save(file_name, info)
        if self.args.verbose:
            print("Save mask rcnn file at: {}".format(file_name))

    def extract_features(self):
        img_names = []
        img_embds = []
        for example in tqdm.tqdm(self.examples, total=len(self.examples)):
            img_paths = example.img_path_seq
            for img_path in img_paths:
                img_path_name, img_path_ext = os.path.splitext(img_path)
                npy_path = img_path.replace(img_path_ext,
                                            "_maskrcnn.npy")
                if self.args.delete_files:  # and args.extract_mode == "multi_files":
                    if os.path.exists(npy_path):
                        if self.args.verbose:
                            print("Found and delete file: {}".format(npy_path))
                        os.remove(npy_path)
                else:
                    if os.path.exists(npy_path):
                        continue
                    img_path = [img_path]
                    features, infos = self.get_detectron_features(img_path)
                    self._save_feature(npy_path, features[0], infos[0])
                    # print(infos[0], features[0].size())
                    # raise
        return img_names, img_embds

    def _save_featare_old(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = file_base_name + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)

    def extract_features_old(self):
        image_dir = self.args.image_dir
        if os.path.isfile(image_dir):
            print ("Processing: {}".format(image_dir))
            features, infos = self.get_detectron_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
        else:
            print ("Processing: {}".format(image_dir))
            files = glob.glob(os.path.join(image_dir, "*"))
            # files = sorted(files)
            # files = [files[i: i+1000] for i in range(0, len(files), 1000)][self.args.partition]
            i = 0
            bar = tqdm.tqdm(total=len(files) / self.args.batch_size)
            for chunk in self._chunks(files, self.args.batch_size):
                # print (i, chunk)
                # try:
                features, infos = self.get_detectron_features(chunk)
                for idx, file_name in enumerate(chunk):
                    self._save_feature(file_name, features[idx], infos[idx])
                # except BaseException:
                #     continue
                i += 1
                bar.update(1)


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
