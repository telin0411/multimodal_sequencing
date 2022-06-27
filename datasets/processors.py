import sys
import csv
import glob
import json
import logging
import random
import os
from enum import Enum
from typing import List, Optional, Union
import numpy as np
import argparse

from tqdm import tqdm, trange

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from torchvision import transforms

from .utils import InputPairWiseExample, InputPairWiseFeatures
from .utils import InputAbductiveExample
from .utils import Permutation

from .img_utils import read_and_transform_img_from_filename
from .img_utils import Rescale, RandomCrop, ToTensor

# Processors.
from .recipeqa import RecipeQAPairWiseProcessor, RecipeQAGeneralProcessor
from .recipeqa import RecipeQAAbductiveProcessor
from .wikihow import WikiHowPairWiseProcessor, WikiHowGeneralProcessor
from .wikihow import WikiHowAbductiveProcessor

from nltk.tokenize import word_tokenize

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Processors.
data_names = {
    "roc": "ROC",
    "vist": "VIST",
    "recipeqa": "RecipeQA",
    "mpii_movie": "MPIIMovie",
    "wikihow": "WikiHow"
}

task_typed_processors = {
    "pairwise": "PairWiseProcessor",
    "head": "GeneralProcessor",
    "sort": "GeneralProcessor",
    "abductive": "AbductiveProcessor",
    "pure_class": "GeneralProcessor",
    "pure_decode": "GeneralProcessor",
    "pretrain": "GeneralProcessor",
    "hl_v1": "GeneralProcessor",
    "retrieve": "GeneralProcessor",
}

data_processors = {}
output_modes = {}

for data_name in data_names:
    for task_typed_processor in task_typed_processors:
        data_task_name = data_name + "_" + task_typed_processor
        data_task_class_name = data_names[data_name] + \
            task_typed_processors[task_typed_processor]
        try:
            data_processors[data_task_name] = getattr(
                sys.modules[__name__], data_task_class_name)
        except:
            data_processors[data_task_name] = None
        output_modes[data_task_name] = "classification"
        output_modes[task_typed_processor] = "classification"
    pass
pass


def _pairwise_convert_examples_to_features(
    examples: List[InputPairWiseExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    multimodal=False,
    img_transform_func=None,
):
    if max_length is None:
        max_length = tokenizer.max_len
        
    if multimodal:
        if img_transform_func is None:
            img_transform_func = transforms.Compose([Rescale((224,224)),
                                                     ToTensor()])

    if task is not None:
        processor = data_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputPairWiseExample) -> Union[int,
                                                                   float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
    )
    if multimodal:
        batch_encoding["images"] = [None for i in range(len(examples))]
        for i in range(len(examples)):
            example = examples[i]
            imgs = []
            all_regional_features = []
            img_paths = [example.img_path_a, example.img_path_b]
            for img_path in img_paths:
                img = read_and_transform_img_from_filename(img_path, args=self.args,
                    img_transform_func=img_transform_func)
                imgs.append(img)
            imgs = torch.stack(imgs)
            batch_encoding["images"][i] = imgs

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputPairWiseFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


# TODO: Implement the dataset class for binary prediction similar to head.
# So we can also handle per sequence truncation likewise.
class PairwiseDataset(Dataset):
    """Pairwise Prediction Dataset."""

    def __init__(self, examples, tokenizer, max_length=None,
                 per_seq_max_length=32, max_story_length=5,
                 min_story_length=5, scramble=True, seed=None,
                 processor=None, output_mode="classification",
                 multimodal=False, img_transform_func=None,
                 num_img_regional_features=None, args=None):
        """
        Args:
            examples (list): input examples of type `InputHeadExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_length (int): maximum length to truncate the input ids.
            per_seq_max_length (int): maximum length to truncate the per
                sequence input ids.
            max_story_length (int): maximum length for the story sequence.
            min_story_length (int): minimum length for the story sequence.
            scrmable (bool): if scramble the sequence.
            seed (int): random seed.
            processor (class): processor.
            output_mode (str): output mode.
            multimodal (bool): if multimodal.
            img_transform_func (function): the function for image
                transformation.
            num_img_regional_features (int): the number of detection image
                regional features to use.
        """
        self.args = args
        self.examples = examples
        self.tokenizer = tokenizer
        self.scramble = scramble
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed
        self.max_length = max_length
        self.per_seq_max_length = per_seq_max_length
        self.multimodal = multimodal
        self.img_transform_func = img_transform_func
        if self.img_transform_func is None:  # At least to tensor.
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.img_transform_func = transforms.Compose([Rescale((224,224)),
                                                          ToTensor(),
                                                          normalize])

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        label_list = processor.get_labels()
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.output_mode = output_mode
        self.num_img_regional_features = num_img_regional_features

        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    def label_from_example(self, example: InputPairWiseExample) -> Union[int,
                                                                         float,
                                                                         None]:
        if example.label is None:
            return None
        if self.output_mode == "classification":
            return self.label_map[example.label]
        elif self.output_mode == "regression":
            return float(example.label)
        raise KeyError(self.output_mode)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        story_seq = [self.examples[idx].text_a,
                     self.examples[idx].text_b]
        guid = self.examples[idx].guid

        batch_encoding = self.tokenizer(
            story_seq,
            max_length=self.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.zeros(self.max_length, dtype=int) + self.pad_id
        padded_token_type_ids = np.zeros(self.max_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=self.pad_id]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(self.max_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()

        label = self.label_from_example(self.examples[idx])
        label = torch.Tensor([label]).long()[0]

        if self.multimodal:
            imgs = []
            all_regional_features = []
            img_paths = [self.examples[idx].img_path_a,
                         self.examples[idx].img_path_b]
            for img_path in img_paths:
                img = read_and_transform_img_from_filename(img_path, args=self.args,
                    img_transform_func=self.img_transform_func)
                if self.num_img_regional_features is not None and False:
                    img_place, img_path_ext = os.path.splitext(img_path)
                    maskrcnn_path = img_place + "_maskrcnn.npy"
                    # TODO: remove this
                    # if True:
                    if not os.path.exists(maskrcnn_path):
                        # raise ValueError("No such file {}".format(maskrcnn_path))
                        pass
                    else:
                        maskrcnn_d = np.load(maskrcnn_path, allow_pickle=True)
                        maskrcnn_d = maskrcnn_d.item()
                        regional_features = maskrcnn_d["features"][:self.num_img_regional_features]
                        regional_features = torch.Tensor(regional_features).type_as(img)
                        all_regional_features.append(regional_features)
                imgs.append(img)
            imgs = torch.stack(imgs)
            if self.num_img_regional_features:
                if len(all_regional_features) > 0:
                    all_regional_features = torch.stack(all_regional_features)
                else:
                    all_regional_features = torch.Tensor([0])
                return (input_ids, attention_mask, token_type_ids, label,
                        all_regional_features, guid, imgs)
            return (input_ids, attention_mask, token_type_ids, label, guid, imgs)

        return (input_ids, attention_mask, token_type_ids, label, guid)


class HeadPredDataset(Dataset):
    """Head Prediction Dataset."""

    def __init__(self, examples, tokenizer, max_length=None,
                 per_seq_max_length=32, max_story_length=5,
                 min_story_length=5, scramble=True, seed=None,
                 multimodal=False, img_transform_func=None,
                 num_img_regional_features=None, args=None):
        """
        Args:
            examples (list): input examples of type `InputHeadExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_length (int): maximum length to truncate the input ids.
            per_seq_max_length (int): maximum length to truncate the per
                sequence input ids.
            max_story_length (int): maximum length for the story sequence.
            min_story_length (int): minimum length for the story sequence.
            scrmable (bool): if scramble the sequence.
            seed (int): random seed.
            multimodal (bool): if multimodal.
            img_transform_func (function): the function for image
                transformation.
            num_img_regional_features (int): the number of detection image
                regional features to use.
        """
        self.args = args
        self.examples = examples
        self.tokenizer = tokenizer
        self.scramble = scramble
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed
        self.max_length = max_length
        self.per_seq_max_length = per_seq_max_length
        self.multimodal = multimodal
        self.img_transform_func = img_transform_func
        if self.img_transform_func is None:  # At least to tensor.
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.img_transform_func = transforms.Compose([Rescale((224,224)),
                                                          ToTensor(),
                                                          normalize])

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length
        self.num_img_regional_features = num_img_regional_features

        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        story_seq = self.examples[idx].text_seq
        story_seq = story_seq[:self.max_story_length]
        idx_seq = np.arange(len(story_seq))
        if self.scramble:
            np.random.shuffle(idx_seq)
            story_seq_new = [story_seq[idx_seq[i]]
                             for i in range(len(story_seq))]
            story_seq = story_seq_new
        head_idx = int(np.argwhere(idx_seq==0)[0][0])

        batch_encoding = self.tokenizer(
            story_seq,
            max_length=self.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.zeros(self.max_length, dtype=int) + self.pad_id
        padded_token_type_ids = np.zeros(self.max_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=self.pad_id]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(self.max_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()
        label = torch.Tensor([head_idx]).long()[0]
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # print(label)
        if self.multimodal:
            imgs = []
            all_regional_features = []
            img_path_seq = self.examples[idx].img_path_seq
            img_path_seq = [img_path_seq[idx_seq[i]]
                            for i in range(len(story_seq))]
            for img_path in img_path_seq:
                img = read_and_transform_img_from_filename(img_path, args=self.args,
                    img_transform_func=self.img_transform_func)
                if self.num_img_regional_features is not None:
                    raise NotImplementedError("Not done yet!")
                    img_place, img_path_ext = os.path.splitext(img_path)
                    maskrcnn_path = img_place + "_maskrcnn.npy"
                    maskrcnn_d = np.load(maskrcnn_path, allow_pickle=True)
                    maskrcnn_d = maskrcnn_d.item()
                    regional_features = maskrcnn_d["features"][:self.num_img_regional_features]
                    regional_features = torch.Tensor(regional_features).type_as(img)
                    all_regional_features.append(regional_features)
                imgs.append(img)
            imgs = torch.stack(imgs)
            if self.num_img_regional_features:
                all_regional_features = torch.stack(all_regional_features)
                return (input_ids, attention_mask, token_type_ids, label,
                        all_regional_features, imgs)
            return (input_ids, attention_mask, token_type_ids, label, imgs)

        return (input_ids, attention_mask, token_type_ids, label)


class AbductiveDataset(Dataset):
    """Abductive Reasoning Dataset."""

    def __init__(self, examples, tokenizer, max_length=None,
                 per_seq_max_length=32, pred_method="binary",
                 multimodal=False, img_transform_funac=None,
                 num_img_regional_features=None, args=None):
        """
        Args:
            examples (list): input examples of type `InputHeadExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_length (int): maximum length to truncate the input ids.
            per_seq_max_length (int): maximum length to truncate the per
                sequence input ids.
            pred_method: the method of the predictions, can be binary or
                contrastive 
            multimodal (bool): if multimodal.
            img_transform_func (function): the function for image
                transformation.
            num_img_regional_features (int): the number of detection image
                regional features to use.
        """
        self.args = args
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.per_seq_max_length = per_seq_max_length
        self.pred_method = pred_method
        self.multimodal = multimodal
        self.img_transform_func = img_transform_func
        if self.img_transform_func is None:  # At least to tensor.
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.img_transform_func = transforms.Compose([Rescale((224,224)),
                                                          ToTensor(),
                                                          normalize])
        self.num_img_regional_features = num_img_regional_features

        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        story_seq = [self.examples[idx].text_h1, self.examples[idx].text_h2,
                     self.examples[idx].text_h3]
        
        batch_encoding = self.tokenizer(
            story_seq,
            max_length=self.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.zeros(self.max_length, dtype=int) + self.pad_id
        padded_token_type_ids = np.zeros(self.max_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=self.pad_id]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(self.max_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()
        if self.pred_method == "binary":
            if self.examples[idx].label == "unordered":
                label = torch.Tensor([0]).long()[0]
            else:
                label = torch.Tensor([1]).long()[0]
        elif self.pred_method == "contrastive":
            raise NotImplementedError("Prediction method {} not"
                                      " done yet!".format(self.pred_method))
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # print(label)
        if self.multimodal:
            imgs = []
            all_regional_features = []
            img_paths = [
                self.examples[idx].img_path_h1,
                self.examples[idx].img_path_h2,
                self.examples[idx].img_path_h3,
            ]
            for img_path in img_paths:
                img = read_and_transform_img_from_filename(img_path, args=self.args,
                    img_transform_func=self.img_transform_func)
                if self.num_img_regional_features is not None:
                    raise NotImplementedError("Not done yet!")
                    img_place, img_path_ext = os.path.splitext(img_path)
                    maskrcnn_path = img_place + "_maskrcnn.npy"
                    maskrcnn_d = np.load(maskrcnn_path, allow_pickle=True)
                    maskrcnn_d = maskrcnn_d.item()
                    regional_features = maskrcnn_d["features"][:self.num_img_regional_features]
                    regional_features = torch.Tensor(regional_features).type_as(img)
                    all_regional_features.append(regional_features)
                imgs.append(img)
            imgs = torch.stack(imgs)
            if self.num_img_regional_features:
                all_regional_features = torch.stack(all_regional_features)
                return (input_ids, attention_mask, token_type_ids, label,
                        all_regional_features, imgs)
            return (input_ids, attention_mask, token_type_ids, label, imgs)

        return (input_ids, attention_mask, token_type_ids, label)


class PureClassDataset(Dataset):
    """Pure Classification Dataset."""

    def __init__(self, examples, tokenizer, max_length=None,
                 per_seq_max_length=32, max_story_length=5,
                 min_story_length=5, multimodal=False,
                 scramble=True, seed=None, decode=False,
                 img_transform_func=None, num_img_regional_features=None,
                 args=None):
        """
        Args:
            examples (list): input examples of type `InputHeadExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_length (int): maximum length to truncate the input ids.
            per_seq_max_length (int): maximum length to truncate the per
                sequence input ids.
            max_story_length (int): maximum length for the story sequence.
            min_story_length (int): minimum length for the story sequence.
            scrmable (bool): if scramble the sequence.
            seed (int): random seed.
            decode (bool): if used as decoding dataset.
            multimodal (bool): if multimodal.
            img_transform_func (function): the function for image
                transformation.
            num_img_regional_features (int): the number of detection image
                regional features to use.
        """
        self.args = args
        self.examples = examples
        self.tokenizer = tokenizer
        self.scramble = scramble
        self.decode = decode
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed
        self.max_length = max_length
        self.per_seq_max_length = per_seq_max_length
        self.multimodal = multimodal
        self.img_transform_func = img_transform_func
        if self.img_transform_func is None:  # At least to tensor.
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.img_transform_func = transforms.Compose([Rescale((224,224)),
                                                          ToTensor(),
                                                          normalize])

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        max_story_length = min(max_story_length,
                               len(self.examples[0].text_seq))
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length
        self.num_img_regional_features = num_img_regional_features

        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        self._construct_labels()

    def _construct_labels(self):
        indices = list(range(self.max_story_length))
        self.label2id = {}
        self.id2label = {}
        self._perm = Permutation()

        curr_id = 0
        while True:
            indices_str = [str(x) for x in indices]
            indices_str = '_'.join(indices_str)
            if indices_str in self.label2id:
                break
            self.label2id[indices_str] = curr_id
            self.id2label[curr_id] = indices
            indices = self._perm.nextPermutation(indices)
            curr_id += 1
        pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        story_seq = self.examples[idx].text_seq
        story_seq = story_seq[:self.max_story_length]
        guid = self.examples[idx].guid
        idx_seq = np.arange(len(story_seq))
        if self.scramble:
            np.random.shuffle(idx_seq)
            story_seq_new = [story_seq[idx_seq[i]]
                             for i in range(len(story_seq))]
            story_seq = story_seq_new
        idx_seq_l = list(idx_seq)

        if self.decode:
            if self.examples[idx].multiref_gt is not None:
                multiref_gt = self.examples[idx].multiref_gt
                assert len(multiref_gt) >= 1 and type(multiref_gt) is list
                label_offset = min(multiref_gt[0])
                multiref_gt = [[x - label_offset for x in y] for y in multiref_gt]
                assert list(range(0, self.max_story_length)) in multiref_gt, (
                    "Forgot the original 12345 GT for data: {}?".format(guid))
                multiref_gt = sorted(multiref_gt)
                assert list(range(0, self.max_story_length)) == multiref_gt[0]
                scrambled_multiref_gt = [[x[idx] for idx in idx_seq] for x in multiref_gt]
                label = [np.argsort(np.asarray(x)) for x in scrambled_multiref_gt]
                label = np.asarray(label)
            else:
                label = np.argsort(np.asarray(idx_seq))
        else:
            idx_seq_l_str = [str(x) for x in idx_seq_l]
            idx_seq_l_str = '_'.join(idx_seq_l_str)
            label = self.label2id[idx_seq_l_str]

        batch_encoding = self.tokenizer(
            story_seq,
            max_length=self.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.zeros(self.max_length, dtype=int) + self.pad_id
        padded_token_type_ids = np.zeros(self.max_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=self.pad_id]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(self.max_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()

        if self.decode:
            label = torch.Tensor(label).long()
        else:
            label = torch.Tensor([label]).long()[0]
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # print(label)

        if self.multimodal:
            imgs = []
            all_regional_features = []
            img_path_seq = self.examples[idx].img_path_seq
            img_path_seq = [img_path_seq[idx_seq[i]]
                            for i in range(len(story_seq))]
            for img_path in img_path_seq:
                img = read_and_transform_img_from_filename(img_path, args=self.args,
                    img_transform_func=self.img_transform_func)
                if self.num_img_regional_features is not None:
                    img_place, img_path_ext = os.path.splitext(img_path)
                    maskrcnn_path = img_place + "_maskrcnn.npy"
                    if not os.path.exists(maskrcnn_path):
                        # raise ValueError("No such file {}".format(maskrcnn_path))
                        pass
                    else:
                        maskrcnn_d = np.load(maskrcnn_path, allow_pickle=True)
                        maskrcnn_d = maskrcnn_d.item()
                        regional_features = maskrcnn_d["features"][:self.num_img_regional_features]
                        regional_features = torch.Tensor(regional_features).type_as(img)
                        all_regional_features.append(regional_features)
                imgs.append(img)
            imgs = torch.stack(imgs)
            if self.num_img_regional_features:
                if len(all_regional_features) > 0:
                    all_regional_features = torch.stack(all_regional_features)
                else:
                    all_regional_features = torch.Tensor([0])
            return (input_ids, attention_mask, token_type_ids, label, guid, imgs)

        return (input_ids, attention_mask, token_type_ids, label, guid)


class SortDatasetV1(Dataset):
    """Sorting Prediction Dataset."""

    def __init__(self, examples, tokenizer, max_length=None,
                 per_seq_max_length=32, max_story_length=5,
                 min_story_length=5, multimodal=False,
                 scramble=True, seed=None, img_transform_func=None,
                 num_img_regional_features=None, args=None):
        """
        Args:
            examples (list): input examples of type `InputHeadExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_length (int): maximum length to truncate the input ids.
            per_seq_max_length (int): maximum length to truncate the per
                sequence input ids.
            max_story_length (int): maximum length for the story sequence.
            min_story_length (int): minimum length for the story sequence.
            scrmable (bool): if scramble the sequence.
            seed (int): random seed.
            multimodal (bool): if multimodal.
            img_transform_func (function): the function for image
                transformation.
            num_img_regional_features (int): the number of detection image
                regional features to use.
        """
        self.args = args
        self.examples = examples
        self.tokenizer = tokenizer
        self.scramble = scramble
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed
        self.max_length = max_length
        self.per_seq_max_length = per_seq_max_length
        self.multimodal = multimodal
        self.img_transform_func = img_transform_func
        if self.img_transform_func is None:  # At least to tensor.
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.img_transform_func = transforms.Compose([Rescale((224,224)),
                                                          ToTensor(),
                                                          normalize])

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        max_story_length = min(max_story_length,
                               len(self.examples[0].text_seq))
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length
        self.num_img_regional_features = num_img_regional_features

        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        story_seq = self.examples[idx].text_seq
        story_seq = story_seq[:self.max_story_length]
        idx_seq = np.arange(len(story_seq))
        if self.scramble:
            np.random.shuffle(idx_seq)
            story_seq_new = [story_seq[idx_seq[i]]
                             for i in range(len(story_seq))]
            story_seq = story_seq_new

        inputs = story_seq

        if self.examples[idx].multiref_gt is not None:
            multiref_gt = self.examples[idx].multiref_gt
            assert len(multiref_gt) >= 1 and type(multiref_gt) is list
            label_offset = min(multiref_gt[0])
            multiref_gt = [[x - label_offset for x in y] for y in multiref_gt]
            assert list(range(0, self.max_story_length)) in multiref_gt, (
                "Forgot the original 12345 GT for data: {}?".format(guid))
            multiref_gt = sorted(multiref_gt)
            assert list(range(0, self.max_story_length)) == multiref_gt[0]
            scrambled_multiref_gt = [[x[idx] for idx in idx_seq] for x in multiref_gt]
            label = [np.argsort(np.asarray(x)) for x in scrambled_multiref_gt]
            label = np.asarray(label)
        else:
            label = np.argsort(np.asarray(idx_seq))

        if self.multimodal:
            imgs = []
            all_regional_features = []
            img_path_seq = self.examples[idx].img_path_seq
            img_path_seq = [img_path_seq[idx_seq[i]]
                            for i in range(len(story_seq))]
            for img_path in img_path_seq:
                img = read_and_transform_img_from_filename(img_path, args=self.args,
                    img_transform_func=self.img_transform_func)
                if self.num_img_regional_features is not None:
                    img_place, img_path_ext = os.path.splitext(img_path)
                    maskrcnn_path = img_place + "_maskrcnn.npy"
                    if not os.path.exists(maskrcnn_path):
                        # raise ValueError("No such file {}".format(maskrcnn_path))
                        pass
                    else:
                        maskrcnn_d = np.load(maskrcnn_path, allow_pickle=True)
                        maskrcnn_d = maskrcnn_d.item()
                        regional_features = maskrcnn_d["features"][:self.num_img_regional_features]
                        regional_features = torch.Tensor(regional_features).type_as(img)
                        all_regional_features.append(regional_features)
                imgs.append(img)
            imgs = torch.stack(imgs)
            if self.num_img_regional_features:
                if len(all_regional_features) > 0:
                    all_regional_features = torch.stack(all_regional_features)
                else:
                    all_regional_features = torch.Tensor([0])
                return (inputs, label, imgs, all_regional_features)
            return (inputs, label, imgs)

        return (inputs, label)


class PretrainDataset(Dataset):
    """Pretraining Dataset."""

    def __init__(self, examples, tokenizer, max_length=None,
                 per_seq_max_length=32, max_story_length=5,
                 min_story_length=5, scramble=False, seed=None,
                 multimodal=False, img_transform_func=None,
                 visual_token_dict_path=None, data_names=None,
                 num_img_regional_features=None, args=None, get_guid=False):
        """
        Args:
            examples (list): input examples of type `InputHeadExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_length (int): maximum length to truncate the input ids.
            per_seq_max_length (int): maximum length to truncate the per
                sequence input ids.
            max_story_length (int): maximum length for the story sequence.
            min_story_length (int): minimum length for the story sequence.
            scrmable (bool): if scramble the sequence.
            seed (int): random seed.
            multimodal (bool): if multimodal.
            img_transform_func (function): the function for image
                transformation.
            visual_token_dict_path (str): the path to the visual token dict
                which stores the results of hkmeans quantizations.
            data_names (list): list of data names.
            num_img_regional_features (int): the number of detection image
                regional features to use.
        """
        self.args = args
        self.examples = examples
        self.tokenizer = tokenizer
        self.scramble = scramble
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed
        self.max_length = max_length
        self.per_seq_max_length = per_seq_max_length
        self.multimodal = multimodal
        self.img_transform_func = img_transform_func
        if self.img_transform_func is None:  # At least to tensor.
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.img_transform_func = transforms.Compose([Rescale((224,224)),
                                                          ToTensor(),
                                                          normalize])
        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        self.data_names = data_names
        self.vt_dict = None
        if self.multimodal and visual_token_dict_path is not None:
            self.visual_token_dict_path = visual_token_dict_path
            self.prepare_visual_tokens()
            self.vt_dict = True
        self.num_img_regional_features = num_img_regional_features

        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.get_guid = get_guid

    def __len__(self):
        return len(self.examples)

    def prepare_visual_tokens(self):
        for data_name in self.data_names:
            if data_name not in self.visual_token_dict_path:
                return
                raise ValueError("Dataset {} might not be in this dict "
                    "{}".format(data_name, self.visual_token_dict_path))
        d = np.load(self.visual_token_dict_path, allow_pickle=True)
        d = d.item()

        vt_dict = {}
        img_names = list(sorted(d.keys()))
        for img_name in img_names:
            visual_tkn = d.get(img_name)
            img_name_woext, ext  = os.path.splitext(img_name)
            vt_dict[img_name_woext] = visual_tkn
        self.vt_dict = vt_dict

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        story_seq = self.examples[idx].text_seq
        story_seq = story_seq[:self.max_story_length]
        idx_seq = np.arange(len(story_seq))
        if self.scramble:
            np.random.shuffle(idx_seq)
            story_seq_new = [story_seq[idx_seq[i]]
                             for i in range(len(story_seq))]
            story_seq = story_seq_new
        head_idx = int(np.argwhere(idx_seq==0)[0][0])

        batch_encoding = self.tokenizer(
            story_seq,
            max_length=self.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.zeros(self.max_length, dtype=int) + self.pad_id
        padded_token_type_ids = np.zeros(self.max_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=self.pad_id]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(self.max_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()
        label = torch.Tensor([head_idx]).long()[0]

        if self.get_guid:
            if self.scramble:
                input_ids = (input_ids, self.examples[idx].guid, idx_seq)
            else:
                input_ids = (input_ids, self.examples[idx].guid)

        if self.multimodal:
            imgs = []
            all_regional_features = []
            img_path_seq = self.examples[idx].img_path_seq
            img_path_seq = [img_path_seq[idx_seq[i]]
                            for i in range(len(story_seq))]
            vis_tokens = []
            for img_path in img_path_seq:
                img = read_and_transform_img_from_filename(img_path, args=self.args,
                    img_transform_func=self.img_transform_func)
                if self.num_img_regional_features is not None:
                    img_place, img_path_ext = os.path.splitext(img_path)
                    maskrcnn_path = img_place + "_maskrcnn.npy"
                    if not os.path.exists(maskrcnn_path):
                        # raise ValueError("No such file {}".format(maskrcnn_path))
                        pass
                    else:
                        maskrcnn_d = np.load(maskrcnn_path, allow_pickle=True)
                        maskrcnn_d = maskrcnn_d.item()
                        regional_features = maskrcnn_d["features"][:self.num_img_regional_features]
                        regional_features = torch.Tensor(regional_features).type_as(img)
                        all_regional_features.append(regional_features)
                imgs.append(img)
                # TODO: Add some conditions here.
                if False:
                    img_path_, ext = os.path.splitext(img_path)
                    if self.vt_dict is not None:
                        if img_path_ not in self.vt_dict:
                            raise ValueError("{} not in visual token dict "
                                "{}".format(img_path, self.visual_token_dict_path))
                        vis_token = self.vt_dict[img_path_]
                        vis_tokens.append(vis_token)
            imgs = torch.stack(imgs)
            if self.vt_dict is not None:
                vis_tokens = torch.Tensor(vis_tokens).long()
            else:
                vis_tokens = torch.Tensor([0]).long()

            # TODO: Add some conditions here.
            if True:
                if self.num_img_regional_features:
                    if len(all_regional_features) > 0:
                        all_regional_features = torch.stack(all_regional_features)
                    else:
                        all_regional_features = torch.Tensor([0])
                    return (input_ids, attention_mask, token_type_ids, label,
                            vis_tokens, all_regional_features, imgs)
                return (input_ids, attention_mask, token_type_ids, label,
                        vis_tokens, imgs)
            if self.num_img_regional_features:
                if len(all_regional_features) > 0:
                    all_regional_features = torch.stack(all_regional_features)
                else:
                    all_regional_features = torch.Tensor([0])
                return (input_ids, attention_mask, token_type_ids, label,
                        all_regional_features, imgs)
            return (input_ids, attention_mask, token_type_ids, label, imgs)

        return (input_ids, attention_mask, token_type_ids, label)


class RetrievalDataset(Dataset):
    """Pretraining Dataset."""

    def __init__(self, examples, tokenizer, max_length=None,
                 per_seq_max_length=32, max_story_length=5,
                 min_story_length=5, scramble=False, seed=None,
                 multimodal=False, img_transform_func=None,
                 data_names=None, num_img_regional_features=None,
                 args=None):
        """
        Args:
            examples (list): input examples of type `InputHeadExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_length (int): maximum length to truncate the input ids.
            per_seq_max_length (int): maximum length to truncate the per
                sequence input ids.
            max_story_length (int): maximum length for the story sequence.
            min_story_length (int): minimum length for the story sequence.
            scrmable (bool): if scramble the sequence.
            seed (int): random seed.
            multimodal (bool): if multimodal.
            img_transform_func (function): the function for image
                transformation.
            visual_token_dict_path (str): the path to the visual token dict
                which stores the results of hkmeans quantizations.
            data_names (list): list of data names.
            num_img_regional_features (int): the number of detection image
                regional features to use.
        """
        self.args = args
        self.examples = examples
        self.tokenizer = tokenizer
        self.scramble = scramble
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed
        self.max_length = max_length
        self.per_seq_max_length = per_seq_max_length
        self.multimodal = multimodal
        self.img_transform_func = img_transform_func
        if self.img_transform_func is None:  # At least to tensor.
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.img_transform_func = transforms.Compose([Rescale((224,224)),
                                                          ToTensor(),
                                                          normalize])
        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        self.data_names = data_names
        self.vt_dict = None
        self.num_img_regional_features = num_img_regional_features

        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    def __len__(self):
        return len(self.examples)

    def candidates_list(self):

        all_candidates = []

        print("Processing candidates...")

        cand_iterator = tqdm(self.examples, desc="Candidates")
        for i, example in enumerate(cand_iterator):
            story_seq = example.text_seq
            img_path_seq = example.img_path_seq
            story_seq = story_seq[:self.max_story_length]
            guid = example.guid

            for j in range(len(story_seq)):

                guid_j = guid + "###{}".format(j)
                story_item = story_seq[j]
                batch_encoding = self.tokenizer(
                    story_item,
                    max_length=self.per_seq_max_length,
                    padding="max_length",
                    truncation=True,
                )
                input_ids = batch_encoding["input_ids"]
                attention_mask = batch_encoding["attention_mask"]

                if self.multimodal:
                    img_path = img_path_seq[j]
                    imgs = []
                    img = read_and_transform_img_from_filename(img_path, args=self.args,
                        img_transform_func=self.img_transform_func)
                    imgs.append(img)
                    imgs = torch.stack(imgs)
                    item = (input_ids, attention_mask, guid_j, imgs)
                else:
                    item = (input_ids, attention_mask, guid_j)

                all_candidates.append(item)

        return all_candidates

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        story_seq = self.examples[idx].text_seq
        story_seq = story_seq[:self.max_story_length]
        idx_seq = np.arange(len(story_seq))
        random_skip = np.random.randint(0, 5)

        guid = self.examples[idx].guid
        batch_encoding = self.tokenizer(
            story_seq,
            max_length=self.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.zeros(self.max_length, dtype=int) + self.pad_id
        padded_token_type_ids = np.zeros(self.max_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        guid += "###{}".format(random_skip)

        label = []
        for i in range(len(seqs_input_ids)):
            if random_skip == i:
                continue
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=self.pad_id]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
            label.append(i)

        max_length = min(self.max_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()

        label += [random_skip]
        arg_sort_label = np.argsort(np.asarray(label))
        label = arg_sort_label

        if self.multimodal:
            imgs = []
            all_regional_features = []
            img_path_seq = self.examples[idx].img_path_seq
            img_path_seq = [img_path_seq[idx_seq[i]]
                            for i in range(len(story_seq))]
            for img_i in range(len(img_path_seq)):
                if img_i == random_skip:
                    continue
                img_path = img_path_seq[i]
                img = read_and_transform_img_from_filename(img_path, args=self.args,
                    img_transform_func=self.img_transform_func)
                imgs.append(img)
            imgs = torch.stack(imgs)
            if self.num_img_regional_features:
                if len(all_regional_features) > 0:
                    all_regional_features = torch.stack(all_regional_features)
                else:
                    all_regional_features = torch.Tensor([0])
                return (input_ids, attention_mask, token_type_ids, label,
                        all_regional_features, imgs)
            return (input_ids, attention_mask, token_type_ids, label, guid, imgs)

        return (input_ids, attention_mask, token_type_ids, label, guid, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_task", type=str, default=None,
                        choices=["roc", "recipeqa", "mpii_movie"],
                        help="Testing Task.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "pretrained_models/roberta/large",
        do_lower_case=True,
        cache_dir=None,
    )

    if args.testing_task == "roc":

        pairwise_proc = ROCPairWiseProcessor()
        test_examples = pairwise_proc.get_test_examples()

        feats = _pairwise_convert_examples_to_features(
            examples=test_examples,
            tokenizer=tokenizer,
            max_length=512,
            task="roc_pairwise",
            output_mode="classification",
        )

        head_proc = ROCGeneralProcessor()
        abd_proc = ROCAbductiveProcessor()
        cls_proc = ROCGeneralProcessor()

        multimodal = False
        img_transform_func = None

    elif args.testing_task in ["recipeqa", "mpii_movie"]:
        pairwise_proc = data_processors["{}_pairwise".format(
            args.testing_task)]()
        test_examples = pairwise_proc.get_test_examples()

        feats = _pairwise_convert_examples_to_features(
            examples=test_examples,
            tokenizer=tokenizer,
            max_length=512,
            task="{}_pairwise".format(args.testing_task),
            output_mode="classification",
            # multimodal=True
        )

        head_proc = data_processors["{}_head".format(args.testing_task)]()
        abd_proc = data_processors["{}_abductive".format(args.testing_task)]()
        cls_proc = data_processors["{}_pure_class".format(args.testing_task)]()

        multimodal = True
        img_transform_func = transforms.Compose([Rescale((224,224)),
                                                 ToTensor()])

    # Data processing.

    test_examples = head_proc.get_test_examples()

    head_dataset = HeadPredDataset(test_examples, tokenizer,
                                   max_length=64, max_story_length=5,
                                   seed=42, multimodal=multimodal,
                                   img_transform_func=img_transform_func)
    print(len(head_dataset))
    sampler = RandomSampler(head_dataset)
    # sampler = DistributedSampler(head_dataset)
    dataloader = DataLoader(head_dataset, sampler=sampler, batch_size=4)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                  "token_type_ids": batch[2], "labels": batch[3]}
        for inp in inputs:
            print(inp, inputs[inp])
        if multimodal:
            print(batch[-1].size())
        break

    sort_dataset = SortDatasetV1(test_examples, tokenizer,
                                 max_length=64, max_story_length=5,
                                 seed=42, multimodal=multimodal,
                                 img_transform_func=img_transform_func)
    sampler = RandomSampler(sort_dataset)
    # sampler = DistributedSampler(sort_dataset)
    dataloader = DataLoader(sort_dataset, sampler=sampler, batch_size=1)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print(batch[0])
        print(batch[1])
        if multimodal:
            print(batch[-1].size())
        break

    test_examples = abd_proc.get_test_examples()
    abd_dataset = AbductiveDataset(test_examples, tokenizer, max_length=64,
                                   multimodal=multimodal,
                                   img_transform_func=img_transform_func)
    print(len(abd_dataset))
    sampler = RandomSampler(abd_dataset)
    dataloader = DataLoader(abd_dataset, sampler=sampler, batch_size=2)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print(batch[0])
        print(batch[1])
        print(batch[3])
        if multimodal:
            print(batch[-1].size())
        break
    
    test_examples = cls_proc.get_test_examples()
    cls_dataset = PureClassDataset(test_examples, tokenizer,
                                   max_length=64, decode=True,
                                   seed=42, multimodal=multimodal,
                                   img_transform_func=img_transform_func)
    print(len(cls_dataset))
    sampler = RandomSampler(cls_dataset)
    dataloader = DataLoader(cls_dataset, sampler=sampler, batch_size=2)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print(batch[0])
        print(batch[1])
        print(batch[3])
        if multimodal:
            print(batch[-1].size())
        break

    test_examples = pairwise_proc.get_test_examples()
    pairwise_dataset = PairwiseDataset(test_examples, tokenizer,
                                       max_length=64, seed=42,
                                       multimodal=multimodal,
                                       processor=pairwise_proc,
                                       img_transform_func=img_transform_func)
    print(len(cls_dataset))
    sampler = RandomSampler(pairwise_dataset)
    dataloader = DataLoader(pairwise_dataset, sampler=sampler, batch_size=2)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print(batch[0])
        print(batch[1])
        print(batch[3])
        if multimodal:
            print(batch[-1].size())
        break
    
    test_examples = head_proc.get_test_examples()
    visual_token_dict_path = ("inspection_results/mpii_movie_resnet18"
                              "_hkmeans_10_10_10_iter500.npy")
    pret_dataset = PretrainDataset(test_examples, tokenizer,
                                   max_length=64, max_story_length=5,
                                   seed=42, multimodal=multimodal,
                                   img_transform_func=img_transform_func,
                                   visual_token_dict_path=visual_token_dict_path,
                                   data_names=[args.testing_task])
    sampler = RandomSampler(pret_dataset)
    # sampler = DistributedSampler(pret_dataset)
    dataloader = DataLoader(pret_dataset, sampler=sampler, batch_size=1)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print(batch[0])
        print(batch[1])
        if multimodal:
            print(batch[-1].size())
        break

    pass
    exit(-1)

    # Testing multi processors.
    if True:
        examples = []
        for task in ["recipeqa", "mpii_movie"]:
            head_proc = data_processors["{}_head".format(task)]()
            examples_curr = head_proc.get_train_examples()
            examples += examples_curr[:4]

        head_dataset = HeadPredDataset(examples, tokenizer,
                                       max_length=64, max_story_length=5,
                                       seed=42, multimodal=True,
                                       img_transform_func=img_transform_func)
        print(len(head_dataset))
        sampler = RandomSampler(head_dataset)
        # sampler = DistributedSampler(head_dataset)
        dataloader = DataLoader(head_dataset, sampler=sampler, batch_size=4)
        epoch_iterator = tqdm(dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2], "labels": batch[3]}
            for inp in inputs:
                print(inp, inputs[inp])
            if multimodal:
                print(batch[-1].size())
            break
