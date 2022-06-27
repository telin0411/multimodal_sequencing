# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning models on sorting task (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_xnli.py`"""


import sys
import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    EncoderDecoderModel, EncoderDecoderConfig,
)
from transformers.file_utils import is_sklearn_available, requires_sklearn
from datasets.processors import _pairwise_convert_examples_to_features
from datasets.processors import data_processors as processors
from datasets.processors import output_modes

# Custom dataset classes.
from datasets.processors import HeadPredDataset
from datasets.processors import SortDatasetV1
from datasets.processors import PureClassDataset

# Sorting methods.
from .topological_sort import Graph

# Metrics.
from .metrics import compute_metrics

# Multimodality.
from .multimodal_utils import get_multimodal_utils

# Some training utils.
from .train_utils import heatmap2order, render_order_heatmap

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_multimodal_models(args, tokenizer, ckpt_paths=None, configs=None,
                          language_models=None):
    from models.naive_model import NaiveMultimodalModel
    from models.vilbert.vilbert import VILBertForVLTasks
    from models.vilbert.vilbert import BertConfig as VILBertConfig
    new_models = []

    if "pure_decode" in args.sort_method:
        raise NotImplementedError("Does not handle yet!")

    if language_models is None:
        for i in range(len(ckpt_paths)):
            ckpt_path = ckpt_paths[i]
            config = configs[i]
            vision_model, img_transform_func = get_multimodal_utils(args=args)
            if args.use_multimodal_model and not args.multimodal:
                vision_model, img_transform_func = None, None
            args.img_transform_func = img_transform_func

            if args.multimodal_model_type in ["naive", "naive_model"]:
                language_model = AutoModelForSequenceClassification.from_pretrained(
                    "pretrained_models/roberta/large",
                    from_tf=False,
                    config=config,
                )
                model = NaiveMultimodalModel(args=args,
                                             language_model=language_model,
                                             vision_model=vision_model)
                model_to_load = model.module if hasattr(model, 'module') else model
                bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
                model_to_load.load_state_dict(torch.load(bin_path))

            elif args.multimodal_model_type == "visualbert":
                from models.visualbert.visual_bert_mmf import VisualBERT
                from transformers import BertConfig
                # More configs.
                args.config_file = os.path.join(ckpt_path,
                                                "config.json")
                config = BertConfig.from_json_file(args.config_file)
                config.wrapper_model_type = args.wrapper_model_type

                model = VisualBERT.from_pretrained(ckpt_path,
                    vision_model=vision_model, tokenizer=tokenizer,
                    multimodal_text_part=args.multimodal_text_part,
                    multimodal_img_part=args.multimodal_img_part,
                    additional_config=config)

            elif (args.multimodal_model_type == "uniter"
                  or args.multimodal_model_type == "UNITER"):
                from models.UNITER.model.vcr import UniterForVisualCommonsenseReasoning
                from models.UNITER.model.vqa import UniterForVisualQuestionAnswering
                from transformers import BertConfig
                model_config = "models/UNITER/config/train-vcr-large-4gpu.json"
                model_config = "models/UNITER/config/train-vqa-large-8gpu.json"
                model_config = "models/UNITER/config/uniter-base.json"
                model_config = "models/UNITER/config/uniter-large.json"
                checkpoint = torch.load(ckpt_path)
                IMG_DIM = 2048
                if vision_model is not None:
                    IMG_DIM = vision_model.fc.in_features
                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
                # model = UniterForVisualCommonsenseReasoning.from_pretrained(
                #     model_config, checkpoint, img_dim=IMG_DIM)

                # More configs.
                args.config_file = os.path.join(ckpt_path,
                                                "config.json")
                args.config_file = ckpt_path.replace("pytorch_model.bin", "config.json")
                config = BertConfig.from_json_file(args.config_file)
                config.wrapper_model_type = args.wrapper_model_type

                model = UniterForVisualQuestionAnswering.from_pretrained(
                    model_config, checkpoint, img_dim=IMG_DIM,
                    num_answer=config.num_labels,
                    vision_model=vision_model, tokenizer=tokenizer,
                    multimodal_text_part=args.multimodal_text_part,
                    multimodal_img_part=args.multimodal_img_part,
                    additional_config=config)

            elif args.multimodal_model_type == "vilbert":
                args.config_file = os.path.join(ckpt_path,
                                                "config.json")
                config = VILBertConfig.from_json_file(args.config_file)
                if "roberta" in ckpt_path:
                    config.model = "roberta"
                    config.model_type = "roberta"
                if "v_target_size" not in config.__dict__:
                    config.v_target_size = 1601
                config.task_specific_tokens = False
                config.dynamic_attention = False
                try:
                    config.v_feature_size = vision_model.fc.in_features
                except:
                    config.v_feature_size = args.vision_feature_dim
                config.freeze_vision_model = args.freeze_vision_model
                config.multimodal_text_part = args.multimodal_text_part
                config.multimodal_img_part = args.multimodal_img_part
                config.fusion_method = args.multimodal_fusion_method
                config.vilbert_use_3way_logits = args.vilbert_use_3way_logits
                bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
                cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
                if cls_id == bos_id:
                    config.cls_id = cls_id
                else:
                    config.cls_id = bos_id
                config.hierarchical_version = args.hierarchical_version
                config.max_story_length = args.max_story_length
                if args.vilbert_without_coattention:
                    config.with_coattention = False
                config.simple_img_classifier = False
                config.vilbert_paired_coattention = args.vilbert_paired_coattention
                if "include_num_img_regional_features" not in config.__dict__:
                    config.include_num_img_regional_features = \
                        args.include_num_img_regional_features
                args.include_num_img_regional_features =  \
                    config.include_num_img_regional_features
                if "heatmap_decode_beam_size" not in config.__dict__:
                    config.heatmap_decode_beam_size = args.heatmap_decode_beam_size
                else:
                    args.heatmap_decode_beam_size = config.heatmap_decode_beam_size
                config.wrapper_model_type = args.wrapper_model_type

                default_gpu = False
                if args.local_rank in [-1, 0]:
                    default_gpu = True
                args.pretrained_bin = os.path.join(ckpt_path,
                                                   "pytorch_model.bin")
                model = VILBertForVLTasks.from_pretrained(
                    args.pretrained_bin,
                    config=config,
                    num_labels=config.num_labels,
                    default_gpu=default_gpu,
                    output_loading_info=False,
                    vision_model=vision_model,
                )

            elif args.multimodal_model_type == "clip":
                sys.path.insert(0, "models/CLIP/src")
                sys.path.insert(0, "models/CLIP/clip")
                from models.CLIP.src.lxrt.modeling import LXRTModel
                
                cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
                args.cls_id = cls_id
                sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
                args.sep_id = sep_id

                model = LXRTModel.from_pretrained(
                    args.model_name_or_path_1,
                    multimodal_text_part=args.multimodal_text_part,
                    multimodal_img_part=args.multimodal_img_part,
                    cls_id=args.cls_id,
                    sep_id=args.sep_id,
                    max_story_length=args.max_story_length,
                    clip_model_name=args.clip_model_name,
                    num_labels=2,
                )
                # print(model)
                # raise

            else:
                raise NotImplementedError("Multimodal model type: "
                    "{} not done yet!".format(args.multimodal_model_type))
            new_models.append(model)
    else:
        raise NotImplementedError("Does not handle yet!")

    return new_models


def get_models(args):
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()
     
    if "pure_decode" in args.sort_method:
        config_1 = EncoderDecoderConfig.from_pretrained(
            args.config_name_1 if args.config_name_1 \
                               else args.model_name_or_path_1,
        )
        tokenizer_1 = AutoTokenizer.from_pretrained(
            args.tokenizer_name_1 if args.tokenizer_name_1 \
                                  else args.model_name_or_path_1,
            do_lower_case=args.do_lower_case,
        )
        model_1 = EncoderDecoderModel.from_pretrained(
            args.model_name_or_path_1,
            from_tf=bool(".ckpt" in args.model_name_or_path_1),
            config=config_1,
        )
        model_1.to(args.device)
        return model_1, tokenizer_1, config_1

    elif "head_and_topological" in args.sort_method:
        config_1 = AutoConfig.from_pretrained(
            args.config_name_1 if args.config_name_1 \
                               else args.model_name_or_path_1,
            num_labels=2,
        )
        config_2 = AutoConfig.from_pretrained(
            args.config_name_2 if args.config_name_2 \
                               else args.model_name_or_path_2,
            num_labels=args.max_story_length,
        )
        tokenizer_1 = AutoTokenizer.from_pretrained(
            args.tokenizer_name_1 if args.tokenizer_name_1 \
                                  else args.model_name_or_path_1,
            do_lower_case=args.do_lower_case,
        )
        if args.multimodal or args.use_multimodal_model:
            ckpt_paths = [args.model_name_or_path_1, args.model_name_or_path_2]
            configs = [config_1, config_2]
            models = get_multimodal_models(args, tokenizer_1,
                                           ckpt_paths=ckpt_paths,
                                           configs=configs,
                                           language_models=None)
            model_1, model_2 = models[0], models[1]
        else:
            model_1 = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path_1,
                from_tf=bool(".ckpt" in args.model_name_or_path_1),
                config=config_1,
            )
            model_2 = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path_2,
                from_tf=bool(".ckpt" in args.model_name_or_path_2),
                config=config_2,
            )
        model_1.to(args.device)
        model_2.to(args.device)
            
        return model_1, model_2, tokenizer_1, config_1, config_2

    elif ("topological" in args.sort_method
          or "pure_classification" == args.sort_method
          or "heat_map" == args.sort_method):
        config_1 = AutoConfig.from_pretrained(
            args.config_name_1 if args.config_name_1 \
                               else args.model_name_or_path_1,
            # num_labels=2,
        )
        tokenizer_1 = AutoTokenizer.from_pretrained(
            args.tokenizer_name_1 if args.tokenizer_name_1 \
                                  else args.model_name_or_path_1,
            do_lower_case=args.do_lower_case,
        )
        if args.multimodal or args.use_multimodal_model:
            ckpt_paths = [args.model_name_or_path_1]
            configs = [config_1]
            models = get_multimodal_models(args, tokenizer_1,
                                           ckpt_paths=ckpt_paths,
                                           configs=configs,
                                           language_models=None)
            model_1 = models[0]
        else:
            model_1 = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path_1,
                from_tf=bool(".ckpt" in args.model_name_or_path_1),
                config=config_1,
            )
        model_1.to(args.device)
        return model_1, tokenizer_1, config_1

    elif "head_and_pairwise" in args.sort_method:
        config_1 = AutoConfig.from_pretrained(
            args.config_name_1 if args.config_name_1 \
                               else args.model_name_or_path_1,
            num_labels=2,
        )
        config_2 = AutoConfig.from_pretrained(
            args.config_name_2 if args.config_name_2 \
                               else args.model_name_or_path_2,
            num_labels=args.max_story_length,
        )
        tokenizer_1 = AutoTokenizer.from_pretrained(
            args.tokenizer_name_1 if args.tokenizer_name_1 \
                                  else args.model_name_or_path_1,
            do_lower_case=args.do_lower_case,
        )
        if args.multimodal or args.use_multimodal_model:
            ckpt_paths = [args.model_name_or_path_1, args.model_name_or_path_2]
            configs = [config_1, config_2]
            models = get_multimodal_models(args, tokenizer_1,
                                           ckpt_paths=ckpt_paths,
                                           configs=configs,
                                           language_models=None)
            model_1, model_2 = models[0], models[1]
        else:
            model_1 = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path_1,
                from_tf=bool(".ckpt" in args.model_name_or_path_1),
                config=config_1,
            )
            model_2 = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path_2,
                from_tf=bool(".ckpt" in args.model_name_or_path_2),
                config=config_2,
            )
        model_1.to(args.device)
        model_2.to(args.device)

        if args.sort_method == "head_and_pairwise_abductive":
            if args.abd_pred_method == "binary":
                abd_num_labels = 2
            else:
                raise NotImplementedError("Pred method {} not implemented"
                                          " yet!".format(args.abd_pred_method))
            config_3 = AutoConfig.from_pretrained(
                args.config_name_3 if args.config_name_3 \
                                   else args.model_name_or_path_3,
                num_labels=abd_num_labels,
            )
            if args.multimodal or args.use_multimodal_model:
                ckpt_paths = [args.model_name_or_path_3]
                configs = [config_3]
                models = get_multimodal_models(args, ckpt_paths=ckpt_paths,
                                               configs=configs,
                                               language_models=None)
                model_3 = models[0]
            else:
                model_3 = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name_or_path_3,
                    from_tf=bool(".ckpt" in args.model_name_or_path_3),
                    config=config_3,
                )
            model_3.to(args.device)
            return (model_1, model_2, model_3, tokenizer_1,
                    config_1, config_2, config_3)
            
        return model_1, model_2, tokenizer_1, config_1, config_2
    else:
        raise NotImplementedError("Sort method {} not "
                                  "implemented yet.".format(args.sort_method))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier() 

    return None, None, None


def topological_inference(args, model, seqs, tokenizer, images=None,
                          batch=None):
    batch_seqs = debatch_stories(seqs)
    len_seq = len(seqs)
    
    adj_mat = [[0 for i in range(len_seq)] for j in range(len_seq)]

    preds = []
    cnt = 0
    loss = 0
    for seq_idx in range(len(batch_seqs)):
        curr_seq = batch_seqs[seq_idx]
        graph = Graph(len_seq)
        for i in range(len_seq):
            for j in range(len_seq):
                if i < j:
                    text_a = curr_seq[i]
                    text_b = curr_seq[j]

                    if args.multimodal or args.use_multimodal_model or not args.use_cached:
                        curr_seq_new = [text_a, text_b]
                        batch_encoding = tokenizer(
                            curr_seq_new,
                            max_length=args.per_seq_max_length,
                            padding="max_length",
                            truncation=True,
                        )

                        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
                        padded_input_ids = np.ones(args.max_seq_length, dtype=int)
                        padded_token_type_ids = np.zeros(args.max_seq_length, dtype=int)
                        cat_input_ids = np.asarray([], dtype=int)
                        cat_token_type_ids = np.asarray([], dtype=int)

                        for k in range(len(seqs_input_ids)):
                            seq_input_ids = seqs_input_ids[k]
                            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=1]
                            cat_input_ids = np.concatenate((cat_input_ids,
                                                            seq_input_ids_unpad), axis=0)
                            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * k
                            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                                 token_type_ids), axis=0)
                        max_length = min(args.max_seq_length, len(cat_input_ids))
                        padded_input_ids[:max_length] = cat_input_ids[:max_length]
                        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
                        input_ids = torch.Tensor(padded_input_ids).long()
                        attention_mask = (input_ids != 1).long()
                        token_type_ids = torch.Tensor(padded_token_type_ids).long()

                        input_ids = input_ids.unsqueeze(0)
                        attention_mask = attention_mask.unsqueeze(0)
                        token_type_ids = token_type_ids.unsqueeze(0)
                        batch_encoding = {
                            "input_ids": input_ids.to(args.device),
                            "attention_mask": attention_mask.to(args.device),
                            "token_type_ids": token_type_ids.to(args.device),
                        }

                        if args.multimodal and not args.multimodal_text_part:
                            images_curr = [
                                images[seq_idx, i, :, :, :],
                                images[seq_idx, j, :, :, :],
                            ]
                            images_curr = torch.stack(images_curr, dim=0)
                            images_curr = images_curr.unsqueeze(0)
                            batch_encoding["images"] = images_curr
                            if args.include_num_img_regional_features is not None:
                                batch_encoding["img_regional_features"] = batch[-1]

                    else:
                        batch_encoding = tokenizer(
                            [(text_a, text_b)],
                            max_length=args.max_seq_length,
                            padding="max_length",
                            truncation=True,
                            return_token_type_ids=True,
                            return_tensors="pt",
                        )

                    batch_encoding = {k: batch_encoding[k].to(args.device)
                                      for k in batch_encoding}
                    if not args.replace_token_type_embeddings:
                        batch_encoding["token_type_ids"] = None

                    with torch.no_grad():
                        if args.multimodal or args.use_multimodal_model:
                            if args.multimodal_model_type == "clip":
                                batch_encoding["visual_feats"] = batch_encoding["images"]
                                del batch_encoding["images"]
                                logits = model(**batch_encoding)
                            else:
                                logits = model(batch_encoding)
                        else:
                            logits = model(**batch_encoding)
                        binary_preds = logits[0].detach().cpu().numpy()[0]
                        pred_label = np.argmax(binary_preds)
                    if pred_label == 1:  # Ordered.
                        graph.addEdge(i, j)
                    else:  # Unordered.
                        graph.addEdge(j, i)
                    cnt += 1
        top_sort = graph.topologicalSort()
        preds.append(top_sort)
    loss /= cnt
    return preds, loss


def head_and_topological_inference(args, head_model, pairwise_model,
                                   seqs, tokenizer, abductive_model=None,
                                   images=None, batch=None):

    head_idx = head_and_sequential_inference(args, head_model, pairwise_model,
                                             seqs, tokenizer,
                                             abductive_model=None,
                                             return_head_idx=True,
                                             images=images)

    batch_seqs = debatch_stories(seqs)
    len_seq = len(seqs)
    
    adj_mat = [[0 for i in range(len_seq)] for j in range(len_seq)]

    preds = []
    cnt = 0
    loss = 0
    for seq_idx in range(len(batch_seqs)):
        curr_seq = batch_seqs[seq_idx]
        graph = Graph(len_seq)
        for i in range(len_seq):
            for j in range(len_seq):
                if i < j:
                    text_a = curr_seq[i]
                    text_b = curr_seq[j]

                    if args.multimodal or args.use_multimodal_model or not args.use_cached:
                        curr_seq_new = [text_a, text_b]
                        batch_encoding = tokenizer(
                            curr_seq_new,
                            max_length=args.per_seq_max_length,
                            padding="max_length",
                            truncation=True,
                        )

                        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
                        padded_input_ids = np.ones(args.max_seq_length, dtype=int)
                        padded_token_type_ids = np.zeros(args.max_seq_length, dtype=int)
                        cat_input_ids = np.asarray([], dtype=int)
                        cat_token_type_ids = np.asarray([], dtype=int)

                        for k in range(len(seqs_input_ids)):
                            seq_input_ids = seqs_input_ids[k]
                            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=1]
                            cat_input_ids = np.concatenate((cat_input_ids,
                                                            seq_input_ids_unpad), axis=0)
                            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * k
                            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                                 token_type_ids), axis=0)
                        max_length = min(args.max_seq_length, len(cat_input_ids))
                        padded_input_ids[:max_length] = cat_input_ids[:max_length]
                        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
                        input_ids = torch.Tensor(padded_input_ids).long()
                        attention_mask = (input_ids != 1).long()
                        token_type_ids = torch.Tensor(padded_token_type_ids).long()

                        input_ids = input_ids.unsqueeze(0)
                        attention_mask = attention_mask.unsqueeze(0)
                        token_type_ids = token_type_ids.unsqueeze(0)
                        batch_encoding = {
                            "input_ids": input_ids.to(args.device),
                            "attention_mask": attention_mask.to(args.device),
                            "token_type_ids": token_type_ids.to(args.device),
                        }
                        if args.multimodal and not args.multimodal_text_part:
                            images_curr = [
                                images[seq_idx, i, :, :, :],
                                images[seq_idx, j, :, :, :],
                            ]
                            images_curr = torch.stack(images_curr, dim=0)
                            images_curr = images_curr.unsqueeze(0)
                            batch_encoding["images"] = images_curr

                    else:
                        batch_encoding = tokenizer(
                            [(text_a, text_b)],
                            max_length=args.max_seq_length,
                            padding="max_length",
                            truncation=True,
                            return_token_type_ids=True,
                            return_tensors="pt",
                        )

                    batch_encoding = {k: batch_encoding[k].to(args.device)
                                      for k in batch_encoding}
                    if not args.replace_token_type_embeddings:
                        batch_encoding["token_type_ids"] = None

                    with torch.no_grad():
                        if args.multimodal or args.use_multimodal_model:
                            logits = pairwise_model(batch_encoding)
                        else:
                            logits = pairwise_model(**batch_encoding)
                        binary_preds = logits[0].detach().cpu().numpy()[0]
                        pred_label = np.argmax(binary_preds)
                    if pred_label == 1:  # Ordered.
                        graph.addEdge(i, j)
                    else:  # Unordered.
                        graph.addEdge(j, i)
                    cnt += 1
        top_sort = graph.topologicalSort(assert_head=head_idx)
        preds.append(top_sort)
    loss /= cnt
    return preds, loss


def head_and_sequential_inference(args, head_model, pairwise_model,
                                  seqs, tokenizer, abductive_model=None,
                                  return_head_idx=False, images=None,
                                  batch=None):
    batch_seqs = debatch_stories(seqs)
    preds = []
    cnt = 0
    loss = 0
    for seq_idx in range(len(batch_seqs)):
        curr_seq = batch_seqs[seq_idx]
       
        # Predicts the head first.
        batch_encoding = tokenizer(
            curr_seq,
            max_length=args.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.ones(args.max_seq_length, dtype=int)
        padded_token_type_ids = np.zeros(args.max_seq_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=1]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(args.max_seq_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        token_type_ids = token_type_ids.unsqueeze(0)
        batch_encoding = {
            "input_ids": input_ids.to(args.device),
            "attention_mask": attention_mask.to(args.device),
            "token_type_ids": token_type_ids.to(args.device),
        }
        if args.multimodal and not args.multimodal_text_part:
            batch_encoding["images"] = images.to(args.device)
            
        if not args.replace_token_type_embeddings:
            batch_encoding["token_type_ids"] = None
        with torch.no_grad():
            if args.multimodal or args.use_multimodal_model:
                logits = head_model(batch_encoding)
            else:
                logits = head_model(**batch_encoding)
            head_preds = logits[0].detach().cpu().numpy()[0]
            head_preds_label = np.argmax(head_preds)

        if return_head_idx:
            return head_preds_label

        curr_pred = [head_preds_label]
        story_seq_idx = list(range(min(args.max_story_length, len(curr_seq))))
        
        # Sequentially predict nexts.
        while len(story_seq_idx) > 1:  # When only 1 element left, do nothing.
            story_seq_idx.remove(curr_pred[-1])
            curr_prev_idx = curr_pred[-1]
            next_seq_idx = select_next(args, pairwise_model, tokenizer,
                                       story_seq_idx, curr_pred, curr_seq,
                                       abductive_model=abductive_model,
                                       images=images)
            curr_pred.append(next_seq_idx)
        preds.append(curr_pred)
        cnt += 1

    loss /= cnt
    return preds, loss


def select_next(args, pairwise_model, tokenizer, seq_idx_left,
                curr_pred, seq, abductive_model=None, images=None):
    scores = []
    prev_idx = curr_pred[-1]
    for idx in seq_idx_left:
        sent_cand = seq[idx]
        curr_sent = seq[prev_idx]

        if args.multimodal or args.use_multimodal_model:
            curr_seq_new = [curr_sent, sent_cand]
            batch_encoding = tokenizer(
                curr_seq_new,
                max_length=args.per_seq_max_length,
                padding="max_length",
                truncation=True,
            )

            seqs_input_ids = np.asarray(batch_encoding["input_ids"])
            padded_input_ids = np.ones(args.max_seq_length, dtype=int)
            padded_token_type_ids = np.zeros(args.max_seq_length, dtype=int)
            cat_input_ids = np.asarray([], dtype=int)
            cat_token_type_ids = np.asarray([], dtype=int)

            for i in range(len(seqs_input_ids)):
                seq_input_ids = seqs_input_ids[i]
                seq_input_ids_unpad = seq_input_ids[seq_input_ids!=1]
                cat_input_ids = np.concatenate((cat_input_ids,
                                                seq_input_ids_unpad), axis=0)
                token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
                cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                     token_type_ids), axis=0)
            max_length = min(args.max_seq_length, len(cat_input_ids))
            padded_input_ids[:max_length] = cat_input_ids[:max_length]
            padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
            input_ids = torch.Tensor(padded_input_ids).long()
            attention_mask = (input_ids != 1).long()
            token_type_ids = torch.Tensor(padded_token_type_ids).long()

            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            batch_encoding = {
                "input_ids": input_ids.to(args.device),
                "attention_mask": attention_mask.to(args.device),
                "token_type_ids": token_type_ids.to(args.device),
            }
            if not args.multimodal_text_part:
                images_curr = [
                    images[:, prev_idx, :, :, :],
                    images[:, idx, :, :, :],
                ]
                images_curr = torch.stack(images_curr, dim=1)
                batch_encoding["images"] = images_curr
            pw_score = pairwise_score(args, pairwise_model, tokenizer,
                                      curr_sent, sent_cand,
                                      batch_encoding=batch_encoding,
                                      images=images)
        else:
            pw_score = pairwise_score(args, pairwise_model, tokenizer,
                                      curr_sent, sent_cand, images=images)
        if abductive_model is not None and len(curr_pred) >= 2:
            abd_score = abductive_score(args, abductive_model, tokenizer,
                                        curr_pred, idx, seq, images=images)
            score = pw_score + abd_score * 0.1
        else:
            score = pw_score
        scores.append(score)
    scores = np.asarray(scores)
    next_seq_idx = int(np.argmax(scores))
    next_seq_idx = seq_idx_left[next_seq_idx]
    return next_seq_idx

def abductive_score(args, abductive_model, tokenizer, curr_pred, idx,
                    seq, images=None):
    if images is not None:
        raise NotImplementedError("Multimodal not done yet!")

    text_h1 = seq[curr_pred[-2]]
    text_h2 = seq[curr_pred[-1]]
    text_h3 = seq[idx]

    story_seq = [text_h1, text_h2, text_h3]

    batch_encoding = tokenizer(
        story_seq,
        max_length=args.per_seq_max_length,
        padding="max_length",
        truncation=True,
    )

    seqs_input_ids = np.asarray(batch_encoding["input_ids"])
    padded_input_ids = np.ones(args.max_seq_length, dtype=int)
    padded_token_type_ids = np.zeros(args.max_seq_length, dtype=int)
    cat_input_ids = np.asarray([], dtype=int)
    cat_token_type_ids = np.asarray([], dtype=int)

    for i in range(len(seqs_input_ids)):
        seq_input_ids = seqs_input_ids[i]
        seq_input_ids_unpad = seq_input_ids[seq_input_ids!=1]
        cat_input_ids = np.concatenate((cat_input_ids,
                                        seq_input_ids_unpad), axis=0)
        token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
        cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                             token_type_ids), axis=0)
    max_length = min(args.max_seq_length, len(cat_input_ids))
    padded_input_ids[:max_length] = cat_input_ids[:max_length]
    padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
    input_ids = torch.Tensor(padded_input_ids).long()
    attention_mask = (input_ids != 1).long()
    token_type_ids = torch.Tensor(padded_token_type_ids).long()

    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    token_type_ids = token_type_ids.unsqueeze(0)
    batch_encoding = {
        "input_ids": input_ids.to(args.device),
        "attention_mask": attention_mask.to(args.device),
        "token_type_ids": token_type_ids.to(args.device),
    }
    if not args.replace_token_type_embeddings:
        batch_encoding["token_type_ids"] = None

    if args.abd_pred_method == "binary":
        with torch.no_grad():
            logits = abductive_model(**batch_encoding)
            binary_preds = logits[0].detach().cpu().numpy()[0]
            # Score is determined by the logits at position 1.
            score = binary_preds[1]
    elif args.abd_pred_method == "contrastive":
        raise NotImplementedError("Prediction method {} not"
                                  " done yet!".format(args.abd_pred_method))
    return score


def pairwise_score(args, pairwise_model, tokenizer, curr_sent, next_sent,
                   batch_encoding=None, images=None):

    if batch_encoding is None:
        batch_encoding = tokenizer(
            [(curr_sent, next_sent)],
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
    batch_encoding = {k: batch_encoding[k].to(args.device)
                      for k in batch_encoding}
    if not args.replace_token_type_embeddings:
        batch_encoding["token_type_ids"] = None
    with torch.no_grad():
        if args.multimodal or args.use_multimodal_model:
            logits = pairwise_model(batch_encoding)
        else:
            logits = pairwise_model(**batch_encoding)
        binary_preds = logits[0].detach().cpu().numpy()[0]
        # Score is determined by the logits at position 1.
        score = binary_preds[1]
    return score


def pure_class_inference(args, pure_class_model, seqs, tokenizer, id2label, 
                         images=None, batch=None):
    if images is not None:
        raise NotImplementedError("Multimodal not done yet!")

    batch_seqs = debatch_stories(seqs)
    preds = []
    cnt = 0
    loss = 0
    for seq_idx in range(len(batch_seqs)):
        curr_seq = batch_seqs[seq_idx]
       
        # Predicts the head first.
        batch_encoding = tokenizer(
            curr_seq,
            max_length=args.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.ones(args.max_seq_length, dtype=int)
        padded_token_type_ids = np.zeros(args.max_seq_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=1]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(args.max_seq_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        token_type_ids = token_type_ids.unsqueeze(0)
        batch_encoding = {
            "input_ids": input_ids.to(args.device),
            "attention_mask": attention_mask.to(args.device),
            "token_type_ids": token_type_ids.to(args.device),
        }
        if not args.replace_token_type_embeddings:
            batch_encoding["token_type_ids"] = None
        with torch.no_grad():
            logits = pure_class_model(**batch_encoding)
            curr_preds = logits[0].detach().cpu().numpy()[0]
            preds_label = int(np.argmax(curr_preds))

        curr_pred = id2label[preds_label]
        preds.append(curr_pred)
        cnt += 1

    loss /= cnt
    return preds, loss


def pure_decode_inference(args, encoder_decoder_model, seqs, tokenizer,
                          images=None, batch=None):
    if images is not None:
        raise NotImplementedError("Multimodal not done yet!")

    batch_seqs = debatch_stories(seqs)
    preds = []
    cnt = 0
    loss = 0
    for seq_idx in range(len(batch_seqs)):
        curr_seq = batch_seqs[seq_idx]
       
        # Predicts the head first.
        batch_encoding = tokenizer(
            curr_seq,
            max_length=args.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.ones(args.max_seq_length, dtype=int)
        padded_token_type_ids = np.zeros(args.max_seq_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=1]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(args.max_seq_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        token_type_ids = token_type_ids.unsqueeze(0)
        batch_encoding = {
            "input_ids": input_ids.to(args.device),
            "attention_mask": attention_mask.to(args.device),
            "token_type_ids": token_type_ids.to(args.device),
        }
        if not args.replace_token_type_embeddings:
            batch_encoding["token_type_ids"] = None
        with torch.no_grad():
            # https://huggingface.co/blog/how-to-generate
            outputs = encoder_decoder_model.generate(
                batch_encoding["input_ids"],
                max_length=len(seqs),
                num_beams=5,
                no_repeat_ngram_size=2,
                decoder_start_token_id=encoder_decoder_model.config.decoder.pad_token_id,
            )

        curr_pred = list(outputs.cpu().numpy()[0])
        preds.append(curr_pred)
        cnt += 1

    loss /= cnt
    return preds, loss


def heat_map_inference(args, hl_model, seqs, tokenizer, labels,
                       images=None, batch=None):
    batch_seqs = debatch_stories(seqs)
    preds = []
    cnt = 0
    loss = 0
    for seq_idx in range(len(batch_seqs)):
        curr_seq = batch_seqs[seq_idx]
       
        # Predicts the head first.
        batch_encoding = tokenizer(
            curr_seq,
            max_length=args.per_seq_max_length,
            padding="max_length",
            truncation=True,
        )

        seqs_input_ids = np.asarray(batch_encoding["input_ids"])
        padded_input_ids = np.ones(args.max_seq_length, dtype=int)
        padded_token_type_ids = np.zeros(args.max_seq_length, dtype=int)
        cat_input_ids = np.asarray([], dtype=int)
        cat_token_type_ids = np.asarray([], dtype=int)

        for i in range(len(seqs_input_ids)):
            seq_input_ids = seqs_input_ids[i]
            seq_input_ids_unpad = seq_input_ids[seq_input_ids!=1]
            cat_input_ids = np.concatenate((cat_input_ids,
                                            seq_input_ids_unpad), axis=0)
            token_type_ids = np.ones(len(seq_input_ids_unpad), dtype=int) * i
            cat_token_type_ids = np.concatenate((cat_token_type_ids,
                                                 token_type_ids), axis=0)
        max_length = min(args.max_seq_length, len(cat_input_ids))
        padded_input_ids[:max_length] = cat_input_ids[:max_length]
        padded_token_type_ids[:max_length] = cat_token_type_ids[:max_length]
        input_ids = torch.Tensor(padded_input_ids).long()
        attention_mask = (input_ids != 1).long()
        token_type_ids = torch.Tensor(padded_token_type_ids).long()

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        token_type_ids = token_type_ids.unsqueeze(0)
        batch_encoding = {
            "input_ids": input_ids.to(args.device),
            "attention_mask": attention_mask.to(args.device),
            "token_type_ids": token_type_ids.to(args.device),
        }
        if args.multimodal and not args.multimodal_text_part:
            batch_encoding["images"] = images[seq_idx].to(args.device).unsqueeze(0)

        batch_encoding["labels"] = labels
            
        if not args.replace_token_type_embeddings:
            batch_encoding["token_type_ids"] = None
        with torch.no_grad():
            if args.multimodal or args.use_multimodal_model:
                logits = hl_model(batch_encoding)
            else:
                logits = hl_model(**batch_encoding)
            heat_map_preds = logits[1].detach().cpu().numpy()[0]
            heatmap_labels = logits[2].detach().cpu().numpy()[0]
            # print(heat_map_preds)
            # print(labels)
            curr_pred = heatmap2order(args, heat_map_preds, soft=False)
            # print(curr_pred);raise

        preds.append(curr_pred)
        cnt += 1

    loss /= cnt
    return preds, loss


def debatch_stories(seqs):
    len_seq = len(seqs)
    batch_size = len(seqs[0])
    batch_seqs = []
    for i in range(batch_size):
        batch_seq = []
        for j in range(len_seq):
            batch_seq.append(seqs[j][i])
        batch_seqs.append(batch_seq)
    return batch_seqs


def model_wise_evaluate(args, models, batch, tokenizer, id2label=None):
    if not args.multimodal:
        stories, labels = batch
        images = None
    elif args.multimodal and args.multimodal_text_part:
        stories, labels, _ = batch
        images = None
    else:
        if args.include_num_img_regional_features is not None:
            stories, labels, images, img_regional_features = batch
            if args.sort_method not in ["topological"]:
                raise NotImplementedError("Not done yet!")
        else:
            stories, labels, images = batch
    if args.sort_method == "head_and_topological":
        pairwise_model, head_model = models[0], models[1]
        preds, loss = head_and_topological_inference(args, head_model,
                                                     pairwise_model,
                                                     stories, tokenizer,
                                                     images=images,
                                                     batch=batch)
    elif "topological" in args.sort_method:
        pairwise_model = models[0]
        preds, loss = topological_inference(args, pairwise_model,
                                            stories, tokenizer,
                                            images=images,
                                            batch=batch)
    elif args.sort_method == "head_and_pairwise":
        pairwise_model, head_model = models[0], models[1]
        preds, loss = head_and_sequential_inference(args, head_model,
                                                    pairwise_model,
                                                    stories, tokenizer,
                                                    images=images,
                                                    batch=batch)
    elif args.sort_method == "head_and_pairwise_abductive":
        pairwise_model, head_model = models[0], models[1]
        abductive_model = models[2]
        preds, loss = head_and_sequential_inference(args,
            head_model, pairwise_model,
            stories, tokenizer,
            abductive_model=abductive_model,
            images=images,
            batch=batch)
    elif args.sort_method == "pure_classification":
        pure_class_model = models[0]
        preds, loss = pure_class_inference(args, pure_class_model,
                                           stories, tokenizer, id2label,
                                           images=images,
                                           batch=batch)
    elif args.sort_method == "pure_decode":
        encoder_decoder_model = models[0]
        preds, loss = pure_decode_inference(args, encoder_decoder_model,
                                            stories, tokenizer,
                                            images=images,
                                            batch=batch)
    elif args.sort_method == "heat_map":
        hl_model = models[0]
        preds, loss = heat_map_inference(args, hl_model,
                                         stories, tokenizer, labels,
                                         images=images,
                                         batch=batch)
    else:
        raise NotImplementedError("Sort method {} not "
                                  "implemented yet.".format(args.sort_method))

    return preds, labels, loss


def evaluate(args, models, tokenizer, prefix=""):
    eval_outputs_dir = args.output_dir

    results = {}

    for eval_task in args.task_names:
        task_proc_class = processors[eval_task]
        if task_proc_class is None:
            logger.error("No processor for task: {}".format(eval_task))
            continue

        # TODO: Caption transformations
        if args.caption_transformations is not None:
            assert type(args.caption_transformations) == list
            from trainers.caption_utils import CaptionTransformations
            caption_transforms = CaptionTransformations(args, eval_task)

            if "recipeqa" not in eval_task and "wikihow" not in eval_task:
                raise NotImplementedError("Only deal with RecipeQA and WikiHow now!")
        else:
            caption_transforms = None

        split_version_text = args.data_split.split("-")[-1]
        if len(args.data_split.split("-")) == 1:
            split_version_text = None
        else:
            split_version_text = split_version_text.strip()
        data_split_curr = args.data_split.split("-")[0]

        task_proc = processors[eval_task](
            max_story_length=args.max_story_length,
            caption_transforms=caption_transforms,
            version_text=split_version_text)

        if data_split_curr == "val" or data_split_curr == "dev":
            eval_examples = task_proc.get_dev_examples()
        elif data_split_curr == "train":
            eval_examples = task_proc.get_train_examples()
        else:
            eval_examples = task_proc.get_test_examples()

        eval_dataset = SortDatasetV1(eval_examples, tokenizer,
                                     args=args,
                                     max_length=args.max_seq_length,
                                     max_story_length=args.max_story_length,
                                     multimodal=args.multimodal,
                                     img_transform_func=args.img_transform_func,
                                     num_img_regional_features=args.include_num_img_regional_features,
                                     seed=args.seed)

        if "pure_classification" == args.sort_method:
            dummy_dataset = PureClassDataset(eval_examples, tokenizer,
                                             max_length=args.max_seq_length,
                                             max_story_length=args.max_story_length,
                                             seed=args.seed)
            id2label = dummy_dataset.id2label
        else:
            id2label = None

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(models[0], torch.nn.DataParallel):
            for i in range(len(models)):
                models[i] = torch.nn.DataParallel(models[i])

        # Eval!
        logger.info("***** Running evaluation on {} of split: {} *****".format(
            eval_task, args.data_split))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        nb_eval_steps_cnt = 0
        preds = []
        labels = []
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if args.eval_on_every_iter is not None:
                if nb_eval_steps_cnt % args.eval_on_every_iter != 0:
                    nb_eval_steps_cnt += 1
                    continue
            b_preds, b_labels, b_eval_loss = model_wise_evaluate(args, models,
                                                                 batch,
                                                                 tokenizer,
                                                                 id2label=id2label)
            preds += b_preds
            labels += b_labels
            nb_eval_steps += 1
            nb_eval_steps_cnt += 1

            if args.max_eval_steps > 0 and nb_eval_steps >= args.max_eval_steps:
                logging.info("Early stopping"
                    " evaluation at step: {}".format(args.max_eval_steps))
                break

        eval_loss = eval_loss / nb_eval_steps

        res = {}
        for metrics in args.metrics:
            acc = compute_metrics(args, metrics, preds, labels)
            res[metrics] = acc
            acc = round(acc, 6)
            result = {
                eval_task+"_accuracy on split: {} of {}".format(
                    args.data_split, metrics): acc,
            }
            results.update(result)

        headers = "& PM    & EM    & Lseq & Lstr & tau  & Dist."
        # content = "& 00.00    & 00.00    & 0.00     & 0.00     & 0.00     & 0.00 "
        content = "& {:03.2f} & {:03.2f} & {:03.2f} & {:03.2f} & {:03.2f} & {:03.2f}".format(
            res["partial_match"] * 100,
            res["exact_match"] * 100,
            res["lcs"],
            res["lcs_substr"],
            res["tau"],
            res["distance_based"],
        )
        logger.info("***** Paper Results *****")
        logger.info(" {}".format(headers))
        logger.info(" {}".format(content))


    output_eval_file = os.path.join(eval_outputs_dir, "downstream_eval_results_split_{}.txt".format(args.data_split))

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval Results *****")

        if args.model_name_or_path_1 is not None:
            logger.info("  Model 1 from: {}".format(args.model_name_or_path_1))
            writer.write("Model 1 from: {}".format(args.model_name_or_path_1))

        if args.model_name_or_path_2 is not None:
            logger.info("  Model 2 from: {}".format(args.model_name_or_path_2))
            writer.write("Model 2 from: {}".format(args.model_name_or_path_2))

        if args.model_name_or_path_3 is not None:
            logger.info("  Model 3 from: {}".format(args.model_name_or_path_3))
            writer.write("Model 3 from: {}".format(args.model_name_or_path_3))

        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

    logger.info("Results saved at: {}".format(output_eval_file))

    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--model_name_or_path_1",
        default=None,
        type=str,
        required=True,
        help=("Path to pretrained model or model identifier from "
              "huggingface.co/models"),
    )
    parser.add_argument(
        "--model_name_or_path_2",
        default=None,
        type=str,
        required=False,
        help=("Path to pretrained model or model identifier from "
              "huggingface.co/models"),
    )
    parser.add_argument(
        "--model_name_or_path_3",
        default=None,
        type=str,
        required=False,
        help=("Path to pretrained model or model identifier from "
              "huggingface.co/models"),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help=("The output directory where the results will be stored."),
    )
    parser.add_argument(
        "--output_root",
        default=None,
        type=str,
        help=("The prefix root directory where the model predictions and "
              "checkpoints will be written."),
    )
    parser.add_argument(
        "--max_eval_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of evaluating steps to perform.",
    )
    parser.add_argument(
        "--max_story_length",
        default=5,
        type=int,
        required=False,
        help=("The maximum length of the story sequence."),
    )
    parser.add_argument(
        "--per_seq_max_length",
        default=64,
        type=int,
        required=False,
        help=("The maximum length of EACH of the story sequence."),
    )
    parser.add_argument(
        "--sort_method",
        default=None,
        type=str,
        choices=[
            "topological", "topological_sort", "head_and_topological",
            "head_and_pairwise", "head_and_pairwise_abductive",
            "pure_classification", "pure_decode",
            "heat_map",
        ],
        required=True,
        help=("The method for predicting the sorted sequence."),
    )
    parser.add_argument(
        "--abd_pred_method",
        default="binary",
        type=str,
        required=False,
        choices=["binary", "contrastive"],
        help=("The prediction method, see datasets/roc.py"),
    )
    parser.add_argument(
        "--data_splits",
        default="val",
        # choices=["train", "val", "test"],
        type=str,
        required=False,
        nargs="+",
        help=("The phase of evaluation data loading."),
    )
    parser.add_argument(
        "--task_names",
        default=None,
        nargs="+",
        type=str,
        required=True,
        help=("The list of task names."),
    )
    parser.add_argument(
        "--metrics",
        default="partial_match",
        nargs="+",
        choices=["partial_match", "exact_match", "distance_based",
                 "longest_common_subsequence", "lcs", "lcs_substr", "tau",
                 "ms", "wms", "others", "tau",
                 "head_prediction", "pairwise_prediction"],
        type=str,
        required=False,
        help=("The metrics for evaluating the sorted sequence."),
    )
    parser.add_argument(
        "--multiref_metrics",
        default="max",
        choices=["max", "nist", "ngram_bleu"],
        type=str,
        required=False,
        help=("The way to compute multiref gt references."),
    )
    parser.add_argument(
        "--replace_token_type_embeddings",
        action="store_true",
        help="If replace the pretrained token type embedding with new one.",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="If using multimodal inputs",
    )
    parser.add_argument(
        "--use_multimodal_model",
        action="store_true",
        help="If using multimodal models",
    )
    parser.add_argument(
        "--multimodal_text_part",
        action="store_true",
        help="If using multimodal inputs",
    )
    parser.add_argument(
        "--multimodal_img_part",
        action="store_true",
        help="If using multimodal inputs",
    )
    parser.add_argument(
        "--vilbert_without_coattention",
        action="store_true",
        help="If not using coattention",
    )
    parser.add_argument(
        "--freeze_vision_model",
        action="store_true",
        help="If training the vision model",
    )
    parser.add_argument(
        "--multimodal_model_type",
        default="",
        type=str,
        required=False,
        choices=[                                                            
            "naive", "naive_model",
            "visualbert", "vilbert", "vlbert",
            "uniter", "UNITER", "clip",
        ],
        help=("The order criteria of gt labels, see datasets/roc.py"),
    )
    parser.add_argument(
        "--clip_model_name",
        default="RN50",
        type=str,
        required=False,
        choices=[
            "RN50", "ViT-B/32",
        ],
        help=("The type of the visual part of the clip model"),
    )
    parser.add_argument(
        "--clip_visual_model_weights",
        default=None,
        type=str,
        required=False,
        help=("The trained visual model weights"),
    )
    parser.add_argument(
        "--wrapper_model_type",
        default=None,
        type=str,
        required=False,
        choices=[
            "berson",
        ],
        help=("Wrapping the base model."),
    )
    parser.add_argument(
        "--eval_on_every_iter",
        default=None,
        type=int,
        help=("If eval only run on evert indicated iterations,"
              " usually for movie script type datasets."),
    )
    parser.add_argument(
        "--use_cached",
        action="store_true",
        help="If using original cached text processing",
    )
    parser.add_argument(
        "--vision_model",
        default="resnet18",
        type=str,
        required=False,
        # choices=[
        #     "resnet18", "resnet50", "resnet101",
        # ],
        help=("The vision model."),
    )
    parser.add_argument(
        "--vision_feature_dim",
        type=int,
        default=None,
        help="The vision feature dimension.",
    )
    parser.add_argument(
        "--vision_model_checkpoint",
        default=None,
        type=str,
        required=False,
        help=("The vision model pretrained checkpoint."),
    )
    parser.add_argument(
        "--multimodal_fusion_method",
        type=str,
        choices=["sum", "mul", "text_only", "img_only"],
        default="mul",
        help="The fusion method for multimodal models",
    )
    parser.add_argument(
        "--vilbert_use_3way_logits",
        action="store_true",
        help="If using logits from all modalities in ViL-BERT.",
    )
    parser.add_argument(
        "--hierarchical_version",
        type=str,
        choices=["v0", "v1", "v2", "p0", "p1"],
        default="v0",
        help="The version of hierarchical layers.",
    )
    parser.add_argument(
        "--vilbert_paired_coattention",
        action="store_true",
        help="If using paired coattention",
    )
    parser.add_argument(
        "--include_num_img_regional_features",
        type=int,
        default=None,
        help="The number of image regional features to include.",
    )
    parser.add_argument(
        "--include_full_img_features",
        type=int,
        default=None,
        help="To include full image features when using regional feature.",
    )
    parser.add_argument(
        "--heatmap_decode_method",
        default="naive",
        type=str,
        required=False,
        choices=[                                                            
            "super_naive", "naive", "mst", "naive_v2", "naive_v2_sum",
            "naive_sum",
        ],
        help=("The heatmap decoding method."),
    )
    parser.add_argument(
        "--heatmap_decode_beam_size",
        default=1,
        type=int,
        help=("The heatmap naive decode method beam size."),
    )
    parser.add_argument(
        "--caption_transformations",
        default=None,
        type=str,
        required=False,
        nargs="+",
        choices=[                                                            
            "remove_1st",
        ],
        help=("The transformations applied to the captions (surface)."),
    )

    # Other parameters
    parser.add_argument(
        "--config_name_1", default="", type=str, help=("Pretrained config name "
            "or path if not the same as model_name")
    )
    parser.add_argument(
        "--tokenizer_name_1",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--config_name_2", default="", type=str, help=("Pretrained config name "
            "or path if not the same as model_name")
    )
    parser.add_argument(
        "--tokenizer_name_2",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--config_name_3", default="", type=str, help=("Pretrained config name "
            "or path if not the same as model_name")
    )
    parser.add_argument(
        "--tokenizer_name_3",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
             "Sequences longer than this will be truncated, sequences "
             "shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", 
        help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--logging_steps", type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision "
             "(through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in "
             "['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="",
                        help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="",
                        help="For distant debugging.")
    args = parser.parse_args()
    
    if args.output_root is not None:
        args.output_dir = os.path.join(args.output_root, args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port),
                            redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed "
        "training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    
    args.img_transform_func = None

    if "head_and_topological" in args.sort_method:
        (pairwise_model, head_model, tokenizer,
         pairwise_config, head_config) = get_models(args)
        args.model_type = head_config.model_type
        models = [pairwise_model, head_model]
    elif "topological" in args.sort_method:
        pairwise_model, tokenizer, pairwise_config = get_models(args)
        args.model_type = pairwise_config.model_type
        models = [pairwise_model]
    elif args.sort_method == "head_and_pairwise":
        (pairwise_model, head_model, tokenizer,
         pairwise_config, head_config) = get_models(args)
        args.model_type = pairwise_config.model_type
        models = [pairwise_model, head_model]
    elif args.sort_method == "head_and_pairwise_abductive":
        (pairwise_model, head_model, abductive_model, tokenizer,
         pairwise_config, head_config, abductive_config) = get_models(args)
        args.model_type = pairwise_config.model_type
        models = [pairwise_model, head_model, abductive_model]
    elif "pure_classification" in args.sort_method:
        pure_class_model, tokenizer, pure_class_config = get_models(args)
        args.model_type = pure_class_config.model_type
        models = [pure_class_model]
    elif "pure_decode" in args.sort_method:
        encoder_decoder_model, tokenizer, pure_decode_config = get_models(args)
        args.model_type = pure_decode_config.model_type
        models = [encoder_decoder_model]
    elif "heat_map" in args.sort_method:
        hl_model, tokenizer, hl_config = get_models(args)
        args.model_type = hl_config.model_type
        models = [hl_model]
    else:
        raise NotImplementedError("Sort method {} not "
                                  "implemented yet.".format(args.sort_method))

    args.task_name = args.sort_method

    logger.info("Training/evaluation parameters %s", args)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("!!! Number of Params: {} M".format(count_parameters(models[0])/float(1000000)))

    # Evaluation
    results = {}
    if args.local_rank in [-1, 0]:
        for data_split in args.data_splits:
            args.data_split = data_split
            prefix = ""
            result = evaluate(args, models, tokenizer, prefix=prefix)
            result = dict((k, v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
