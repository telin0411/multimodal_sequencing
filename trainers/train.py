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
import csv
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
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertForMaskedLM,
)
from transformers.file_utils import is_sklearn_available, requires_sklearn
from datasets.processors import _pairwise_convert_examples_to_features
from datasets.processors import data_processors as processors
from datasets.processors import output_modes

# Custom dataset classes.
from datasets.processors import HeadPredDataset, PureClassDataset
from datasets.processors import PairwiseDataset

# Multimodality.
from .multimodal_utils import get_multimodal_utils
from models.berson.eval import berson_evaluate

# Some training utils.
from .train_utils import heatmap2order, render_order_heatmap
from .metrics import compute_metrics as compute_metrics_func

# Pretraining.
from trainers.train_utils import mask_tokens_sentence

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    requires_sklearn(simple_accuracy)
    return (preds == labels).mean()


def unbalanced_accuracy(preds, labels):
    acc = 0
    for i in range(len(preds)):
        pred, label = preds[i], labels[i]
        min_len = min(len(pred), len(label))
        pred, label = pred[:min_len], label[:min_len]
        acc += (pred == label).mean()
    return acc / len(preds)


def heatmap_accuracy(preds, labels, args=None, method=None):
    acc = 0
    print_max = 2
    for i in range(len(preds)):
        heatmap, label = preds[i], labels[i]
        min_len = min(len(heatmap), len(label))
        if "p" not in args.hierarchical_version:
            pred = heatmap2order(args, heatmap, soft=False)
        else:
            pred = heatmap
        if i < print_max:
            logging.info("\n{}".format(heatmap))
            logging.info("Pred: {}  Label: {}".format(pred, label))
            logging.info("\n{}".format(render_order_heatmap(args, label, soft=False, ranking_based=False)))
            logging.info('-'*50)
        acc_curr = compute_metrics_func(args, method, [pred], [label])
        acc += acc_curr
    return acc / len(preds)


def compute_metrics(task_name, preds, labels, args=None):
    requires_sklearn(compute_metrics)
    assert len(preds) == len(labels), f("Predictions and labels have "
                                       "mismatched lengths {len(preds)} "
                                       "and {len(labels)}")
    if ("pairwise" in task_name or "head" in task_name
        or "pure_class" in task_name):
        return {"{}_acc".format(task_name): simple_accuracy(preds, labels)}
    elif "pure_decode" in task_name:
        return {"{}_acc".format(task_name): unbalanced_accuracy(preds, labels)}
    elif "hl_v" in task_name:
        if args.metrics is None:
            return {
                "1. {}_partial_acc".format(task_name): heatmap_accuracy(preds, labels, args=args, method="partial_match"),
                "2. {}_exact_acc".format(task_name): heatmap_accuracy(preds, labels, args=args, method="exact_match"),
                "3. {}_distance_based".format(task_name): heatmap_accuracy(preds, labels, args=args, method="distance_based"),
                "4. {}_lcs".format(task_name): heatmap_accuracy(preds, labels, args=args, method="lcs"),
                "5. {}_min_swap".format(task_name): heatmap_accuracy(preds, labels, args=args, method="ms"),
                "6. {}_weighted_min_swap".format(task_name): heatmap_accuracy(preds, labels, args=args, method="wms"),
            }
        else:
            res_dict = {}
            for i in range(len(args.metrics)):
                metric_key = "{}. {}".format(i+1, args.metrics[i])
                acc_ = heatmap_accuracy(preds, labels, args=args, method=args.metrics[i])
                res_dict[metric_key] = acc_
            return res_dict
    else:
        raise KeyError(task_name)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0] and not args.debug:
        multimodal_str = "multimodal" if args.multimodal else "unimodal"
        output_str = args.output_dir.split("/")[-1]
        task_str = "_".join(args.task_names)
        comment_str = "_{}_{}_{}".format(output_str, multimodal_str,
                                         task_str)
        tb_writer = SummaryWriter(comment=comment_str)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 \
        else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) \
                                // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
                       # if not any(nd in n for nd in no_decay) and "hl_bin_hm_pred_layer" not in n],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
                    # if any(nd in n for nd in no_decay) and "hl_bin_hm_pred_layer" not in n], "weight_decay": 0.0},
        # {"params": model.model.heatmap.hl_bin_hm_pred_layer.parameters(), "lr": args.learning_rate * 10}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")
        and not args.do_not_load_optimizer
    )):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(
                                  args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(
                                  args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github."
                              "com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed "
        "& accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    best_acc = 0
    best_pmr = 0
    # Check if continuing training from a checkpoint
    if (os.path.exists(args.model_name_or_path)
        and not args.do_not_load_optimizer):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            # global_step = int(
            #     args.model_name_or_path.split("-")[-1].split("/")[0])
            global_step = int(
                args.model_name_or_path.split("/")[-1].split("-")[-1])
        except:
            global_step = 0  # If start fresh.
        epochs_trained = global_step // (len(train_dataloader) \
                         // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) \
                                         // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip"
                    " to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch",
        disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch if type(t) != tuple)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "roberta"] else None
                )  # XLM and DistilBERT don't use segment_ids
                if not args.replace_token_type_embeddings:
                    inputs["token_type_ids"] = None

            if args.include_num_img_regional_features is not None:
                inputs["img_regional_features"] = batch[-2]
            # TODO: pretraining aux objectives.
            if (args.hl_include_objectives is not None
                and ("mlm" in args.hl_include_objectives 
                     or "mlm_wo_loss" in args.hl_include_objectives)):
                masked_inputs, lm_labels = mask_tokens_sentence(
                    inputs["input_ids"], tokenizer, args)
                inputs["input_ids"] = masked_inputs
                inputs["masked_lm_labels"] = lm_labels
            if args.multimodal:
                inputs["images"] = batch[-1]
                if args.multimodal_model_type == "clip" and args.wrapper_model_type is None:
                    inputs["visual_feats"] = inputs["images"]
                    del inputs["images"]
                    outputs = model(**inputs)
                else:
                    outputs = model(inputs)
            elif "pure_decode" in args.task_type:
                outputs = model(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                token_type_ids=inputs["token_type_ids"],
                                decoder_input_ids=inputs["labels"],
                                labels=inputs["labels"],
                                return_dict=True)
            elif args.use_multimodal_model:
                # from models.berson.process_inputs_for_berson import prepare_berson_inputs
                # berson_inputs = prepare_berson_inputs(inputs, tokenizer, args=args)
                # for k in berson_inputs:
                #     print(k, berson_inputs[k])
                # raise
                # outputs = model(**berson_inputs)
                outputs = model(inputs)
            elif args.wrapper_model_type is not None:
                outputs = model(inputs)
            else:
                outputs = model(**inputs)

            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            # loss = outputs

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel training
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # print(loss)
            # print(torch.sum(model.bert.encoder.layer[0].attention.self.query.weight))

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (args.local_rank in [-1, 0] and args.logging_steps > 0
                    and global_step % args.logging_steps == 0):
                    # Log metrics
                    if (
                        # Only evaluate when single GPU otherwise metrics may
                        # not average well
                        args.local_rank == -1 and args.evaluate_during_training
                    ):
                        if type(args.eval_splits) != list:
                            args.eval_splits = [args.eval_splits]
                        for eval_split in args.eval_splits:
                            if "berson" == args.wrapper_model_type:
                                results = berson_evaluate(args, model,
                                    load_and_cache_examples, tokenizer,
                                    prefix="", data_split=eval_split,
                                    human_evaluate=False)
                            else:
                                results = evaluate(args, model, tokenizer,
                                                   data_split=eval_split)
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_on_{}_{}".format(eval_split, key),
                                    value, global_step)
                            if results["acc_dev"] + results["pmr_dev"] >= best_acc + best_pmr:
                                # Save model checkpoint
                                logger.info("Previous best acc: {:.3f}  best pmr: {:.3f}".format(best_acc, best_pmr))
                                best_acc, best_pmr = results["acc_dev"], results["pmr_dev"]
                                logger.info("New      best acc: {:.3f}  best pmr: {:.3f}".format(best_acc, best_pmr))
                                output_dir = os.path.join(args.output_dir,
                                    "checkpoint-{}".format("best"))
                                # FIXME:
                                output_dir = os.path.join(args.best_dir,
                                    "checkpoint-{}".format("best"))
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)

                                torch.save(args, os.path.join(output_dir,
                                           "training_args.bin"))
                                logger.info("Saving model checkpoint to %s", output_dir)

                                torch.save(optimizer.state_dict(), os.path.join(
                                    output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(
                                    output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s",
                                            output_dir)
                                output_eval_file = os.path.join(args.output_dir,
                                                                "best_eval_results_split_{}.txt".format(eval_split))
                                with open(output_eval_file, "w") as writer:
                                    for key in sorted(results.keys()):
                                        writer.write("%s = %s\n" % (key, str(results[key])))
                                pass  #####
                                
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0],
                                         global_step)
                    tb_writer.add_scalar("loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step)
                    logging_loss = tr_loss

                if (args.local_rank in [-1, 0] and args.save_steps > 0
                    and global_step % args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir,
                        "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir,
                               "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(
                        output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(
                        output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s",
                                output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", data_split="test"):
    # eval_task_names = (args.task_name,)
    eval_task_names = args.task_names
    # eval_outputs_dirs = (args.output_dir,)
    eval_outputs_dirs = [args.output_dir] * len(args.task_names)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, [eval_task],
                                               tokenizer, evaluate=True,
                                               data_split=data_split)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation on split: {} {} *****".format(
            data_split, prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        guids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) if type(t) != tuple else t for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                          "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "roberta"] \
                                 else None
                    )  # XLM and DistilBERT don't use segment_ids
                    if not args.replace_token_type_embeddings:
                        inputs["token_type_ids"] = None

                # TODO: pretraining aux objectives.
                if (args.hl_include_objectives is not None
                    and ("mlm" in args.hl_include_objectives 
                         or "mlm_wo_loss" in args.hl_include_objectives)):
                    masked_inputs, lm_labels = mask_tokens_sentence(
                        inputs["input_ids"], tokenizer, args)
                    inputs["input_ids"] = masked_inputs
                    inputs["masked_lm_labels"] = lm_labels
                if args.multimodal:
                    inputs["images"] = batch[-1]
                    if args.include_num_img_regional_features:
                        inputs["img_regional_features"] = batch[-2]
                    if args.multimodal_model_type == "clip" and args.wrapper_model_type is None:
                        inputs["visual_feats"] = inputs["images"]
                        del inputs["images"]
                        outputs = model(**inputs)
                    else:
                        outputs = model(inputs)
                    tmp_eval_loss, logits = outputs[:2]
                elif "pure_decode" in args.task_type:
                    outputs = model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    token_type_ids=inputs["token_type_ids"],
                                    decoder_input_ids=inputs["labels"],
                                    labels=inputs["labels"],
                                    return_dict=True)
                    tmp_eval_loss, logits = outputs.loss, outputs.logits
                elif args.use_multimodal_model:
                    outputs = model(inputs)
                    tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                guid = batch[4][0]
                guids = [str(guid).split("###")[0]]
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids,
                    inputs["labels"].detach().cpu().numpy(), axis=0)
                guids.append(batch[4][0].split("###")[0])

            if args.max_eval_steps > 0 and nb_eval_steps >= args.max_eval_steps:
                logging.info("Early stopping"
                    " evaluation at step: {}".format(args.max_eval_steps))
                break

        eval_loss = eval_loss / nb_eval_steps
        if (args.output_mode == "classification"
            and args.hierarchical_version == "v0"):
            preds = np.argmax(preds, axis=1)
        elif args.hierarchical_version != "v0":
            pass
        else:
            raise ValueError("No other `output_mode` for Sorting.")
        result = compute_metrics(eval_task, preds, out_label_ids, args)
        eval_loss_dict = {"{}_loss".format(eval_task): eval_loss}
        results.update(result)
        results.update(eval_loss_dict)

        if args.eval_save_all_results:

            out_csv = os.path.join(args.output_dir, "all_predictions.csv")
            csv_f = open(out_csv, "w")
            fieldnames = ["url", "pm", "em", "lcs_substr", "lcs", "ms", "wms", "dist"]
            csv_r = csv.DictWriter(csv_f, fieldnames=fieldnames)
            csv_r.writeheader()

            for c in range(len(preds)):
                pred, label = [preds[c]], [out_label_ids[c]]
                
                metric_methods = {
                    "pm": "partial_match",
                    "em": "exact_match",
                    "lcs_substr": "lcs_substr",
                    "lcs": "lcs",
                    "ms": "ms",
                    "wms": "wms",
                    "dist": "distance_based",
                }
                for met in ["pm", "em", "lcs_substr", "lcs", "ms", "wms", "dist"]:
                    met_name = metric_methods[met]
                    acc_c = heatmap_accuracy(pred, label, args=args, method=met_name)
                    row[met] = acc_c
                url = guids[c]
                row["url"] = url
                csv_r.write(row)

            csv_f.close()
            print("Saving all prediction csv file at: {}".format(out_csv))

    output_eval_file = os.path.join(eval_output_dir,
                                    prefix, "eval_results_split_{}.txt".format(data_split))

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} on split: {} *****".format(prefix, data_split))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

    return results


def load_and_cache_examples(args, tasks, tokenizer,
                            evaluate=False, data_split="test"):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier()

    # Getting the examples.

    split_version_text = data_split.split("-")[-1]
    if len(data_split.split("-")) == 1:
        split_version_text = None
    else:
        split_version_text = split_version_text.strip()
    data_split_curr = data_split.split("-")[0]
    args.split_version_text = split_version_text

    examples = []
    for i in range(len(tasks)):
        task = tasks[i]
        data_dir_curr = args.data_dirs[i]
        logging.info("Processing task: {}".format(task))

        # TODO: Caption transformations
        if args.caption_transformations is not None:
            assert type(args.caption_transformations) == list
            from trainers.caption_utils import CaptionTransformations
            caption_transformation_list = []
            for caption_transformation in args.caption_transformations:
                if "train" in caption_transformation and not evaluate:
                    caption_transformation_list.append(caption_transformation)
                elif "eval" in caption_transformation and evaluate:
                    caption_transformation_list.append(caption_transformation)
                elif "train" not in caption_transformation and "eval" not in caption_transformation:
                    caption_transformation_list.append(caption_transformation)
            
            if len(caption_transformation_list) == 0:
                caption_transforms = None
            else:
                caption_transforms = CaptionTransformations(args, task, caption_transformation_list)

            if "recipeqa" not in task and "wikihow" not in task:
                raise NotImplementedError("Only deal with RecipeQA and WikiHow now!")
        else:
            caption_transforms = None

        if "pairwise" in task:
            processor = processors[task](order_criteria=args.order_criteria,
                                         caption_transforms=caption_transforms,
                                         version_text=split_version_text)
        elif "head" in task:
            processor = processors[task](max_story_length=args.max_story_length,
                                         min_story_length=args.min_story_length,
                                         caption_transforms=caption_transforms,
                                         version_text=split_version_text)
        elif "pure_class" in task:
            processor = processors[task](max_story_length=args.max_story_length,
                                         min_story_length=args.min_story_length,
                                         pure_class=True,
                                         caption_transforms=caption_transforms,
                                         version_text=split_version_text)
        elif "pure_decode" in task:
            processor = processors[task](max_story_length=args.max_story_length,
                                         min_story_length=args.min_story_length,
                                         pure_class=False,
                                         caption_transforms=caption_transforms,
                                         version_text=split_version_text)
        elif "hl_v" in task:
            processor = processors[task](max_story_length=args.max_story_length,
                                         min_story_length=args.min_story_length,
                                         pure_class=False,
                                         caption_transforms=caption_transforms,
                                         version_text=split_version_text)

        if data_split_curr == "test" and evaluate:
            examples_curr = (processor.get_test_examples())
        elif data_split_curr == "val" and evaluate or data_split_curr == "dev":
            examples_curr = (processor.get_dev_examples())
        elif data_split_curr == "train" and evaluate:
            examples_curr = (processor.get_train_examples())
        else:
            examples_curr = (
                processor.get_test_examples()
                if evaluate else processor.get_train_examples()
            )
        examples += examples_curr
        output_mode = output_modes[args.task_type]

    # Load data features from cache or dataset file
    if "pairwise" in args.task_type and not args.multimodal and args.use_cached:
        if args.data_names is not None:
            task_names_str = "_".join(sorted(args.data_names))
            data_dir_str = "joint"
        else:
            task_names_str = args.task_name
            data_dir_str = args.data_dir
        cached_features_file = os.path.join(
            data_dir_str,
            "cached_{}_{}_{}_{}".format(
                "test" if evaluate else "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task_names_str),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s",
                        cached_features_file)
            features = torch.load(cached_features_file)

        else:
            logger.info("Creating features from dataset file at %s", data_dir_str)
            label_list = processor.get_labels()
            if "pairwise" in task:
                convert_examples_to_features = _pairwise_convert_examples_to_features
            elif "head" in task:
                convert_examples_to_features = _head_convert_examples_to_features
            else:
                raise NotImplementedError("Task: {} not handled.".format(task))

            features = convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                label_list=label_list,
                output_mode=output_mode,
                multimodal=args.multimodal,
                img_transform_func=args.img_transform_func,
            )

            if args.local_rank in [-1, 0] and not args.multimodal:
                logger.info("Saving features into cached file %s",
                            cached_features_file)
                torch.save(features, cached_features_file)

    elif "pairwise" in args.task_type and (args.multimodal or not args.use_cached):
        label_list = processor.get_labels()
        dataset = PairwiseDataset(examples, tokenizer,
                                  args=args,
                                  max_length=args.max_seq_length,
                                  per_seq_max_length=args.per_seq_max_length,
                                  max_story_length=args.max_story_length,
                                  min_story_length=args.min_story_length,
                                  multimodal=args.multimodal,
                                  img_transform_func=args.img_transform_func,
                                  processor=processor,
                                  num_img_regional_features=args.include_num_img_regional_features,
                                  seed=args.seed)

    elif "head" in args.task_type:
        label_list = processor.get_labels()
        dataset = HeadPredDataset(examples, tokenizer,
                                  args=args,
                                  max_length=args.max_seq_length,
                                  per_seq_max_length=args.per_seq_max_length,
                                  max_story_length=args.max_story_length,
                                  min_story_length=args.min_story_length,
                                  multimodal=args.multimodal,
                                  img_transform_func=args.img_transform_func,
                                  num_img_regional_features=args.include_num_img_regional_features,
                                  seed=args.seed)

    elif "pure_class" in args.task_type:
        label_list = processor.get_labels()
        dataset = PureClassDataset(examples, tokenizer,
                                   args=args,
                                   max_length=args.max_seq_length,
                                   per_seq_max_length=args.per_seq_max_length,
                                   max_story_length=args.max_story_length,
                                   min_story_length=args.min_story_length,
                                   multimodal=args.multimodal,
                                   img_transform_func=args.img_transform_func,
                                   num_img_regional_features=args.include_num_img_regional_features,
                                   seed=args.seed)

    elif "pure_decode" in args.task_type:
        label_list = processor.get_labels()
        dataset = PureClassDataset(examples, tokenizer,
                                   args=args,
                                   max_length=args.max_seq_length,
                                   per_seq_max_length=args.per_seq_max_length,
                                   max_story_length=args.max_story_length,
                                   min_story_length=args.min_story_length,
                                   multimodal=args.multimodal,
                                   img_transform_func=args.img_transform_func,
                                   num_img_regional_features=args.include_num_img_regional_features,
                                   seed=args.seed, decode=True)

    elif "hl_v" in args.task_type:
        label_list = processor.get_labels()
        dataset = PureClassDataset(examples, tokenizer,
                                   args=args,
                                   max_length=args.max_seq_length,
                                   per_seq_max_length=args.per_seq_max_length,
                                   max_story_length=args.max_story_length,
                                   min_story_length=args.min_story_length,
                                   multimodal=args.multimodal,
                                   img_transform_func=args.img_transform_func,
                                   num_img_regional_features=args.include_num_img_regional_features,
                                   seed=args.seed, decode=True)
        logging.info("Heatmap Decode Beam Size = {}".format(
            args.heatmap_decode_beam_size))

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier() 

    # Convert to Tensors and build dataset
    if "pairwise" in args.task_type and not args.multimodal and args.use_cached:
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                          dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                          dtype=torch.long)
        if args.multimodal:
            all_images = [f.images for f in features]
            all_images = torch.stack(all_images)

        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features],
                                      dtype=torch.long)
        else:
            raise ValueError("No other `output_mode` for Sorting.")

        if args.multimodal:
            dataset = TensorDataset(all_input_ids, all_attention_mask,
                                    all_token_type_ids, all_labels,
                                    all_images)
            return dataset

        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_labels)
    return dataset


def main():
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=False,
        help="The model type.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        type=str,
        required=False,
        help="The input data root dir.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help=("Path to pretrained model or model identifier from "
              "huggingface.co/models"),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=("The output directory where the model predictions and "
              "checkpoints will be written."),
    )
    parser.add_argument(
        "--output_root",
        default=None,
        type=str,
        help=("The prefix root directory where the model predictions and "
              "checkpoints will be written."),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=False,
        help=("The name of the task, see datasets/processors.py"),
    )
    parser.add_argument(
        "--task_type",
        default=None,
        type=str,
        required=False,
        choices=["pairwise", "head", "abductive", "pure_class", "pure_decode"],
        help=("The type of the task, see datasets/processors.py"),
    )
    parser.add_argument(
        "--data_names",
        default=None,
        nargs="+",
        type=str,
        required=False,
        help=("The names of the datasets, see datasets/processors.py"),
    )
    parser.add_argument(
        "--data_dirs",
        default=None,
        nargs="+",
        type=str,
        required=False,
        help=("The dirs of the datasets, see datasets/processors.py"),
    )
    parser.add_argument(
        "--order_criteria",
        default="tight",
        type=str,
        required=False,
        choices=["tight", "loose"],
        help=("The order criteria of gt labels, see datasets/roc.py"),
    )
    parser.add_argument(
        "--max_story_length",
        default=5,
        type=int,
        required=False,
        help=("The maximum length of the story sequence."),
    )
    parser.add_argument(
        "--min_story_length",
        default=5,
        type=int,
        required=False,
        help=("The minimum length of the story sequence."),
    )
    parser.add_argument(
        "--per_seq_max_length",
        default=64,
        type=int,
        required=False,
        help=("The maximum length of EACH of the story sequence."),
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
        "--multimodal_loss",
        action="store_true",
        help="If using multimodal loss training",
    )
    parser.add_argument(
        "--use_multimodal_model",
        action="store_true",
        help="If using multimodal models",
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
        help=("The type of the multimodal model"),
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
        "--ref_json_file",
        default=None,
        type=str,
        required=False,
        help=("The reference json file for performance recording"),
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
        "--additional_wrapper_level_objectives",
        default=None,
        type=str,
        required=False,
        choices=[
            "time_contrastive",
        ],
        nargs="+",
        help=("Objectives added on top of wrapper model."),
    )
    parser.add_argument(
        "--wrapper_model_with_heatmap",
        action="store_true",
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
        "--multimodal_fusion_method",
        type=str,
        choices=["sum", "mul", "text_only", "img_only"],
        default="mul",
        help="The fusion method for multimodal models",
    )
    parser.add_argument(
        "--do_not_load_optimizer",
        action="store_true",
        help="If prohibit loading the optimizer state dict (using pretrain).",
    )
    parser.add_argument(
        "--vilbert_use_3way_logits",
        action="store_true",
        help="If using logits from all modalities in ViL-BERT.",
    )
    parser.add_argument(
        "--hierarchical_version",
        type=str,
        choices=["v0", "v1", "v2", "v3", "p0", "p1"],
        default="v0",
        help="The version of hierarchical layers.",
    )
    parser.add_argument(
        "--vilbert_without_coattention",
        action="store_true",
        help="If not using coattention",
    )
    parser.add_argument(
        "--vilbert_paired_coattention",
        action="store_true",
        help="If using paired coattention",
    )
    parser.add_argument(
        "--vilbert_original_configs",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--img_text_paired_coattention",
        action="store_true",
        help="If using paired coattention",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="The splits to evaluate on.",
    )
    parser.add_argument(
        "--eval_splits",
        type=str,
        # choices=["train", "val", "test"],
        default="test",
        nargs="+",
        help="The splits to evaluate on.",
    )
    parser.add_argument(
        "--vilbert_v_num_hidden_layers",
        type=int,
        default=3,
        help="The number of hidden layers for v stream.",
    )
    parser.add_argument(
        "--include_full_img_features",
        type=int,
        default=None,
        help="To include full image features when using regional feature.",
    )
    parser.add_argument(
        "--include_num_img_regional_features",
        type=int,
        default=None,
        help="The number of image regional features to include.",
    )
    parser.add_argument(
        "--heatmap_decode_method",
        default="naive",
        type=str,
        required=False,
        choices=[                                                            
            "super_naive", "naive", "naive_v2", "naive_v2_sum", "naive_sum", "naive_v3",
            "mst", "topological",
        ],
        help=("The heatmap decoding method."),
    )
    parser.add_argument(
        "--hl_include_objectives",
        default=None,
        type=str,
        required=False,
        nargs="+",
        choices=[                                                            
            "head", "binary", "mlm", "itm", "mlm_wo_loss",
            "binary_cross_modal", "cross_modal_dependence",
            "heatmap_module", "heatmap_pairwise_ranking",
            "variable_length_lstm", "variable_length_transformer",
            "variable_length_cross_modal", "pairwise_binary_heatmap",
        ],
        help=("The heatmap model with some auxiliary objectives."),
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
        # choices=[                                                            
        #     "remove_1st", "train_max_sentence_4", "eval_max_sentence_4",
        # ],
        help=("The transformations applied to the captions (surface)."),
    )
    parser.add_argument(
        "--eval_save_all_results",
        action="store_true",
        help="If saving per sample predicted results",
    )

    # Pretraining args.
    parser.add_argument(
        "--mlm_probability",
        default=0.8,
        type=float,
        required=False,
        help=("The MLM probability."),
    )
    parser.add_argument(
        "--mlm_ignore_index",
        default=-100,
        type=int,
        required=False,
        help=("The MLM CE loss ignored index."),
    )

    # Other parameters
    parser.add_argument(
        "--debug",
        action="store_true",
        help="The mode of development, does not save tensorboard summaries.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        nargs="+",
        choices=["partial_match", "exact_match", "distance_based",
                 "longest_common_subsequence", "lcs",
                 "longest_common_substring", "lcs_substr",
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
        "--config_name", default="", type=str, help=("Pretrained config name "
            "or path if not the same as model_name")
    )
    parser.add_argument(
        "--img_config_name", default=None, type=str, help=("Pretrained config name "
            "or path if not the same as model_name")
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help=("Where do you want to store the pre-trained models "
              "downloaded from s3"),
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
             "Sequences longer than this will be truncated, sequences "
             "shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true",
        help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", 
        help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing "
             "a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. "
             "Override num_train_epochs.",
    )
    parser.add_argument(
        "--max_eval_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of evaluating steps to perform.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as "
             "model_name ending and ending with step number",
    )
    parser.add_argument("--iters_to_eval", default=None, type=str, nargs="+",
                        help="Iterations of checkpoints to evaluate.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
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

    # Writes the prefix to the output dir path.
    args.best_dir = args.output_dir
    if not os.path.exists(args.best_dir):
        os.makedirs(args.best_dir)
    if args.output_root is not None:
        args.output_dir = os.path.join(args.output_root, args.output_dir)
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use "
            "--overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    if args.vilbert_paired_coattention:
        args.img_text_paired_coattention = True

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

    # TODO Multiple datasets.
    if args.task_name is None and args.data_names is None:
        raise ValueError("At least one of `task_name`"
                         " and `data_names` should NOT be None.")
    elif args.task_name is not None and args.data_names is not None:
        raise ValueError("At least one of `task_name`"
                         " and `data_names` should BE None.")
    if args.data_dir is None and args.data_dirs is None:
        raise ValueError("At least one of `data_dir`"
                         " and `data_dirs` should not be None.")
    elif args.data_dir is not None and args.data_dirs is not None:
        raise ValueError("At least one of `data_dir`"
                         " and `data_dirs` should BE None.")

    # Prepare the task
    args.task_names = []
    if args.task_name is not None:
        args.task_names = [args.task_name]
    else:
        for data_name in args.data_names:
            task_name = "{}_{}".format(data_name, args.task_type)
            if task_name not in processors:
                raise ValueError("Task not found: %s" % (task_name))
            args.task_names.append(task_name)
    for task_name in args.task_names:
        if "pairwise" in task_name:
            processor = processors[task_name](
                order_criteria=args.order_criteria)
        elif "head" in task_name:
            processor = processors[task_name](
                max_story_length=args.max_story_length,
                min_story_length=args.min_story_length)
        elif "pure_class" in task_name:
            processor = processors[task_name](
                max_story_length=args.max_story_length,
                min_story_length=args.min_story_length,
                pure_class=True)
        elif "pure_decode" in task_name:
            processor = processors[args.task_name](
                max_story_length=args.max_story_length,
                min_story_length=args.min_story_length,
                pure_class=False)
        elif "hl_v" in task_name:
            paired_with_image = True
            if args.multimodal_text_part:
                paired_with_image = False
            processor = processors[args.task_name](
                max_story_length=args.max_story_length,
                min_story_length=args.min_story_length,
                paired_with_image=paired_with_image,
                pure_class=False)
    
    if args.task_name is not None:
        args.task_type = args.task_name
        args.data_dirs = [args.data_dir]
    
    args.output_mode = output_modes[args.task_type]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier() 

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    if args.model_type is None:
        args.model_type = config.model_type
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )
    bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    args.cls_id = cls_id

    if args.multimodal or args.use_multimodal_model:
        # TODO: Adds args.
        if args.vision_model_checkpoint is not None:
            vision_model = 0
        elif args.use_multimodal_model and not args.multimodal:
            vision_model, img_transform_func = None, None
        else:
            vision_model, img_transform_func = get_multimodal_utils(args=args)
            args.img_transform_func = img_transform_func
        # TODO: Change the followings.
        if args.multimodal_model_type in ["naive", "naive_model"]:
            # model = AutoModelForSequenceClassification.from_pretrained(
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )
            # from models.naive_model import NaiveMultimodalModel
            # model = NaiveMultimodalModel(args=args,
            #                              language_model=model,
            #                              vision_model=vision_model)
            # TODO Test loading.
            # print("Loading models")
            # model_to_load = model.module if hasattr(model, 'module') else model
            # model_to_load.load_state_dict(torch.load(
            #     "exp_outputs/recipeQA/recipeqa_head_multimodal"
            #     "/trial1/checkpoint-10/pytorch_model.bin"))
            # print("Done Loading models")
            # raise

        elif args.multimodal_model_type == "visualbert":
            from models.visualbert.visual_bert_mmf import VisualBERT
            from transformers import BertConfig
            # More configs.
            config = BertConfig.from_pretrained(args.config_name)
            if args.multimodal_img_part and args.img_config_name is not None:
                config = BertConfig.from_pretrained(args.img_config_name)
            if "roberta" in args.model_name_or_path:
                config.model = "roberta"
                config.model_type = "roberta"
            # if "v_target_size" not in config.__dict__:
            #     config.v_target_size = 1601
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
            config.cls_id = cls_id
            args.cls_id = cls_id
            config.hierarchical_version = args.hierarchical_version
            config.num_labels = num_labels
            config.num_labels = 2
            config.max_story_length = args.max_story_length
            config.mlm_ignore_index = args.mlm_ignore_index
            config.img_text_paired_coattention = args.img_text_paired_coattention
            config.include_num_img_regional_features = args.include_num_img_regional_features
            config.hl_include_objectives = args.hl_include_objectives
            if args.hl_include_objectives is not None:
                if "itm" in args.hl_include_objectives:
                    config.swapping_based_nsp = True
                else:
                    config.swapping_based_nsp = False

            if args.hierarchical_version == "v0":
                config.training_head_type = "sort_pairwise"
            else:
                config.training_head_type = "sort_heatmap"
            config.multimodal_loss = args.multimodal_loss

            if False:
                args.pretrained_bin = os.path.join(args.model_name_or_path,
                                                   "pytorch_model_visualbert_mmf.bin")

                if not os.path.exists(args.pretrained_bin):
                    args.pretrained_bin_ = os.path.join(args.model_name_or_path,
                                                        "pytorch_model.bin")
                    state_dict = torch.load(args.pretrained_bin_)
                    state_dict_new = {}
                    for k, v in state_dict.items():
                        if "roberta" in k:
                            # k_new = k.partition(
                            #     "{}.".format(config.model_type))[2]
                            k_new = k.replace("roberta", "model.bert")
                            # k_new = k.replace("roberta.", "roberta.")
                            state_dict_new[k_new] = v
                        elif "bert" in k:
                            k_new = k.replace("bert", "model.bert")
                            k_new = k_new.replace("LayerNorm.gamma", "LayerNorm.weight")
                            k_new = k_new.replace("LayerNorm.beta", "LayerNorm.bias")
                            state_dict_new[k_new] = v
                        else:
                            state_dict_new[k] = v
                    state_dict = state_dict_new
                    # for k, v in state_dict.items():
                    #     print("new", k)
                    torch.save(state_dict, args.pretrained_bin)
                    raise ValueError("Saved before?")
                else:
                    pass

            if args.multimodal_img_part and args.img_config_name is not None:
                # Initializing
                from mmf.utils.configuration import load_yaml
                img_config_name = os.path.join(args.model_name_or_path, "config.yaml")
                img_config = load_yaml(img_config_name)
                model = VisualBERT.from_pretrained(args.model_name_or_path,  # img_config,
                    vision_model=vision_model, tokenizer=tokenizer,
                    multimodal_text_part=args.multimodal_text_part,
                    multimodal_img_part=args.multimodal_img_part,
                    additional_config=config)
                # model.build()
            else:
                model = VisualBERT.from_pretrained(args.model_name_or_path,
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
            if "roberta" in args.model_name_or_path or "roberta" == args.model_type:
                if ("roberta/large" in args.model_name_or_path
                    or "roberta/base" in args.model_name_or_path):
                    args.model_name_or_path = os.path.join(args.model_name_or_path,
                                                           "pytorch_model_uniter.bin")
                    model_config = os.path.join(args.model_name_or_path,
                                                "config.json")
                else:
                    model_config = args.model_name_or_path.replace(
                        "pytorch_model.bin", "config.json")
                logging.info("[ROBERTA] using roberta!")
            checkpoint = torch.load(args.model_name_or_path)
            logging.info("[UNITER] Loading from: {}".format(args.model_name_or_path))
            IMG_DIM = 2048
            if vision_model is not None:
                IMG_DIM = vision_model.fc.in_features
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            # model = UniterForVisualCommonsenseReasoning.from_pretrained(
            #     model_config, checkpoint, img_dim=IMG_DIM)

            # More configs.
            config = BertConfig.from_json_file(model_config)
            if "roberta" in args.model_name_or_path:
                config.model = "roberta"
                config.model_type = "roberta"
            if "v_target_size" not in config.__dict__:
                config.v_target_size = 1601
            config.task_specific_tokens = False
            config.dynamic_attention = False
            config.v_feature_size = vision_model.fc.in_features
            config.freeze_vision_model = args.freeze_vision_model
            config.multimodal_text_part = args.multimodal_text_part
            config.multimodal_img_part = args.multimodal_img_part
            config.fusion_method = args.multimodal_fusion_method
            config.vilbert_use_3way_logits = args.vilbert_use_3way_logits
            bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
            cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
            config.cls_id = cls_id
            args.cls_id = cls_id
            config.hierarchical_version = args.hierarchical_version
            config.num_labels = num_labels
            config.max_story_length = args.max_story_length
            config.mlm_ignore_index = args.mlm_ignore_index
            config.img_text_paired_coattention = args.img_text_paired_coattention
            config.include_num_img_regional_features = args.include_num_img_regional_features
            config.hl_include_objectives = args.hl_include_objectives

            model = UniterForVisualQuestionAnswering.from_pretrained(
                model_config, checkpoint, img_dim=IMG_DIM,
                num_answer=num_labels,
                vision_model=vision_model, tokenizer=tokenizer,
                multimodal_text_part=args.multimodal_text_part,
                multimodal_img_part=args.multimodal_img_part,
                additional_config=config)

        elif args.multimodal_model_type == "vilbert":
            from models.vilbert.vilbert import VILBertForVLTasks
            from models.vilbert.vilbert import BertConfig
            args.config_file = os.path.join(args.model_name_or_path,
                                            "config.json")
            config = BertConfig.from_json_file(args.config_file)
            if "roberta" in args.model_name_or_path:
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
            if bos_id is not None:
                if cls_id == bos_id:
                    config.cls_id = cls_id
                else:
                    config.cls_id = bos_id
            else:
                config.cls_id = cls_id
            args.cls_id = cls_id
            config.hierarchical_version = args.hierarchical_version
            config.max_story_length = args.max_story_length
            if args.vilbert_without_coattention:
                config.with_coattention = False
            config.simple_img_classifier = False
            config.vilbert_paired_coattention = args.vilbert_paired_coattention
            config.img_text_paired_coattention = args.img_text_paired_coattention
            config.v_num_hidden_layers = args.vilbert_v_num_hidden_layers
            config.include_num_img_regional_features = args.include_num_img_regional_features
            config.hl_include_objectives = args.hl_include_objectives
            config.mlm_ignore_index = args.mlm_ignore_index
            config.heatmap_decode_method = args.heatmap_decode_method
            # config.v_hidden_size = 1024
            # config.v_num_attention_heads = 8
            # FIXME: Comment below
            # config.v_num_hidden_layers = 1
            # config.v_intermediate_size = 1024
            # config.v_biattention_id = [0]
            # config.t_biattention_id = [10]
            if args.hl_include_objectives is not None:
                if "itm" in args.hl_include_objectives:
                    config.swapping_based_nsp = True
                else:
                    config.swapping_based_nsp = False
            config.multimodal_loss = args.multimodal_loss

            default_gpu = False
            if args.local_rank in [-1, 0]:
                default_gpu = True

            if args.vilbert_original_configs is not None:
                org_config = BertConfig.from_json_file(args.vilbert_original_configs)
                config.v_num_hidden_layers = org_config.v_num_hidden_layers
                config.bi_attention_type = org_config.bi_attention_type
                config.v_attention_probs_dropout_prob = org_config.v_attention_probs_dropout_prob
                config.v_biattention_id = org_config.v_biattention_id
                config.t_biattention_id = org_config.t_biattention_id
                config.v_hidden_size = org_config.v_hidden_size
                config.v_num_attention_heads = org_config.v_num_attention_heads

            args.pretrained_bin = os.path.join(args.model_name_or_path,
                                               "pytorch_model.bin")
            model, loading_info = VILBertForVLTasks.from_pretrained(
                args.pretrained_bin,
                config=config,
                num_labels=num_labels,
                default_gpu=default_gpu,
                output_loading_info=True,
                print_missing_weights_info=True,
                vision_model=vision_model,
            )
            print()
            if ("{}.embeddings.word_embeddings.weight".format(config.model_type)
                in loading_info["unexpected_keys"]):
                logging.info("Extracting new models...")
                args.pretrained_bin = os.path.join(args.model_name_or_path,
                                                   "pytorch_model_stripped.bin")
                if not os.path.exists(args.pretrained_bin):
                    args.pretrained_bin_ = os.path.join(args.model_name_or_path,
                                                        "pytorch_model.bin")
                    state_dict = torch.load(args.pretrained_bin_)
                    state_dict_new = {}
                    for k, v in state_dict.items():
                        if config.model_type in k:
                            k_new = k.partition(
                                "{}.".format(config.model_type))[2]
                            state_dict_new[k_new] = v
                        else:
                            state_dict_new[k] = v
                    state_dict = state_dict_new
                    torch.save(state_dict, args.pretrained_bin)
                    raise ValueError("Saved before?")
                else:
                    pass

                vision_model, _ = get_multimodal_utils(args=args)
                model = VILBertForVLTasks.from_pretrained(
                    args.pretrained_bin,
                    config=config,
                    num_labels=num_labels,
                    default_gpu=default_gpu,
                    vision_model=vision_model,
                )

            else:
                logging.info("No need to extracting new models!")

        elif args.multimodal_model_type == "clip":
            sys.path.insert(0, "models/CLIP/src")
            sys.path.insert(0, "models/CLIP/clip")
            from models.CLIP.src.lxrt.modeling import LXRTModel
            
            cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
            args.cls_id = cls_id
            sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            args.sep_id = sep_id

            if args.wrapper_model_type is None:
                num_labels_clip = 2
            else:
                num_labels_clip = None

            model = LXRTModel.from_pretrained(
                args.model_name_or_path,
                multimodal_text_part=args.multimodal_text_part,
                multimodal_img_part=args.multimodal_img_part,
                cls_id=args.cls_id,
                sep_id=args.sep_id,
                max_story_length=args.max_story_length,
                hl_include_objectives=args.hl_include_objectives,
                mlm_ignore_index=args.mlm_ignore_index,
                clip_model_name=args.clip_model_name,
                num_labels=num_labels_clip,
            )
            # print(model)
            # raise

            print()
            if args.clip_visual_model_weights is not None:
                visual_model_weights = torch.load(args.clip_visual_model_weights)
                visual_weights_dict = {}
                for k, v in visual_model_weights.items():
                    if "visual" in k:
                        k_names = k.split(".")
                        if k_names[0] != "encoder":
                            k_name = ".".join(k_names[1:])
                        else:
                            k_name = k
                        assert k_name in model.state_dict()
                        visual_weights_dict[k_name] = v
                model.load_state_dict(visual_weights_dict, strict=False)
                logger.warning("Loading clip visual weights from: {}".format(
                    args.clip_visual_model_weights))
                # for k, v in model.state_dict().items():
                #     print(k)
            # print(torch.sum(model.encoder.visual_model.visual.layer4[0].bn2.weight))
            # raise

        else:
            raise NotImplementedError("Multimodal model type: "
                "{} not done yet!".format(args.multimodal_model_type))

        if args.vision_model_checkpoint is not None:
            vision_model, img_transform_func = get_multimodal_utils(args=args)
            args.img_transform_func = img_transform_func
            ckpt = torch.load(args.vision_model_checkpoint)
            vision_parts = {}
            for k, v in ckpt.items():
                if "vision_model" in k:
                    k_new = k.partition(
                        "{}.".format("vision_model"))[-1]
                    vision_parts[k_new] = v
            vision_model.load_state_dict(vision_parts, strict=False)
            vision_model.fc = torch.nn.Identity()
            model.vision_model = vision_model

        # vision_model, img_transform_func = get_multimodal_utils(args=args)
        # model.vision_model = vision_model

    else:
        if args.wrapper_model_type is not None:
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )

    """
    model_state_dict = model.state_dict()
    for k, v in model_state_dict.items():
        print("model:", k)
    ckpt_state_dict = torch.load("pretrained_models/roberta/large/pytorch_model.bin")
    print()
    print()
    for k, v in ckpt_state_dict.items():
        print("ckpt:", k)
    print(model)
    raise
    """

    if "pure_decode" in args.task_type:
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
        config_decoder = BertConfig()
        config_decoder.vocab_size = args.max_story_length + 2
        config_decoder.hidden_size = config.hidden_size
        config_decoder.num_attention_heads = config.num_attention_heads
        config_decoder.num_hidden_layers = 1
        decoder = BertForMaskedLM(config=config_decoder)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        
    if args.model_type in ["roberta", "bert"]:
        pass
    else:
        raise NotImplementedError("Currently not"
                                  " support model {}".format(args.model_type))

    # Resizes the token type embeddings according to the task.
    if "pairwise" in args.task_type and args.replace_token_type_embeddings:
        model.config.type_vocab_size = 2
        model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(
            2, model.config.hidden_size)
        logger.info("Resizing dim of token_type_embeddings to {}".format(2))
    elif (("head" in args.task_type
           or "pure_class" in args.task_type)
          and args.replace_token_type_embeddings):
        model.config.type_vocab_size = args.max_story_length
        model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(
            args.max_story_length, model.config.hidden_size)
        logger.info("Resizing dim of token_type_embeddings to {}".format(
            args.max_story_length))
    elif ("pure_decode" in args.task_type
          and args.replace_token_type_embeddings):
        model.encoder.config.type_vocab_size = args.max_story_length
        model.encoder.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(
            args.max_story_length, model.config.hidden_size)
        logger.info("Resizing dim of token_type_embeddings to {}".format(
            args.max_story_length))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier() 

    # model.to(args.device)

    model.config.wrapper_model_type = args.wrapper_model_type
    if args.wrapper_model_type is not None:
        inner_model = model
        from models.berson import BertForOrdering, beam_search_pointer
        from models.berson import BertConfig as BersonConfig
        berson_config = BersonConfig.from_pretrained(args.model_name_or_path,
            num_labels=1, finetuning_task=args.task_names[0])
        args.ff_size = 3072
        args.heads = 8
        args.para_dropout = 0.1
        args.inter_layers = 2
        args.beam_size = 16
        args.pairwise_loss_lam = 0.6
        berson_config.wrapper_model_with_heatmap = args.wrapper_model_with_heatmap
        berson_config.hierarchical_version = args.hierarchical_version
        berson_config.hl_include_objectives = args.hl_include_objectives
        berson_config.multimodal_loss = args.multimodal_loss
        berson_config.v_feature_size = 1024  # config.v_feature_size
        if "berson" not in args.model_name_or_path:
            new_model = BertForOrdering(config=berson_config, args=args,
                                        inner_model=None, tokenizer=tokenizer)
            new_model.bert = model
            new_model.tokenizer = tokenizer
            model = new_model
        else:
            model = BertForOrdering.from_pretrained(args.model_name_or_path,
                                                    inner_model=model,
                                                    tokenizer=tokenizer,
                                                    config=berson_config,
                                                    load_inner_model=True,
                                                    args=args)
            criterion = torch.nn.NLLLoss(reduction='none')
            model.equip(criterion)

    model.to(args.device)
    # print(model)
    # VisualBERT
    # print(torch.sum(model.model.bert.encoder.layer[0].attention.self.query.weight))
    # print(torch.sum(model.model.classifier.dense.weight))
    # print(torch.sum(model.bert.vision_model.conv1.weight));raise
    # RoBERTa
    # print(torch.sum(model.roberta.encoder.layer[0].attention.self.query.weight))
    # print(torch.sum(model.classifier.dense.weight))
    # raise
    # BERSON with RoBERTa & VisualBERT
    # print(torch.sum(model.bert.model.bert.encoder.layer[0].attention.self.query.weight))
    # print(torch.sum(model.two_level_encoder.linear_in_2.weight))
    # BERSON with ViL-BERT
    # print(torch.sum(model.bert.bert.encoder.layer[0].attention.self.query.weight))
    # print(torch.sum(model.two_level_encoder.linear_in_2.weight))
    # raise

    # Adds num labels.
    config.num_labels = num_labels

    logger.info("Training/evaluation parameters %s", args)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("!!! Number of Params: {} M".format(count_parameters(model)/float(1000000)))

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_names,
                                                tokenizer, data_split=args.train_split,
                                                evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or
                          torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using
        # `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with
        # the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     args.output_dir)
        """
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        if args.multimodal:
            vision_model, img_transform_func = get_multimodal_utils(args=None)
            # TODO: Change the followings.
            if args.multimodal_model_type in ["naive", "naive_model"]:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.output_dir,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir,
                )
                from models.naive_model import NaiveMultimodalModel
                model = NaiveMultimodalModel(args=args,
                                             language_model=model,
                                             vision_model=vision_model)
            elif args.multimodal_model_type == "vilbert":
                from models.vilbert.vilbert import VILBertForVLTasks
                from models.vilbert.vilbert import BertConfig
                args.config_file = os.path.join(args.model_name_or_path,
                                                "config.json")
                config = BertConfig.from_json_file(args.config_file)
                if "roberta" in args.model_name_or_path:
                    config.model = "roberta"
                    config.model_type = "roberta"
                config.v_target_size = 1601
                config.task_specific_tokens = False
                config.dynamic_attention = False
                config.v_feature_size = 512
                default_gpu = False
                if args.local_rank in [-1, 0]:
                    default_gpu = True
                args.pretrained_bin = os.path.join(args.model_name_or_path,
                                                   "pytorch_model.bin")
                model = VILBertForVLTasks.from_pretrained(
                    args.pretrained_bin,
                    config=config,
                    num_labels=num_labels,
                    default_gpu=default_gpu,
                    output_loading_info=False,
                    vision_model=vision_model,
                )
            else:
                raise NotImplementedError("Multimodal model type: "
                    "{} not done yet!".format(args.multimodal_model_type))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )

        if "pure_decode" in args.task_type:
            encoder = AutoModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )
            config_decoder = BertConfig()
            config_decoder.vocab_size = args.max_story_length + 2
            config_decoder.hidden_size = config.hidden_size
            config_decoder.num_attention_heads = config.num_attention_heads
            config_decoder.num_hidden_layers = 1
            decoder = BertForMaskedLM(config=config_decoder)
            model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        model.to(args.device)
        """

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(
                    args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
        else:
            assert args.iters_to_eval is not None, ("At least one"
                " of `iter_to_eval` or `eval_all_checkpoints` should be set.")
            checkpoints = []
            for iter_to_eval in args.iters_to_eval:
                checkpoints_curr = list(
                    os.path.dirname(c) for c in sorted(glob.glob(
                        args.output_dir + "/*-{}/".format(iter_to_eval)
                        + WEIGHTS_NAME, recursive=True))
                )
                checkpoints += checkpoints_curr

        logger.info("\n\nEvaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            logger.info("\n\nEvaluate checkpoint: %s", checkpoint)
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            if args.wrapper_model_type == "berson":
                model = BertForOrdering.from_pretrained(checkpoint,
                                                        inner_model=inner_model,
                                                        tokenizer=tokenizer,
                                                        config=berson_config,
                                                        load_inner_model=True,
                                                        args=args)
                criterion = torch.nn.NLLLoss(reduction='none')
                model.equip(criterion)
            elif args.multimodal:
                print("Loading models")
                model_to_load = model.module if hasattr(model, 'module') else model
                bin_path = os.path.join(checkpoint, "pytorch_model.bin")
                model_to_load.load_state_dict(torch.load(bin_path))
                print("Done Loading models")
            elif "pure_decode" in args.task_type:
                model = EncoderDecoderModel.from_pretrained(checkpoint)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

            model.to(args.device)
            if args.wrapper_model_type == "berson":
                for eval_split in args.eval_splits:
                    result = berson_evaluate(args, model,
                        load_and_cache_examples, tokenizer,
                        prefix="", data_split=eval_split,
                        human_evaluate=False)
                    results.update(result)
            else:
                result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v)
                           for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
