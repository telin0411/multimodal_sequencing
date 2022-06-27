import argparse
import glob
import logging
import os
import random

import itertools
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
from datasets.processors import data_processors as processors
from trainers.metrics import compute_metrics as compute_metrics_func
from .modeling_bert import berson_pointer_network
from .process_inputs_for_berson import prepare_berson_inputs
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def pairs_generator(lenth):
    '''Generate the combinations of sentence pairs

        Args:
            lenth (int): the length of the sentence
        
        Returns:
            combs (list) : all combination of the index pairs in the passage
            num_combs (int) : the total number of all combs
    '''
    indices = list(range(lenth))
    combs_one_side = list(itertools.combinations(indices, 2))
    combs_one_side = [[x1, x2] for x1,x2 in combs_one_side]
    combs_other_side = [[x2, x1] for x1,x2 in combs_one_side]
    combs = combs_one_side + combs_other_side
    return combs, len(combs)


def cosine_knn(inp, data):
    dist = []
    inp = inp[0]
    for i in range(len(data)):
        datum = data[i]
        cos_dist = distance.cosine(inp, datum)
        dist.append(cos_dist)
    dist = np.asarray(dist)
    sorted_dist, sorted_indices = np.sort(dist), np.argsort(dist)
    return sorted_dist, sorted_indices


def retrieval_evaluate(args, model, load_and_cache_examples, tokenizer,
                       prefix="", data_split="test", human_evaluate=False):

    eval_task_names = args.task_names
    eval_outputs_dirs = [args.output_dir] * len(args.task_names)

    cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    results = {}

    ret_dict = {}
    model.eval()

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # Deal with candidates
        # eval_dataset = load_and_cache_examples(args, [eval_task],
        #                                        tokenizer, evaluate=True,
        #                                        data_split=data_split)
        # all_candidates = eval_dataset.candidates_list()
        include_img = True
        with_berson = False
        scramble = False
        encode_each = False
        data_split_cand = data_split
        # data_split_cand = "test-human_annot"
        cand_dataset = load_and_cache_examples(args, [eval_task.replace("retrieve", "pretrain")],
                                               tokenizer, evaluate=True,
                                               data_split=data_split_cand,
                                               get_guid=True, scramble=scramble)
        # logger.info("Number of candidates: {}".format(len(all_candidates)))

        # Encode all the candidates
        cand_reprs = []
        cand_ids = []
        cand_max = 100000

        if args.multimodal:
            cand_save_file = "data/recipeQA/retrieval_cand_{}_pretrain.npy".format(data_split_cand)
            cand_save_file = "data/recipeQA/retrieval_cand_{}.npy".format(data_split_cand)
            # cand_save_file = "data/wikihow/retrieval_cand_{}_pretrain_with_img_scrambled.npy".format(data_split_cand)
            cand_save_file = "data/wikihow/retrieval_cand_{}_pretrain.npy".format(data_split_cand)
            if include_img:
                cand_save_file = "data/wikihow/retrieval_cand_{}_berson_with_img.npy".format(data_split_cand)
                cand_save_file = "data/wikihow/retrieval_cand_{}_berson_with_img.npy".format(data_split_cand)
                # cand_save_file = "data/wikihow/retrieval_cand_{}_with_img_scrambled.npy".format(data_split_cand)
                cand_save_file = "data/wikihow/retrieval_cand_{}_original_pretrain_with_img.npy".format(data_split_cand)
                cand_save_file = "data/wikihow/retrieval_cand_{}_original_with_img.npy".format(data_split_cand)
                cand_save_file = "data/recipeQA/retrieval_cand_{}_with_img.npy".format(data_split_cand)
                cand_save_file = "data/wikihow/retrieval_cand_{}_pretrain_with_img.npy".format(data_split_cand)
                cand_save_file = "data/wikihow/retrieval_cand_{}_with_img.npy".format(data_split_cand)
        else:
            cand_save_file = "data/recipeQA/retrieval_cand_{}_pretrain_text_only.npy".format(data_split_cand)
            cand_save_file = "data/wikihow/retrieval_cand_{}_pretrain_text_only.npy".format(data_split_cand)
            cand_save_file = "data/recipeQA/retrieval_cand_{}_text_only.npy".format(data_split_cand)
            cand_save_file = "data/wikihow/retrieval_cand_{}_text_only_berson.npy".format(data_split_cand)
            cand_save_file = "data/wikihow/retrieval_cand_{}_text_only.npy".format(data_split_cand)

        if os.path.exists(cand_save_file) and encode_each:
            cand_reprs_np = np.load(cand_save_file)
            logger.info("Loading nn file from: {}".format(cand_save_file))
            for cand in tqdm(all_candidates, desc="Evaluating"):
                # FIXME: fix below.
                if len(cand_reprs) >= cand_max:
                    break
                cand_ids.append(cand[2])
        else:
            """
            if True:
                for cand in tqdm(all_candidates, desc="Evaluating"):
                    # FIXME: fix below.
                    if len(cand_reprs) >= cand_max:
                        break
                    cand_inputs = {
                        "input_ids": torch.Tensor(cand[0]).unsqueeze(0).long(),
                        "attention_mask": torch.Tensor(cand[1]).unsqueeze(0),
                    }
                    if args.multimodal:
                        cand_inputs["images"] = cand[-1].unsqueeze(0)
                    cand_inputs = {x: cand_inputs[x].to(args.device) for x in cand_inputs}
                    cand_inputs["token_type_ids"] = None
                    if args.wrapper_model_type is not None:
                        cand_outpus = model.bert(cand_inputs)
                        cand_sequence_outputs = cand_outpus[0]
                        cand_sequence_outputs = cand_sequence_outputs.detach()
                        cand_pooled_outputs = cand_sequence_outputs[:, 0]
                        # cand_pooled_outputs = torch.mean(cand_sequence_outputs, dim=1)
                    else:
                        if args.multimodal:
                            cand_inputs["labels"] = None
                            model.config.wrapper_model_type = "berson"
                            cand_outpus = model(cand_inputs)
                            cand_sequence_outputs = cand_outpus[0]
                            cand_sequence_outputs = cand_sequence_outputs.detach()
                            cand_pooled_outputs = cand_sequence_outputs[:, 0]
                        else:
                            doc_outpus = model(**cand_inputs)
                    cand_pooled_outputs = cand_pooled_outputs[0]
                    if args.multimodal and include_img:
                        txt_len = cand_inputs["input_ids"].size(1)
                        cand_image_outputs = cand_sequence_outputs[:, txt_len:]
                        cand_pooled_outputs = torch.cat([cand_pooled_outputs, cand_image_outputs[0, 0, :]], dim=0)
                    cand_ids.append(cand[2])
                    cand_reprs.append(cand_pooled_outputs)
            """
            if True:
                if os.path.exists(cand_save_file):
                    cand_reprs_np = np.load(cand_save_file)
                    logger.info("Loading nn file from: {}".format(cand_save_file))
                all_candidates = []
                cand_sampler = SequentialSampler(cand_dataset)
                cand_dataloader = DataLoader(cand_dataset, sampler=cand_sampler,
                                             batch_size=1)
                for batch in tqdm(cand_dataloader, desc="Candidates"):
                    inputs = {
                        "input_ids": batch[0][0], "attention_mask": batch[1],
                        # "labels": batch[3]
                    }
                    if args.multimodal:
                        inputs["images"] = batch[-1]

                    cls_pos = torch.nonzero(inputs["input_ids"][0]==cls_id).t()[0]
                    sep_pos = torch.nonzero(inputs["input_ids"][0]==sep_id).t()[0]
                    if not os.path.exists(cand_save_file):
                        inputs = {x: inputs[x].to(args.device) for x in inputs}
                        inputs["token_type_ids"] = None
                        if args.wrapper_model_type is not None:
                            if False:
                                inputs["labels"] = torch.Tensor([list(range(
                                    args.max_story_length))]).long().to(args.device)
                                berson_inputs = prepare_berson_inputs(inputs, tokenizer, args)
                                berson_encoded_outputs = model.encode(**berson_inputs)
                                doc_mat = berson_encoded_outputs[0].detach()
                                cls_pooled_output = berson_encoded_outputs[4].detach()
                                doc_sequence_outputs = doc_mat
                            else:
                                if with_berson:
                                    pairs_list = pairs_generator(args.max_story_length)[0]
                                    inputs["labels"] = torch.Tensor([list(range(
                                        args.max_story_length))]).long().to(args.device)
                                    berson_inputs = prepare_berson_inputs(inputs, tokenizer, args)
                                    berson_encoded_outputs = model.encode(**berson_inputs)
                                    berson_doc_mat = berson_encoded_outputs[0].detach()
                                    # cls_pooled_output = berson_encoded_outputs[4].detach()
                                if args.multimodal:
                                    inputs["labels"] = None
                                    doc_outpus = model.bert(inputs)
                                else:
                                    doc_outpus = model.bert(**inputs)
                                doc_sequence_outputs = doc_outpus[0]
                                doc_sequence_outputs = doc_sequence_outputs.detach()
                                if args.multimodal and include_img:
                                    txt_len = inputs["input_ids"].size(1)
                                    doc_image_outputs = doc_sequence_outputs[:, txt_len:]
                                doc_sequence_outputs = doc_sequence_outputs[:, cls_pos]
                        else:
                            if args.multimodal:
                                inputs["labels"] = None
                                model.config.wrapper_model_type = "berson"
                                doc_outpus = model(inputs)
                                doc_sequence_outputs = doc_outpus[0]
                                doc_sequence_outputs = doc_sequence_outputs.detach()
                                if args.multimodal and include_img:
                                    txt_len = inputs["input_ids"].size(1)
                                    doc_image_outputs = doc_sequence_outputs[:, txt_len:]
                                doc_sequence_outputs = doc_sequence_outputs[:, cls_pos]
                            else:
                                doc_outpus = model(**inputs)
                    guid = batch[0][1][0]
                    for k in range(5):
                        if scramble:
                            idx_seq = batch[0][2][0]
                            k = idx_seq[k]
                        guid_k = guid + "###{}".format(k)
                        if not os.path.exists(cand_save_file):
                            repr_k = doc_sequence_outputs[:, k][0]
                            if with_berson:
                                # pair_repr_k = []
                                # for pair_idx, (u, v) in enumerate(pairs_list):
                                #     if u == k or v == k:
                                #         pair_repr_k.append(cls_pooled_output[pair_idx])
                                # pair_repr_k = torch.stack(pair_repr_k)
                                # pair_repr_k = torch.mean(pair_repr_k, dim=0)
                                pair_repr_k = berson_doc_mat[0, k]
                                repr_k = torch.cat([repr_k, pair_repr_k], dim=0)
                            if args.multimodal and include_img:
                                repr_k = torch.cat([repr_k, doc_image_outputs[0, k, :]], dim=0)
                            cand_reprs.append(repr_k)
                        cand_ids.append(guid_k)
                        input_ids_k = inputs["input_ids"][0, cls_pos[k]:sep_pos[k]+1].cpu().numpy()
                        attention_mask_k = inputs["attention_mask"][0, cls_pos[k]:sep_pos[k]+1].cpu().numpy()
                        if args.multimodal:
                            imgs_k = inputs["images"][:, k].cpu()
                            all_candidates.append((input_ids_k, attention_mask_k, imgs_k))
                        else:
                            all_candidates.append((input_ids_k, attention_mask_k))
            if not os.path.exists(cand_save_file):
                cand_reprs = torch.stack(cand_reprs)
                print(cand_reprs.size())
                cand_reprs_np = cand_reprs.cpu().numpy()
                np.save(cand_save_file, cand_reprs_np)
                logger.info("Saving retrieval candidates to: {}".format(cand_save_file))
            print(cand_reprs_np.shape, len(cand_ids))

        # KNN
        nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute').fit(cand_reprs_np)

        eval_dataset = load_and_cache_examples(args, [eval_task],
                                               tokenizer, evaluate=True,
                                               data_split=data_split)

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
        nn_get_acc = 0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        best_acc = []
        truth = []
        predicted = []

        f = open(os.path.join(args.output_dir, "output_order.txt"), 'w')

        batch_idx = 0
        mrr = []
        mrr_and_correct_pred = []
        mrr_correct_order_cnt = 0
        mrr_correct_order = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if batch_idx >= cand_max:
                break
            # batch = tuple(t.to(args.device) for t in batch)

            tru = batch[3].view(-1).tolist()
            truth.append(tru)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0], "attention_mask": batch[1],
                    # "labels": batch[3]
                }

                if args.multimodal:
                    inputs["images"] = batch[-1]

                # Perform some NN search
                if args.wrapper_model_type is not None:
                    if False:
                        args.max_story_length -= 1
                        inputs["labels"] = batch[3]
                        berson_inputs = prepare_berson_inputs(inputs, tokenizer, args)
                        berson_encoded_outputs = model.encode(**berson_inputs)
                        doc_mat = berson_encoded_outputs[0]
                        cls_pooled_output = berson_encoded_outputs[4]
                        # pooled_doc_mat = doc_mat[:, 0]
                        pooled_doc_mat = torch.mean(doc_mat, dim=1)
                        # pooled_doc_mat = torch.mean(cls_pooled_output, 0).unsqueeze(0)
                        args.max_story_length += 1
                    elif not encode_each:
                        inputs = {x: inputs[x].to(args.device) for x in inputs}
                        inputs["token_type_ids"] = None
                        if args.multimodal:
                            inputs["labels"] = None
                            doc_outpus = model.bert(inputs)
                        else:
                            doc_outpus = model.bert(**inputs)
                        doc_sequence_outputs = doc_outpus[0]
                        doc_sequence_outputs = doc_sequence_outputs.detach()
                        if args.multimodal and include_img:
                            txt_len = inputs["input_ids"].size(1)
                            doc_image_outputs = doc_sequence_outputs[:, txt_len:]
                        cls_pos = torch.nonzero(inputs["input_ids"][0]==cls_id).t()[0]
                        # sep_pos = torch.nonzero(inputs["input_ids"][0]==sep_id).t()[0]
                        doc_sequence_outputs = doc_sequence_outputs[:, cls_pos]
                        # pooled_doc_mat = doc_sequence_outputs[:, 0]
                        # doc_sequence_outputs = doc_sequence_outputs[:, 0:sep_pos[-1]+1]
                        pooled_doc_mat = torch.mean(doc_sequence_outputs, dim=1)
                        if with_berson:
                            args.max_story_length -= 1
                            inputs["labels"] = batch[3]
                            berson_inputs = prepare_berson_inputs(inputs, tokenizer, args)
                            berson_encoded_outputs = model.encode(**berson_inputs)
                            # cls_pooled_output = berson_encoded_outputs[4].detach()
                            # pooled_cls_pooled_output = torch.mean(cls_pooled_output, 0).unsqueeze(0)
                            berson_doc_mat = berson_encoded_outputs[0]
                            pooled_cls_pooled_output = torch.mean(berson_doc_mat, dim=1)
                            pooled_doc_mat = torch.cat([pooled_doc_mat, pooled_cls_pooled_output], dim=-1)
                            args.max_story_length += 1
                        if args.multimodal and include_img:
                            pooled_image_mat = torch.mean(doc_image_outputs, dim=1)
                            pooled_doc_mat = torch.cat([pooled_doc_mat, pooled_image_mat], dim=-1)
                    else:
                        cls_pos = torch.nonzero(inputs["input_ids"][0]==cls_id).t()[0]
                        sep_pos = torch.nonzero(inputs["input_ids"][0]==sep_id).t()[0]
                        doc_mat = []
                        for c in range(len(cls_pos)):
                            curr_input_ids = inputs["input_ids"][:, cls_pos[c]:sep_pos[c]+1]
                            curr_attention_mask = inputs["input_ids"][:, cls_pos[c]:sep_pos[c]+1]
                            curr_inputs = {
                                "input_ids": curr_input_ids,
                                "attention_mask": curr_attention_mask
                            }
                            if args.multimodal:
                                curr_inputs["images"] = inputs["images"][:, c].unsqueeze(1)
                            curr_inputs = {x: curr_inputs[x].to(args.device) for x in curr_inputs}
                            curr_inputs["token_type_ids"] = None
                            if args.multimodal:
                                curr_inputs["labels"] = None
                                curr_doc_outputs = model.bert(curr_inputs)[0]
                            else:
                                curr_doc_outputs = model.bert(**curr_inputs)[0]
                            pooled_doc_outputs = curr_doc_outputs[:, 0].detach()
                            if args.multimodal and include_img:
                                txt_len = curr_inputs["input_ids"].size(1)
                                doc_image_outputs = curr_doc_outputs[:, txt_len:]
                                doc_image_outputs = doc_image_outputs[:, 0]
                                pooled_doc_outputs = torch.cat([pooled_doc_outputs, doc_image_outputs], dim=-1)
                            doc_mat.append(pooled_doc_outputs)
                        doc_mat = torch.stack(doc_mat, dim=1)
                        pooled_doc_mat = torch.mean(doc_mat, dim=1)
                else:
                    if encode_each:
                        model.config.wrapper_model_type = "berson"
                        cls_pos = torch.nonzero(inputs["input_ids"][0]==cls_id).t()[0]
                        sep_pos = torch.nonzero(inputs["input_ids"][0]==sep_id).t()[0]
                        doc_mat = []
                        for c in range(len(cls_pos)):
                            curr_input_ids = inputs["input_ids"][:, cls_pos[c]:sep_pos[c]+1]
                            curr_attention_mask = inputs["input_ids"][:, cls_pos[c]:sep_pos[c]+1]
                            curr_inputs = {
                                "input_ids": curr_input_ids,
                                "attention_mask": curr_attention_mask
                            }
                            if args.multimodal:
                                curr_inputs["images"] = inputs["images"][:, c].unsqueeze(1)
                            curr_inputs = {x: curr_inputs[x].to(args.device) for x in curr_inputs}
                            curr_inputs["token_type_ids"] = None
                            if args.multimodal:
                                curr_inputs["labels"] = None
                                curr_doc_outputs = model(curr_inputs)[0]
                            else:
                                curr_doc_outputs = model(**curr_inputs)[0]
                            pooled_doc_outputs = curr_doc_outputs[:, 0].detach()
                            if args.multimodal and include_img:
                                txt_len = curr_inputs["input_ids"].size(1)
                                doc_image_outputs = curr_doc_outputs[:, txt_len:]
                                doc_image_outputs = doc_image_outputs[:, 0]
                                pooled_doc_outputs = torch.cat([pooled_doc_outputs, doc_image_outputs], dim=-1)
                            doc_mat.append(pooled_doc_outputs)
                        doc_mat = torch.stack(doc_mat, dim=1)
                        pooled_doc_mat = torch.mean(doc_mat, dim=1)
                    else:
                        model.config.wrapper_model_type = "berson"
                        inputs = {x: inputs[x].to(args.device) for x in inputs}
                        inputs["token_type_ids"] = None
                        if args.multimodal:
                            inputs["labels"] = None
                            doc_outpus = model(inputs)
                        else:
                            doc_outpus = model(**inputs)
                        doc_sequence_outputs = doc_outpus[0]
                        doc_sequence_outputs_org = doc_sequence_outputs
                        cls_pos = torch.nonzero(inputs["input_ids"][0]==cls_id).t()[0]
                        doc_sequence_outputs = doc_sequence_outputs[:, cls_pos]
                        # pooled_doc_mat = doc_sequence_outputs[:, 0]
                        pooled_doc_mat = torch.mean(doc_sequence_outputs, dim=1)
                        # raise NotImplementedError
                        if args.multimodal and include_img:
                            txt_len = inputs["input_ids"].size(1)
                            doc_image_outputs = doc_sequence_outputs_org[:, txt_len:]
                            pooled_image_mat = torch.mean(doc_image_outputs, dim=1)
                            pooled_doc_mat = torch.cat([pooled_doc_mat, pooled_image_mat], dim=-1)
                pooled_doc_mat = pooled_doc_mat.detach().cpu().numpy()
                
                nn_distances, nn_indices = nbrs.kneighbors(pooled_doc_mat)
                nn_distances, nn_indices = nn_distances[0], nn_indices[0]
                # nn_distances, nn_indices = cosine_knn(pooled_doc_mat, cand_reprs_np)

                story_miss = batch[-2][0]
                if story_miss not in ret_dict:
                    ret_dict[story_miss] = []

                # print(nn_distances, nn_indices)
                gt_idx = cand_ids.index(story_miss)
                gt_dist = np.linalg.norm(pooled_doc_mat - cand_reprs_np[gt_idx])
                # print("gt_dist: {:.3f}".format(gt_dist))

                if gt_idx in nn_indices:
                    rank_curr = nn_indices.tolist().index(gt_idx)
                else:
                    rank_curr = 1e8
                mrr.append(rank_curr)
                # print(rank_curr)
                sort = False
                # if rank_curr == 0:
                if rank_curr < 100:
                    sort = True
                    nn_indices = [gt_idx]
                if rank_curr == 0:
                    mrr_correct_order_cnt += 1

                nn_get_correct = False
                # print(nn_get_acc)

                # Perform concatenations
                """
                max_kept = 10
                # nn_distances, nn_indices = [0], [rank_curr]
                for nn_idx in range(len(nn_indices)):
                    if len(ret_dict[story_miss]) >= max_kept or not sort:
                        break
                    nn_distance, nn_index = nn_distances[nn_idx], nn_indices[nn_idx]
                    story_cand = cand_ids[nn_index]
                    # print(story_miss, story_cand)
                    nn_get = all_candidates[nn_index]
                    input_ids_miss = batch[0]
                    attention_mask_miss = batch[1]
                    input_ids_cand = torch.Tensor(nn_get[0]).long().unsqueeze(0)
                    attention_mask_cand = torch.Tensor(nn_get[1]).unsqueeze(0)
                    non_pad_entries = torch.nonzero(input_ids_miss[0]-pad_id).t()[0]
                    input_ids_miss = input_ids_miss[:, non_pad_entries]
                    attention_mask_miss = attention_mask_miss[:, non_pad_entries]
                    input_ids_nn = torch.cat([input_ids_miss, input_ids_cand], dim=-1)
                    attention_mask_nn = torch.cat([attention_mask_miss, attention_mask_cand], dim=-1)

                    inputs = {
                        "input_ids": input_ids_nn,
                        "attention_mask": attention_mask_nn,
                        "labels": batch[3],
                    }

                    if args.multimodal:
                        imgs_miss = batch[-1]
                        imgs_cand = nn_get[-1]
                        imgs_cand = imgs_cand.unsqueeze(1)
                        imgs_nn = torch.cat([imgs_miss, imgs_cand], dim=1)
                        inputs["images"] = imgs_nn

                    pred = berson_pointer_network(args, model, tokenizer, inputs)

                    curr_ret = {
                        "story_cand": story_cand,
                        "story_pred": pred
                    }
                    if True:
                        label = batch[3].cpu().numpy()[0]
                        if pred == label.tolist():
                            # print(pred, label, story_miss, story_cand)
                            ret_dict[story_miss].append(curr_ret)
                    else:
                        ret_dict[story_miss].append(curr_ret)
                    if story_miss == story_cand:
                        nn_get_correct = True
                    if pred == label.tolist() and rank_curr == 0:
                        mrr_correct_order += 1
                    if pred == label.tolist():
                        mrr_and_correct_pred.append((rank_curr + 1) * 1)
                    else:
                        mrr_and_correct_pred.append((rank_curr + 1) * 1e10)
                pass
                if nn_get_correct:
                    nn_get_acc += 1
                """

                # predicted.append(pred)
                # print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
                #       file=f)
            batch_idx += 1
            pass

        pass

        print(nn_get_acc)
        nn_get_acc /= float(len(eval_dataset))
        nn_get_acc *= 100.0
        logger.info("KNN Accuracy: {:.3f}%".format(nn_get_acc))

        """
        accs, pmr, taus = cal_result(truth, predicted, best_acc, f, args=args)

        results['acc_dev'] = accs
        results['pmr_dev'] = pmr
        results['taus_dev'] = taus

        output_eval_file = os.path.join(eval_output_dir,
                                        prefix, "eval_results_split_{}.txt".format(data_split))

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

        output_only_eval_file_1 = os.path.join(args.output_dir, "all_eval_results.txt")
        fh = open(output_only_eval_file_1, 'a')
        fh.write(prefix)
        for key in sorted(results.keys()):
            fh.write("%s = %s\n" % (key, str(results[key])))
        fh.close()
        """
        import pickle
        output_ret_file = os.path.join(eval_output_dir,
            prefix, "retrieval_results_split_{}_correct_predicted_orders.pickle".format(data_split))
        with open(output_ret_file, "wb") as handle:
            pickle.dump(ret_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saving retrieval results to: {}".format(output_ret_file))
        
        print(mrr)
        top_1 = np.asarray(mrr)
        top1_acc = len(top_1[top_1==0]) / float(len(mrr))
        mrr = np.asarray([1/float(x+1) for x in mrr])
        mrr = np.mean(mrr)
        print("MRR: {:.4f}".format(mrr))
        mrr_correct_order_acc = mrr_correct_order / float(mrr_correct_order_cnt)
        print("MRR correct order: {}/{} = {:.4f}%".format(mrr_correct_order, mrr_correct_order_cnt, mrr_correct_order_acc))
        print("Top-1: {:.4f}".format(top1_acc))

        mrr_and_correct_pred = [1/float(x) for x in mrr_and_correct_pred]
        mrr_and_correct_pred = np.asarray(mrr_and_correct_pred)
        mean_mrr_and_correct_pred = np.mean(mrr_and_correct_pred)
        print("MRR and Correct: {:.5f}".format(mean_mrr_and_correct_pred))

    return results


def cal_result(truth, predicted, best_acc, f, args):
    right, total = 0, 0
    pmr_right = 0
    taus = []
    accs = []
    # pm
    pm_p, pm_r = [], []

    import itertools
    from sklearn.metrics import accuracy_score

    lcs_result_all = []
    min_dist_all = []
    
    to_compare = []
    idx = 0

    for t, p in zip(truth, predicted):
        # print ('t, p', t, p)
        if len(p) == 1:
            right += 1
            total += 1
            pmr_right += 1
            accs.append(1)
            taus.append(1)
            continue

        # print(t, p)
        eq = np.equal(t, p)
        # print ('eq', eq)
        right += eq.sum()
        # print ('acc', eq.sum(), eq.sum()/len(t))
        accs.append(eq.sum()/len(t))

        total += len(t)

        pmr_right += eq.all()

        # pm
        s_t = set([i for i in itertools.combinations(t, 2)])
        s_p = set([i for i in itertools.combinations(p, 2)])
        pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
        pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

        cn_2 = len(p) * (len(p) - 1) / 2
        pairs = len(s_p) - len(s_p.intersection(s_t))
        tau = 1 - 2 * pairs / cn_2
        # print ('tau', tau)

        taus.append(tau)

        to_compare.append((eq.sum()/len(t), eq.all(), idx, p, t))
        idx += 1

    acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                         list(itertools.chain.from_iterable(predicted)))

    best_acc.append(acc)

    pmr = pmr_right / len(truth)

    # print ('truth', truth[:10])
    # print ('predicted', predicted[:10])
    # print ('accs taus', accs, taus)

    taus = np.mean(taus)

    pm_p = np.mean(pm_p)
    pm_r = np.mean(pm_r)
    pm = 2 * pm_p * pm_r / (pm_p + pm_r)

    # print('acc:', acc)

    f.close()
    # acc = max(best_acc)
    accs = np.mean(accs)

    if "ref_json_file" in args.__dict__:
        if args.ref_json_file is not None:
            import csv
            import json
            ref_data = []
            json_f = open(args.ref_json_file, "r")
            for line in json_f:
                ref_data.append(json.loads(line.strip()))
            print(len(to_compare), len(ref_data))
            assert len(to_compare) == len(ref_data)
            to_compare = sorted(to_compare)

            csv_path = "{}_model_performance.csv".format(args.ref_json_file.split(".json")[0])
            fieldnames = ["partial_match", "exact_match", "index", "url", "prediction", "gt"]
            csv_f = csv.DictWriter(open(csv_path, "w"), fieldnames=fieldnames)
            csv_f.writeheader()

            for each in to_compare:
                acc_curr, pmr_curr, idx, pred, gt = each
                print(acc_curr, pmr_curr, idx, ref_data[idx]["url"], pred, gt)
                row = {}
                row["partial_match"] = acc_curr
                row["exact_match"] = pmr_curr
                row["index"] = idx
                row["url"] = ref_data[idx]["url"]
                row["prediction"] = pred
                row["gt"] = gt
                csv_f.writerow(row)

    if True:
        from trainers.metrics import compute_metrics
        metrics_list = ["partial_match", "exact_match", "lcs", "lcs_substr", "distance_based", "ms", "wms"]
        res = {}
        for metrics in metrics_list:
            perf = compute_metrics(args, metrics, predicted, truth)
            logger.info("Metric: {}  Perf: {:.3f}".format(metrics, perf))
            res[metrics] = perf
        
        headers = "& PM    & EM    & Lseq & Lstr & tau  & Dist."
        # content = "& 00.00    & 00.00    & 0.00     & 0.00     & 0.00     & 0.00 "
        content = "& {:03.2f} & {:03.2f} & {:03.2f} & {:03.2f} & {:03.2f} & {:03.2f}".format(
            res["partial_match"] * 100,
            res["exact_match"] * 100,
            res["lcs"],
            res["lcs_substr"],
            taus,
            res["distance_based"],
        )
        logger.info("***** Paper Results *****")
        logger.info(" {}".format(headers))
        logger.info(" {}".format(content))


    return accs, pmr, taus
