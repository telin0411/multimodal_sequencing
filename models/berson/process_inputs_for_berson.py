import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm
import itertools
import numpy as np

import torch


def prepare_berson_inputs(inputs, tokenizer, args=None):

    # bz = inputs["input_ids"].size(0)
    bz = len(inputs["input_ids"])

    # for k in inputs:
    #     if "images" in k:
    #         continue
    #     print(k, inputs[k])

    encoded_inputs = []

    for i in range(bz):
        ids_i = inputs["input_ids"][i]
        gt_i = inputs["labels"][i]
        pair_ids_i = None
        max_length_i = args.per_seq_max_length
        add_special_tokens_i = True
        stride_i = 0
        truncation_strategy_i = "longest_first"
        return_tensors_i = None
        encoded_inputs_i = prepare_single_instance(
            ids=ids_i, pair_ids=pair_ids_i, max_length=max_length_i,
            add_special_tokens=add_special_tokens_i, stride=stride_i,
            truncation_strategy=truncation_strategy_i, ground_truth=gt_i,
            return_tensors=return_tensors_i, tokenizer=tokenizer, args=args)
        
        encoded_inputs.append(encoded_inputs_i)

    # return encoded_inputs

    # TODO preprocess_batch codes
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    batch = preprocess_batch(encoded_inputs, pad_id)
    berson_inputs = {
        'input_ids':          batch[0],
        'attention_mask':     batch[1],
        'token_type_ids':     batch[2],
        'pairs_list':         batch[3],
        'passage_length':     batch[4],
        "pairs_num":          batch[5],
        "sep_positions":      batch[6],
        "ground_truth":       batch[7],
        "mask_cls":           batch[8],
        "pairwise_labels":    batch[9],
        # "cuda":               args.device,
        "cuda":               "cuda:0",
    }

    # for k in berson_inputs:
    #     print(k, berson_inputs[k])
    # raise
    # print(berson_inputs["sep_positions"])
    # print(berson_inputs["input_ids"][0][0]);raise

    # Multimodal inputs.
    if "images" in inputs and inputs["images"] is not None:
        berson_images = process_images(inputs["images"],
            encoded_inputs=encoded_inputs)
        berson_inputs["images"] = berson_images

    for k in berson_inputs:
        if "cuda" in k:
            continue
        berson_inputs[k] = berson_inputs[k].to(args.device)

    return berson_inputs


def process_images(images, encoded_inputs):
    berson_images = []
    for i in range(len(encoded_inputs)):
        encoded_inputs_i = encoded_inputs[i]
        pairs_list = encoded_inputs_i["pairs_list"]
        berson_images_i = []
        for img_id1, img_id2 in pairs_list:
            img1 = images[i, img_id1, :, :, :]
            img2 = images[i, img_id2, :, :, :]
            paired_img = torch.stack([img1, img2])
            berson_images_i.append(paired_img)
        berson_images_i = torch.stack(berson_images_i)
        berson_images.append(berson_images_i)

    berson_images = torch.stack(berson_images)
    return berson_images


def parse_input_ids(input_ids, cls_id, sep_id):
    input_ids_list = []
    start_indices = torch.nonzero(input_ids==cls_id, as_tuple=False).squeeze()
    end_indices = torch.nonzero(input_ids==sep_id, as_tuple=False).squeeze()
    assert len(start_indices) == len(end_indices)
    for i in range(len(start_indices)):
        start_idx = start_indices[i]
        end_idx = end_indices[i]
        input_ids_i = input_ids[start_idx:end_idx+1]
        input_ids_list.append(input_ids_i)
    return input_ids_list


def prepare_single_instance(ids, ground_truth=None,
                            pair_ids=None, max_length=None,
                            add_special_tokens=False, stride=0,
                            truncation_strategy='longest_first',
                            return_tensors=None, tokenizer=None,
                            args=None, **kwargs):

    cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    # order = list(range(args.max_story_length)) # [0, 1, 2, 3, 4]
    # passage_length = len(order)  # 5
    passage_length = args.max_story_length

    # assert ground_truth is not None
    # ground_truth = list(range(5))

    pairs_list, pairs_num = pairs_generator(passage_length)

    ids_list = parse_input_ids(ids, cls_id, sep_id)
    # ids_list = ids
    # ids_list = ids.strip().split(" <eos> ")

    assert len(ids_list) == args.max_story_length

    gt_list = list(ground_truth.cpu().numpy())
    # gt_list = ground_truth

    # base_index = list(range(5))
    # shuffled_index = base_index
    # import random
    # random.shuffle(shuffled_index)
    # gt_list = list(np.argsort(shuffled_index))

    # print(pairs_list, pairs_num)

    # (tokenizer, pairs, shuffled_index, passage, p_key)
    input_ids = []
    masked_ids = []
    token_type_ids = []
    sep_positions = []
    pairwise_labels = []

    max_sample_len = 0 # save the max length of ids inorder to do padding

    for sent_id1, sent_id2 in pairs_list:
        sep_position = [0,0]

        ### pairwise label ###
        first_pos = gt_list.index(sent_id1)
        sec_pos = gt_list.index(sent_id2)

        # first_pos = shuffled_index[sent_id1]
        # sec_pos = shuffled_index[sent_id2]

        if first_pos > sec_pos:  # 3 1  label=0
            pairwise_label = 0

        if first_pos < sec_pos:
            pairwise_label = 1

        pairwise_labels.append(pairwise_label)

        # sent1 = ids_list[shuffled_index[sent_id1]]  # ids_list[sent_id1]
        # sent2 = ids_list[shuffled_index[sent_id2]]  # ids_list[sent_id2]
        sent1 = ids_list[sent_id1]
        sent2 = ids_list[sent_id2]

        """
        tokenized_sent1 = tokenizer.tokenize(sent1, **kwargs)
        tokenized_sent1 = tokenized_sent1[:args.per_seq_max_length]
        tokenized_sent2 = tokenizer.tokenize(sent2, **kwargs)
        tokenized_sent2 = tokenized_sent2[:args.per_seq_max_length]

        concat_sents = ['[CLS]'] + tokenized_sent1 + ['[SEP]'] 
        sep_position[0] = len(concat_sents) - 1
        token_type_id = [0] * len(concat_sents)
        concat_sents += tokenized_sent2 + ['[SEP]']   #  ['[CLS]', 'we', 'first', 'show', '[SEP]']
        input_id = tokenizer.convert_tokens_to_ids(concat_sents)  # [101, 2057, 2034, 102]

        sep_position[1] = len(concat_sents) - 1
        token_type_id += [1] * len(tokenized_sent2 + ['[SEP]'])
        masked_id = [1] * len(token_type_id)
        """
        sep_position[0] = len(sent1) - 1
        if args.multimodal_img_part:
            sep_position[0] = 0
        token_type_id = [0] * len(sent1)
        input_id = torch.cat([sent1, sent2], dim=0)
        sep_position[1] = len(input_id) - 1
        if args.multimodal_img_part:
            sep_position[1] = 1
        if cls_id == 0:  # cls_id = 0 is roberta-based
            token_type_id += [0] * len(sent2)
        else:
            token_type_id += [1] * len(sent2)
        masked_id = [1] * len(token_type_id)
        # """

        assert len(input_id) == len(token_type_id) == len(masked_id)

        # print(input_id, sep_position)

        # max pair length in this document
        max_sample_len = len(masked_id) if max_sample_len < len(masked_id) else max_sample_len

        input_id = input_id.tolist()

        input_ids.append(input_id)
        masked_ids.append(masked_id)
        token_type_ids.append(token_type_id)
        sep_positions.append(sep_position)

    shuffled_index = None
    ground_truth = gt_list

    encoded_inputs = {}
    encoded_inputs['input_ids'] = input_ids       # [[23, 39, 5, 4], [67, 21, 51, 6, 9]]  2 pairs in this document
    encoded_inputs['masked_ids'] = masked_ids     # [[1, 1, 1, 1], [1, 1, 1, 1, 1]]
    encoded_inputs['token_type_ids'] = token_type_ids  # [[0, 0, 1, 1], [0, 0, 1, 1, 1]]
    encoded_inputs['sep_positions'] = sep_positions    # [[1,3], [2,4]]
    encoded_inputs['shuffled_index'] = shuffled_index  # [1, 0]
    encoded_inputs['max_sample_len'] = max_sample_len  # 5
    encoded_inputs['ground_truth'] = ground_truth      # [0, 1]
    encoded_inputs['passage_length'] = passage_length  # 2
    encoded_inputs['pairs_num'] = pairs_num            # 2
    encoded_inputs['pairs_list'] = pairs_list          # [[0,1],[1,0]]
    encoded_inputs['pairwise_labels'] = pairwise_labels  # [1,0]  (pairs_num)
    encoded_inputs["sent"] = ids

    return encoded_inputs


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


def preprocess_batch(batch, pad_id):

    sen_num_dataset = []
    sample_len_dataset = []
    pairs_num_dataset = []

    for example in batch:
        sample_len_dataset.append(example['max_sample_len'])
        sen_num_dataset.append(example['passage_length'])
        pairs_num_dataset.append(example['pairs_num'])

    max_pair_len = max(sample_len_dataset)
    max_sen_num = max(sen_num_dataset)
    max_pairs_num = max(pairs_num_dataset)
    # print ('max_pair_len', max_pair_len, sample_len_dataset)

    all_input_ids=[]
    all_attention_mask=[]
    all_token_type_ids=[]
    all_pairs_list=[]
    all_passage_length=[]
    all_pairs_num=[]
    all_sep_positions=[]
    all_ground_truth=[]
    all_mask_cls=[]
    all_pairwise_labels = []

    for inputs in batch:  # padding for each example

        input_ids, masked_ids = inputs['input_ids'], inputs['masked_ids']
        # input_ids = [list(input_ids[i].cpu().numpy()) for i in range(len(input_ids))]
        token_type_ids, sep_positions = inputs['token_type_ids'], inputs['sep_positions']
        shuffled_index = inputs['shuffled_index']
        max_sample_len = inputs['max_sample_len']
        # ground_truth = list(inputs['ground_truth'].cpu().numpy())
        ground_truth = inputs['ground_truth']# .cpu().tolist()
        passage_length = inputs['passage_length']
        pairs_num, pairs_list = inputs['pairs_num'], inputs['pairs_list']
        pairwise_labels = inputs['pairwise_labels']  # (pairs_num) [0,0,1,1,1,0]  sentence_num = 3 pair_num=6 

        # print ('max_sample_len', max_sample_len)
        padd_num_sen = max_sen_num - passage_length
        padding_pair_num = max_pairs_num - pairs_num  # 20-2=18

        input_ids_new = []
        masked_ids_new = []
        token_type_ids_new = []
        pairwise_label_new = []

        for item in range(pairs_num):   # padding for each true pair 
            padding_pair_len = max_pair_len - len(input_ids[item])
            # print ('len(input_ids[item])', len(input_ids[item]), padding_pair_len)

            input_ids_new.append(input_ids[item] + [pad_id] * padding_pair_len)
            masked_ids_new.append(masked_ids[item] + [pad_id] * padding_pair_len)
            token_type_ids_new.append(token_type_ids[item] + [0] * padding_pair_len)

        ### padding for padded pairs
        input_ids_new = input_ids_new + [[pad_id] * max_pair_len] * padding_pair_num
        masked_ids_new = masked_ids_new + [[pad_id] * max_pair_len] * padding_pair_num   
        token_type_ids_new = token_type_ids_new + [[0] * max_pair_len] * padding_pair_num
        pairwise_labels_new = pairwise_labels + [0] * padding_pair_num  # [0,0,1,1,1,0, 0,0,0]  for one example

        pairs_list_new = pairs_list + [[0,1]] * padding_pair_num
        passage_length_new = passage_length
        pairs_num_new = pairs_num
        sep_positions_new = sep_positions + [[2,6]] * padding_pair_num

        mask_cls_new = [1] * passage_length_new + [pad_id] * padd_num_sen  # [1,1,1,1,1,0,0,0]
        ground_truth_new = ground_truth + [pad_id] * padd_num_sen   # [2,1,0,3,4,0,0,0]

        # print ('input_ids_new', np.shape(input_ids_new))
        all_input_ids.append(input_ids_new)
        all_attention_mask.append(masked_ids_new)
        all_token_type_ids.append(token_type_ids_new)
        all_pairs_list.append(pairs_list_new)
        all_passage_length.append(passage_length_new)
        all_pairs_num.append(pairs_num_new)
        all_sep_positions.append(sep_positions_new)
        all_ground_truth.append(ground_truth_new)
        all_mask_cls.append(mask_cls_new)
        all_pairwise_labels.append(pairwise_labels_new)

    # print ('all_input_ids', all_input_ids)
    all_input_ids=torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask=torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids=torch.tensor(all_token_type_ids, dtype=torch.long)
    all_pairs_list=torch.tensor(all_pairs_list, dtype=torch.long)
    all_passage_length=torch.tensor(all_passage_length, dtype=torch.long)
    all_pairs_num=torch.tensor(all_pairs_num, dtype=torch.long)
    all_sep_positions=torch.tensor(all_sep_positions, dtype=torch.long)
    all_ground_truth=torch.tensor(all_ground_truth, dtype=torch.long)
    all_mask_cls=torch.tensor(all_mask_cls, dtype=torch.long)
    all_pairwise_labels=torch.tensor(all_pairwise_labels, dtype=torch.long)

    # all_span_index = torch.tensor([f.span_index for f in features], dtype=torch.long)

    new_batch=[
        all_input_ids, all_attention_mask, all_token_type_ids,
        all_pairs_list, all_passage_length, all_pairs_num,
        all_sep_positions, all_ground_truth, all_mask_cls,
        all_pairwise_labels
    ]

    return new_batch
