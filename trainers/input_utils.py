import csv
import glob
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from trainers.topological_sort import Graph
from transformers import AutoTokenizer, AutoModel


def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index


def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len-tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad


def get_detailed_input_feats(batch, tokenizer, args=None):
    if args is None:
        pass

    bz = batch["input_ids"].size()[0]
    seq_length = batch["input_ids"].size(1)

    cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    input_ids = batch["input_ids"]

    padding_idx = 1
    position_ids = torch.arange(
        0,  # padding_idx + 1,
        seq_length,  #  + padding_idx + 1,
        dtype=torch.long,
        device=input_ids.device,
    )
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    text_lens = []
    for i in range(bz):
        input_ids_ = input_ids[i]
        pad_pos = torch.where(input_ids_==pad_id)
        if len(pad_pos[0]) == 0:
            len_text = torch.Tensor([len(input_ids_)])[0].type_as(input_ids_)
        else:
            len_text = pad_pos[0][0]
        text_lens.append(len_text)

    text_lens = torch.stack(text_lens)
    if "images" in batch:
        num_bbs = [f.size(0) for f in batch["images"]]
    else:
        batch["images"] = None
    out_size = batch["attention_mask"].size(1)

    # if batch["images"] is not None:
    #     gather_index = get_gather_index(text_lens, num_bbs, bz,
    #                                     input_ids.size(1),
    #                                     input_ids.size(1)+num_bbs[0])
    # else:
    #     gather_index = None

    if args.img_text_paired_coattention:
        assert batch["images"] is not None, "No images!"
        images = batch["images"]

        bz, text_len = input_ids.size()
        cls_repr_pos = []
        for i in range(bz):
            cls_repr_pos_i = torch.nonzero(
                input_ids[i]==args.cls_id, as_tuple=False)
            cls_repr_pos.append(cls_repr_pos_i)

        paired_co_attention_mask = torch.zeros(
            input_ids.size(0),
            input_ids.size(1) + images.size(1),
            input_ids.size(1) + images.size(1),
        ).type_as(images)
        if args.include_num_img_regional_features is not None:
            paired_co_attention_mask = torch.zeros(
                input_ids.size(0),
                input_ids.size(1) + images.size(1) + args.include_num_img_regional_features * images.size(1),
                input_ids.size(1) + images.size(1) + args.include_num_img_regional_features * images.size(1),
            ).type_as(images)

        # TODO: All text can see each other.
        paired_co_attention_mask[:, :input_ids.size(1), :input_ids.size(1)] = batch["attention_mask"].unsqueeze(1)

        for i in range(bz):
            cls_repr_pos_i = cls_repr_pos[i].squeeze().cpu().numpy()
            for j in range(len(cls_repr_pos_i)):
                # TODO: Currently the last sequence item is not
                # truncated yet.
                if j == len(cls_repr_pos_i) - 1:  # The last item.
                    start = cls_repr_pos_i[j]
                    end = input_ids.size(1)
                else:
                    start = cls_repr_pos_i[j]
                    end = cls_repr_pos_i[j+1]
                if args.include_num_img_regional_features is not None:
                    beta = args.include_num_img_regional_features
                    paired_co_attention_mask[i,
                        input_ids.size(1)+j*beta+j:input_ids.size(1)+(j+1)*beta+j+1,
                        start:end] = 1.0
                    paired_co_attention_mask[i,
                        start:end,
                        input_ids.size(1)+j*beta+j:input_ids.size(1)+(j+1)*beta+j+1] = 1.0
                    paired_co_attention_mask[i,
                        input_ids.size(1)+j*beta+j:input_ids.size(1)+(j+1)*beta+j+1,
                        input_ids.size(1)+j*beta+j:input_ids.size(1)+(j+1)*beta+j+1] = 1.0
                else:
                    paired_co_attention_mask[i, input_ids.size(1)+j, start:end] = 1.0
                    paired_co_attention_mask[i, start:end, input_ids.size(1)+j] = 1.0
                    paired_co_attention_mask[i, input_ids.size(1)+j, input_ids.size(1)+j] = 1.0
        co_attention_mask = paired_co_attention_mask
        batch["attention_mask"] = co_attention_mask

    batch = {
        "input_ids": batch["input_ids"],
        "txt_type_ids": batch["token_type_ids"],
        "position_ids": position_ids,
        "img_feat": batch["images"],
        "img_pos_feat": None,
        "attn_masks": batch["attention_mask"],
        # "gather_index": gather_index,
        "targets": None,
    }

    if "multimodal_pretrain_objectives" in args.__dict__:
        if "itm_ot" in args.multimodal_pretrain_objectives:
            txt_lens = [int(x.cpu().numpy()) for x in text_lens]
            max_tl = max(txt_lens)
            max_nbb = max(num_bbs)
            scatter_len = batch["attn_masks"].size(1)
            if batch["img_feat"] is not None:
                scatter_len += batch["img_feat"].size(1)
            ot_scatter = _compute_ot_scatter(txt_lens, batch["input_ids"].size(1),
                                             scatter_len)
            txt_pad = _compute_pad(txt_lens, batch["input_ids"].size(1))
            img_pad = _compute_pad(num_bbs, batch["img_feat"].size(1))
            ot_inputs = {
                "ot_scatter": ot_scatter,
                "scatter_max": ot_scatter.max().item(),
                "txt_pad": txt_pad,
                "img_pad": img_pad,
            }
            batch["ot_inputs"] = ot_inputs

    return batch


if __name__ == "__main__":
    class dummy_args(object):
        def __init__(self):
            self.device = "cuda"
            self.heatmap_decode_method = "super_naive"
            self.heatmap_decode_method = "mst"
            self.heatmap_decode_method = "naive_v2"
            self.heatmap_decode_method = "naive_v2_sum"
            self.heatmap_decode_beam_size = 2

    args = dummy_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    s = [
        "I am a boy.",
        "I am a good girl."
    ]
    s_t = tokenizer(
        s,
        max_length=10,
        padding="max_length",
        truncation=True,
    )
    print(s_t)
    print(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))
    print(tokenizer.convert_tokens_to_ids(tokenizer.cls_token))

    batch = {x: torch.Tensor(s_t[x]).to(args.device) for x in s_t}
    batch["images"] = torch.rand(len(s), 5, 32, 32, 3)

    batch = get_detailed_input_feats(batch, tokenizer)
    # print(batch)
