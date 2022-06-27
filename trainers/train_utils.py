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


# TODO: Change to better masking from huggingface new codes.
def mask_tokens_sentence(inputs, tokenizer, args, prefix=None):
    """
        Prepare masked tokens inputs/labels for masked language modeling:
        80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    for i in range(len(inputs)):
        input_ = inputs[i]
        label_ = input_.clone()
        pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        non_zero_entries = torch.nonzero(input_-pad_id).t()[0]
        masked_indices_ = torch.bernoulli(torch.full(non_zero_entries.shape, args.mlm_probability)).bool()
        masked_indices_all_ = torch.full(input_.shape, False, dtype=torch.bool)
        start = 0
        masked_indices_all_[start:len(non_zero_entries)+start] = masked_indices_
        non_cls_pos = torch.nonzero(input_==args.cls_id, as_tuple=False)
        masked_indices_all_[non_cls_pos] = False
        # print (masked_indices_all_)
        zero_entries = torch.full(input_.shape, 1, dtype=torch.long)
        zero_entries[non_zero_entries] = 0
        # print (zero_entries)
        ele_mult = masked_indices_all_ * zero_entries
        # print ("n", non_zero_entries)
        # print ("z", zero_entries)
        # print ("m", masked_indices_all_)
        # print ('-----')
        assert torch.sum(ele_mult) == 0
        masked_indices_all_ = masked_indices_all_.bool()

        ##
        label_[~masked_indices_all_] = args.mlm_ignore_index
        # print (label_)

        # 0.8 replace
        # print ("1", input_)
        indices_replaced = torch.bernoulli(torch.full(label_.shape, 0.8)).bool() & masked_indices_all_
        input_[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 0.5 random, 0.5 unchanged
        indices_random = torch.bernoulli(torch.full(label_.shape, 0.5)).bool() & masked_indices_all_ & ~indices_replaced
        random_words = torch.randint(args.cls_id+1, len(tokenizer), label_.shape, dtype=torch.long).type_as(input_)
        input_[indices_random] = random_words[indices_random]
        # print ("2", input_)
        inputs[i] = input_
        labels[i] = label_

    # return the whole inputs
    return inputs, labels


def render_order_heatmap(args, order_list, soft=True, ranking_based=False):
    assert type(order_list) in [list, np.ndarray]

    len_list = len(order_list)
    offset = min(order_list)
    heat_map = np.zeros((len_list, len_list))
    # heat_map = np.ones((len_list, len_list))  #  * 0.5

    ranking_step = 0.9

    adj_d = {}
    for i in range(len_list):
        for j in range(i+1, len_list):
            parent = order_list[i]
            child = order_list[j]
            if parent not in adj_d:
                adj_d[parent] = []
            adj_d[parent].append(child)

    for parent in adj_d:
        len_children = len(adj_d[parent])
        for i in range(len_children):
            child = adj_d[parent][i]
            if i == 0:
                heat_map[parent-offset][child-offset] = 1
            elif ranking_based:
                heat_map[parent-offset][child-offset] = float(len_children - i) / float(len_children) * ranking_step
            elif soft:
                heat_map[parent-offset][child-offset] = 1. / 10.

    # if "v3" in args.heatmap_decode_method:
    #     parent = order_list[-1]
    #     child = order_list[0]
    #     heat_map[parent-offset][child-offset] = -1

    # if args is not None:
    #     heat_map = torch.Tensor(heat_map).to(args.device)
    # else:
    #     heat_map = torch.Tensor(heat_map)
    heat_map = torch.Tensor(heat_map)
    return heat_map


def heatmap2order(args, heat_map, soft=False):
    if type(heat_map) != np.ndarray:
        heat_map = heat_map.cpu().numpy()

    if "v3" not in args.heatmap_decode_method:
        assert np.min(heat_map) >= 0, "heat map cannot have negative values."

    len_seq = len(heat_map)

    if args.heatmap_decode_method == "super_naive":
        max_diff = 0
        start_idx = 0
        for i in range(len_seq):
            curr_diff = np.max(heat_map[i]) - np.min(heat_map[i])
            if curr_diff > max_diff:
                max_diff = max(curr_diff, max_diff)
                start_idx = i

        pred = [start_idx]
        curr_idx = start_idx
        # TODO: Make it deduplicate.
        while len(pred) < len_seq:
            curr_repr = heat_map[curr_idx]
            largest_idx = np.argsort(curr_repr)[-1]
            pred.append(largest_idx)
            curr_idx = largest_idx

    elif "naive" in args.heatmap_decode_method:
        pred = []

        def recursively_find_next(curr_row, heat_map, order_lists, 
                                  order_list_curr, probs, prob_curr):
            # order_list_curr.append(curr_row)
            curr_repr = heat_map[curr_row].copy()
            if len(order_list_curr) >= len_seq:
                if order_list_curr not in order_lists:
                    order_lists.append(order_list_curr)
                    if "v2" in args.heatmap_decode_method or "v3" in args.heatmap_decode_method:
                        prob_curr.append(curr_repr)
                    # if "v3" in args.heatmap_decode_method:
                    #     prob_curr = [abs(x) for x in prob_curr]
                    probs.append(prob_curr)
                return
            # TODO: Better solution? Beam search-like?
            # Finds the max value for each row first.
            # Prevent self-looking-loop or visited node
            # init_idx = 0
            # largest_idx = np.argsort(curr_repr)[init_idx]
            # print(order_list, curr_repr)
            # print(largest_idx)

            visited = []

            for beam_idx in range(args.heatmap_decode_beam_size):

                init_idx = -1
                largest_idx = np.argsort(curr_repr)[init_idx]

                to_next_recursive = True
                # print(order_list_curr, visited, beam_idx, curr_row, largest_idx, init_idx)
                while (largest_idx == curr_row
                       or largest_idx in order_list_curr):
                    init_idx -= 1
                    if init_idx * -1 > len_seq:
                        to_next_recursive = False
                        break
                    largest_idx = np.argsort(curr_repr)[init_idx]  # Next largest.
                    # print(largest_idx, visited)
                    if largest_idx in visited:
                        continue
                # print(largest_idx, to_next_recursive)
                
                if to_next_recursive:
                    next_row = largest_idx
                    visited.append(next_row)
                    prob_val = heat_map[curr_row][next_row]
                    curr_repr[next_row] = -1.0
                    recursively_find_next(next_row, heat_map, order_lists,
                                          order_list_curr + [next_row], probs,
                                          prob_curr + [prob_val])

        max_prob = None
        eps = 1e-8  # Small epsilon to smooth probability.
        for i in range(len_seq):
            order_lists = []
            order_list_curr = [i]
            probs = []
            prob_curr = []
            recursively_find_next(i, heat_map, order_lists, order_list_curr,
                                  probs, prob_curr)
            # print(order_lists)
            # print(probs)
            assert len(order_lists) == len(probs)
            for j in range(len(order_lists)):
                if "v2" in args.heatmap_decode_method or "v3" in args.heatmap_decode_method:
                    assert len(order_lists[j]) == len(probs[j])
                else:
                    assert len(order_lists[j]) == len(probs[j]) + 1
                order_list = order_lists[j]
                prob = probs[j]

                if "v2" in args.heatmap_decode_method:
                    reversed_headed = prob[-1]
                    curr_head = order_list[0]
                    reversed_headed_prob = reversed_headed[curr_head]
                    assert reversed_headed_prob <= 1.0, ("prob is > 1, "
                                                         "sigmoid applied?")
                    reversed_headed_prob = 1 - reversed_headed_prob
                    prob.pop()
                    prob.append(reversed_headed_prob)

                elif "v3" in args.heatmap_decode_method:
                    reversed_headed = prob[-1]
                    curr_head = order_list[0]
                    reversed_headed_prob = reversed_headed[curr_head]
                    reversed_headed_prob = abs(reversed_headed_prob)
                    assert reversed_headed_prob <= 1.0, ("prob is > 1, "
                                                         "sigmoid applied?")
                    prob.pop()
                    prob.append(reversed_headed_prob)

                if "v3" in args.heatmap_decode_method:
                    prob = [abs(x) for x in prob]

                prob = np.asarray(prob)
                if "sum" not in args.heatmap_decode_method:
                    prob += eps  # Smoothing.
                    prob = np.log(prob)
                    total_prob = np.sum(prob)
                else:
                    total_prob = np.sum(prob)
                if max_prob is None:
                    max_prob = total_prob
                    pred = order_list
                elif total_prob > max_prob:
                    max_prob = total_prob
                    pred = order_list

        set_pred = list(set(pred))
        if len(set_pred) != len_seq:
            raise ValueError("The decoded order is not valid: {}".format(pred))

    elif args.heatmap_decode_method == "topological":
        thres = 0.2
        graph = Graph(len_seq)
        for i in range(len_seq):
            for j in range(len_seq):
                if i < j:
                    curr_val = heat_map[i][j]
                    if curr_val > thres:
                        pred_label = 1
                    else:
                        pred_label = 0
                    if pred_label == 1:  # Ordered.
                        graph.addEdge(i, j)
                    else:
                        graph.addEdge(j, i)

        pred = graph.topologicalSort()

        set_pred = list(set(pred))
        if len(set_pred) != len_seq:
            raise ValueError("The decoded order is not valid: {}".format(pred))

    elif args.heatmap_decode_method == "mst":
        # Construct the graph dictionary.
        from trainers.neural_dependency_parser.mst import mst

        graph = {}
        for i in range(len_seq):
            key = i
            item = {}
            for j in range(len_seq):
                if i != j:
                    item[j] = heat_map[i][j]
            graph[key] = item

        mst_tree = mst(graph)
        serl = []        

        # TODO: Implement topo sort over an mst.
        g =  Graph(len_seq)
        for i in range(len_seq):
            u = i
            _v = mst_tree[i]
            for v in _v:
                g.addEdge(u, v)
        pred = g.topologicalSort()

        """
        for i in range(len_seq):
            u = i
            _v = mst_tree[i]
            if len(_v) > 1:
                tmp_serl = sorted([(_v[k], k) for k in _v])
                top_temp_serl = tmp_serl[-1]
                _v = {top_temp_serl[1]: top_temp_serl[0]}
            elif len(_v) < 1:
                pass
            assert len(_v) == 1, "Tree: {}".format(mst_tree)
            v = list(_v.keys())[0]
            val = _v[v]
            serl.append((val, "{}-{}".format(u, v)))
        sorted_serl = sorted(serl)
        head = int(sorted_serl[0][1].split("-")[-1])

        path = []
        curr_node = head
        while len(path) < len_seq:
            path.append(curr_node)
            _next_node = mst_tree[curr_node]
            curr_node = list(_next_node.keys())[0]
        # raise NotImplementedError("MST not sure what to do yet.")
        pred = path
        """

    else:
        raise NotImplementedError("Heatmap decoding method: {} not "
                                  "found.".format(args.heatmap_decode_method))
    
    return pred


if __name__ == "__main__":
    class dummy_args(object):
        def __init__(self):
            self.device = "cpu"
            self.heatmap_decode_method = "super_naive"
            self.heatmap_decode_method = "mst"
            self.heatmap_decode_method = "naive_v2"
            self.heatmap_decode_method = "naive_v2_sum"
            self.heatmap_decode_beam_size = 2

    args = dummy_args()
    order_list = [2, 3, 4, 1, 5]
    heat_map = render_order_heatmap(args, order_list, soft=False)
    print(heat_map)

    heat_map[-1] += 0.2
    heat_map_blurred = gaussian_filter(heat_map.cpu().numpy(), sigma=0.3)
    print(heat_map_blurred)
    print(np.sum(heat_map_blurred, axis=1))

    kl_div = torch.nn.KLDivLoss()
    sm = torch.nn.Softmax()
    sm2 = torch.nn.Softmax2d()
    print(torch.sum(heat_map))
    
    # heat_map_ = sm2(heat_map.unsqueeze(0).unsqueeze(0)).log().squeeze(0)
    heat_map_ = (0.2*heat_map.unsqueeze(0).unsqueeze(0)+1e-6).log().squeeze(0)
    heat_map_blurred = np.random.randn(5, 5)
    heat_map_blurred /= np.sum(heat_map_blurred)
    # heat_map_blurred_ = sm2(torch.Tensor(heat_map_blurred).unsqueeze(0).unsqueeze(0)).cuda().squeeze(0)
    heat_map_blurred_ = torch.Tensor(heat_map_blurred).unsqueeze(0).unsqueeze(0).to(args.device).squeeze(0)
    print(heat_map_)
    print(heat_map_blurred_)
    print(heat_map_.size(), heat_map_blurred_.size())
    kl_loss = kl_div(heat_map_, heat_map_blurred_)
    print(kl_loss)

    print(heat_map)
    pred_order = heatmap2order(args, heat_map, soft=False)
    print(pred_order)
