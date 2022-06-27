import argparse
import glob
import logging
import os
import random
import itertools

import numpy as np
import logging


logger = logging.getLogger(__name__)


METRICS = [
    "partial_match", "exact_match", "lcs_substr", "lcs",
    "tau", "ms", "wms", "distance_based",
]


def multiref_metrics(args, preds, labels):
    res = {}
    if args.multiref_metrics == "max":
        res = {}
        for i in range(len(preds)):
            pred = preds[i]
            multi_labels = labels[i]
            metric_list = []
            for j in range(len(multi_labels)):
                label = multi_labels[j]
                pred, label = make_same_len(pred, label)
                metric_tuple = []
                for metric in METRICS:
                    acc = compute_metrics(args, metric, [pred], [label])
                    metric_tuple.append(acc)
                metric_list.append(metric_tuple)
            metric_list = sorted(metric_list, reverse=True)
            metric_max = metric_list[0]
            for metric_idx in range(len(METRICS)):
                metric = METRICS[metric_idx]
                if metric not in res:
                    res[metric] = 0
                res[metric] += metric_max[metric_idx]
        for metric in METRICS:
            res[metric] /= len(preds)
    elif args.multiref_metrics == "max":
        raise NotImplementedError("Can't deal with multiref metric: {} yet!".format(
            args.multiref_metrics))
    elif args.multiref_metrics == "max":
        raise NotImplementedError("Can't deal with multiref metric: {} yet!".format(
            args.multiref_metrics))
    else:
        raise NotImplementedError("Can't deal with multiref metric: {} yet!".format(
            args.multiref_metrics))
    return res


def compute_metrics(args, metrics, preds, labels):
    assert len(preds) == len(labels), ("Predictions and labels have "
                                       "mismatched lengths {len(preds)} "
                                       "and {len(labels)}")
    
    # multi-ref
    if np.asarray(labels[0]).ndim > 1:
        # logging.info("MultiRef GT Metric: {}".format(args.multiref_metrics))
        res = multiref_metrics(args, preds, labels)
        return res[metrics]

    acc = 0.0
    if metrics == "tau":
        for i in range(len(preds)):
            p = preds[i]
            t = labels[i]
            p, t = make_same_len(p, t)
            s_t = set([i for i in itertools.combinations(t, 2)])
            s_p = set([i for i in itertools.combinations(p, 2)])
            cn_2 = len(p) * (len(p) - 1) / 2
            pairs = len(s_p) - len(s_p.intersection(s_t))
            tau = 1 - 2 * pairs / cn_2
            acc += tau
    elif metrics == "partial_match":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            pred = np.asarray(pred)
            label = np.asarray(label)
            acc_curr = (pred == label).mean()
            acc += acc_curr
    elif metrics == "exact_match":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            pred = np.asarray(pred)
            label = np.asarray(label)
            acc += float((np.sum(pred==label)==len(pred)))
            # if np.sum(pred==label)==len(pred):
            #     print(i, pred, label)
    elif metrics == "distance_based":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            pred = list(pred)
            dist_curr = 0
            for j in range(len(label)):
                gt_idx = label[j]
                if gt_idx not in pred:
                    dist_curr = args.max_story_length
                else:
                    pred_pos = pred.index(gt_idx)
                    dist_curr += abs(j - pred_pos)
            acc += dist_curr
    elif metrics == "longest_common_subsequence" or metrics == "lcs":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            current_lcs = LCS(pred, label, len(pred), len(label))
            acc += current_lcs
    elif metrics == "longest_common_substring" or metrics == "lcs_substr":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            current_lcs_substr = LCSubStr(pred, label, len(pred), len(label))
            acc += current_lcs_substr
    elif metrics == "ms":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            current_ms = MS(pred, label)
            acc += current_ms
    elif metrics == "wms":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            current_ms = MS(pred, label, weighted=True)
            acc += current_ms
    elif metrics == "tau":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            s_t = set([j for j in itertools.combinations(label, 2)])
            s_p = set([k for k in itertools.combinations(pred, 2)])
            cn_2 = len(pred) * (len(pred) - 1) / 2
            pairs = len(s_p) - len(s_p.intersection(s_t))
            tau = 1 - 2 * pairs / cn_2
            acc += tau
    elif metrics == "head_prediction":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            pred_head = pred[0]
            label_head = label[0]
            if pred_head == label_head:
                acc += 1.0
    elif metrics == "pairwise_prediction":
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            pred, label = make_same_len(pred, label)
            all_pairs = {}
            curr_acc = 0.0
            for j in range(len(label)):
                for k in range(j+1, len(label)):
                    parent, child = label[j], label[k]
                    s = str(parent) + "_" + str(child)
                    all_pairs[s] = True
            for j in range(len(pred)):
                for k in range(j+1, len(pred)):
                    parent, child = pred[j], pred[k]
                    s = str(parent) + "_" + str(child)
                    if s in all_pairs:
                        curr_acc += 1.0
            acc += (curr_acc / float(len(all_pairs)))
    else:
        raise NotImplementedError("Metric {} is not "
                                  "implemented yet.".format(metrics))
    acc /= len(preds)
    return acc


def make_same_len(pred, label):
    if type(label) != list:
        label = label.tolist()
    len_pred = len(pred)
    len_label = len(label)
    min_len = min(len_pred, len_label)
    return pred[:min_len], label[:min_len]


def LCSubStr(X, Y, m, n):
 
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains the
    # length of longest common suffix of
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program.
 
    # LCSuff is the table with zero
    # value initially in each cell
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
 
    # To store the length of
    # longest common substring
    result = 0
 
    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result


def LCS(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    elif X[m-1] == Y[n-1]:
        return 1 + LCS(X, Y, m-1, n-1)
    else:
        return max(LCS(X, Y, m, n-1), LCS(X, Y, m-1, n))

# Code adapted from: https://www.geeksforgeeks.org/minimum-number-swaps-required-sort-array/
def MS(pred, label, weighted=False):
    n = len(pred)

    # Create two arrays and use 
    # as pairs where first array 
    # is element and second array
    # is position of first element
    pred_pos = [pred.index(x) for x in label]
     
    # Sort the array by array element 
    # values to get right position of 
    # every element as the elements 
    # of second array.
    label_pos = list(range(n))

    arrpos = [(u, v) for u, v in zip(pred_pos, label_pos)]
     
    # To keep track of visited elements. 
    # Initialize all elements as not 
    # visited or false.
    vis = {k : False for k in range(n)}
     
    # Initialize result
    ans = 0
    for i in range(n):
         
        # alreadt swapped or 
        # alreadt present at 
        # correct position
        if vis[i] or arrpos[i][0] == i:
            continue
             
        # find number of nodes 
        # in this cycle and
        # add it to ans
        cycle_size = 0
        j = i
         
        while not vis[j]:
             
            # mark node as visited
            vis[j] = True
             
            # move to next node
            j = arrpos[j][0]
            cycle_size += 1
             
        # update answer by adding
        # current cycle
        if cycle_size > 0:
            if weighted:
                ans += (cycle_size - 1) * abs(arrpos[i][0] - arrpos[i][1])
            else:
                ans += (cycle_size - 1)
             
    # return answer
    return ans


if __name__ == "__main__":
    X = [1, 2, 3, 4]
    Y = [4, 1, 2, 3]
    print(LCS(X, Y, len(X), len(Y)))

    X = [3, 2, 4, 1]
    Y = [3, 4, 2, 1]
    ms = MS(X, Y)
    wms = MS(X, Y, weighted=True)
    print(ms, wms)

    X = [3, 2, 0, 1, 4]
    Y = [2, 0, 1, 4, 3]
    m, n = len(X), len(Y)
    print("LCSubStr", LCSubStr(X, Y, m, n))

    acc = compute_metrics(None, "pairwise_prediction", [X], [Y])
    acc = compute_metrics(None, "lcs_substr", [X], [Y])
    print(acc)
