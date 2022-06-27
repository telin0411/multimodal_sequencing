import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import csv
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
from trainers.metrics import compute_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def berson_evaluate(args, model, load_and_cache_examples, tokenizer,
                    prefix="", data_split="test", human_evaluate=False):

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

        best_acc = []
        truth = []
        predicted = []
        guids = []

        f = open(os.path.join(args.output_dir, "output_order.txt"), 'w')

        nb_eval_steps = 0
        multiref_gt = False
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            # batch = tuple(t.to(args.device) for t in batch)

            tru = batch[3]
            if tru.ndim > 2:  # multi-ref
                tru = tru[0].tolist()
                if not multiref_gt: multiref_gt = True
            else:
                tru = tru.view(-1).tolist()
            truth.append(tru)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                          "labels": batch[3]}
                if inputs["labels"].ndim > 2:  # multiref
                    inputs["labels"] = inputs["labels"][:, 0, :]

                if args.multimodal:
                    inputs["images"] = batch[-1]
                    if args.include_num_img_regional_features:
                        inputs["img_regional_features"] = batch[-2]

                if len(tru) == 1 and not multiref_gt:
                    pred = tru
                else:
                    pred = berson_pointer_network(args, model, tokenizer, inputs)
                    # from models.berson.process_inputs_for_berson import prepare_berson_inputs
                    # from models.berson.modeling_bert import beam_search_pointer
                    # inputs = prepare_berson_inputs(inputs, tokenizer, args=args)
                    # pred = beam_search_pointer(args, model, **inputs)

                guid = batch[4][0]
                guid = str(guid).split("###")[0]
                guids.append(guid)

                predicted.append(pred)
                print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
                      file=f)

            nb_eval_steps += 1
            if args.max_eval_steps > 0 and nb_eval_steps >= args.max_eval_steps:
                logging.info("Early stopping"
                    " evaluation at step: {}".format(args.max_eval_steps))
                break

        accs, pmr, taus = cal_result(truth, predicted, best_acc, f, args=args)

        results['acc_dev'] = accs
        results['pmr_dev'] = pmr
        results['taus_dev'] = taus

        if args.eval_save_all_results:

            out_csv = os.path.join(args.output_dir, "all_predictions.csv")
            csv_f = open(out_csv, "w")
            fieldnames = ["url", "pm", "em", "lcs_substr", "lcs", "ms", "wms", "dist", "tau"]
            csv_r = csv.DictWriter(csv_f, fieldnames=fieldnames)
            csv_r.writeheader()

            for c in range(len(predicted)):
                pred, label = [predicted[c]], [truth[c]]
                
                metric_methods = {
                    "pm": "partial_match",
                    "em": "exact_match",
                    "lcs_substr": "lcs_substr",
                    "lcs": "lcs",
                    "ms": "ms",
                    "wms": "wms",
                    "dist": "distance_based",
                    "tau": "tau",
                }

                row = {}
                for met in ["pm", "em", "lcs_substr", "lcs", "ms", "wms", "dist", "tau"]:
                    met_name = metric_methods[met]
                    acc_c = compute_metrics(args, met_name, pred, label)
                    row[met] = acc_c
                url = guids[c]
                row["url"] = url
                csv_r.writerow(row)

            csv_f.close()
            print("Saving all prediction csv file at: {}".format(out_csv))

        output_eval_file = os.path.join(eval_output_dir,
                                        prefix, "eval_results_split_{}.txt".format(data_split))

        with open(output_eval_file, "w") as writer:
            # logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                # logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

        output_only_eval_file_1 = os.path.join(args.output_dir, "all_eval_results.txt")
        fh = open(output_only_eval_file_1, 'a')
        fh.write(prefix)
        for key in sorted(results.keys()):
            fh.write("%s = %s\n" % (key, str(results[key])))
        fh.close()

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

    multiref_gt = False
    for t, p in zip(truth, predicted):
        # print ('t, p', t, p)
        if np.asarray(t).ndim > 1:
            t_org = t
            t = t[0]
            if not multiref_gt: multiref_gt = True
        else:
            t_org = t

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

        to_compare.append((eq.sum()/len(t), eq.all(), idx, p, t_org))
        idx += 1

    if multiref_gt:
        truth_flattened = [t[0] for t in truth]
    else:
        truth_flattened = truth
    acc = accuracy_score(list(itertools.chain.from_iterable(truth_flattened)),
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
        metrics_list = ["partial_match", "exact_match", "lcs", "lcs_substr", "distance_based", "ms", "wms", "tau"]
        if args.ref_json_file is not None:
            import csv
            import json
            ref_data = []
            json_f = open(args.ref_json_file, "r")
            if "recipeQA" not in args.ref_json_file:
                for line in json_f:
                    ref_data.append(json.loads(line.strip()))
            else:
                ref_data = json.load(json_f)
                new_ref_data = []
                used_ids = {}
                for ref_d in ref_data["data"]:
                    if ref_d["recipe_id"] in used_ids:
                        continue
                    used_ids[ref_d["recipe_id"]] = True
                    new_ref_data.append(ref_d)
                ref_data = {"data": new_ref_data}
            print(len(to_compare), len(ref_data))
            # assert len(to_compare) == len(ref_data)
            # to_compare = sorted(to_compare)

            csv_path = "{}_model_performance.csv".format(args.ref_json_file.split(".json")[0].split("/")[-1])
            csv_path = os.path.join(args.output_dir, csv_path)
            # fieldnames = ["partial_match", "exact_match", "index", "url", "prediction", "gt"]
            fieldnames = ["index", "url", "prediction", "gt"] + metrics_list
            csv_f = csv.DictWriter(open(csv_path, "w"), fieldnames=fieldnames)
            csv_f.writeheader()

            jsonl_path = "{}_model_performance.jsonl".format(args.ref_json_file.split(".json")[0].split("/")[-1])
            jsonl_path = os.path.join(args.output_dir, jsonl_path)

            rows = []

            for each in to_compare:
                acc_curr, pmr_curr, idx, pred, gt = each
                if "recipeQA" in args.ref_json_file:
                    url = ref_data["data"][idx]["recipe_id"]
                else:
                    url = ref_data[idx]["url"]
                print(acc_curr, pmr_curr, idx, url, pred, gt)
                row = {}
                row["partial_match"] = acc_curr
                row["exact_match"] = pmr_curr
                row["index"] = idx
                row["url"] = url
                row["prediction"] = pred
                row["gt"] = gt
                for metrics in metrics_list:
                    perf = compute_metrics(args, metrics, [pred], [gt])
                    row[metrics] = perf
                csv_f.writerow(row)
                rows.append(row)

            print("Saving performance file to: {}".format(csv_path))

            with open(jsonl_path, "w") as outf:
                if "recipeQA" in args.ref_json_file:
                    rows = sorted(rows, key=lambda x: x["url"])
                for d in rows:
                    outf.write(json.dumps(d)+"\n")
            print("Saving performance file to: {}".format(jsonl_path))

    if True:
        metrics_list = ["partial_match", "exact_match", "lcs", "lcs_substr", "distance_based", "ms", "wms", "tau"]
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
            # taus,
            res["tau"],
            res["distance_based"],
        )
        logger.info("***** Paper Results *****")
        logger.info(" {}".format(headers))
        logger.info(" {}".format(content))


    return accs, pmr, taus
