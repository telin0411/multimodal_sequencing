import csv
import glob
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm
import numpy as np
import argparse

import torch
from torch import nn

from dataclasses import dataclass
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from skimage import io, transform
from torchvision import transforms, utils
import torchvision.models as models
from datasets.img_utils import read_and_transform_img_from_filename
from datasets.img_utils import Rescale, RandomCrop, ToTensor

from datasets.recipeqa import RecipeQAGeneralProcessor
from datasets.mpii_movie import MPIIMovieGeneralProcessor

# HKMeans.
from hkmeans.hierarchical_kmeans import get_hkmeans_clusters, init_hkmeans_clusters
from hkmeans.hierarchical_kmeans import hierarchical_kmeans


processors = {
    "recipeqa": RecipeQAGeneralProcessor,
    "mpii_movie": MPIIMovieGeneralProcessor,
}

logger = logging.getLogger(__name__)


def get_multimodal_utils(args=None):
    if args is None:
        vision_model = "resnet18"
    else:
        vision_model = args.vision_model

    if "resnet" in args.vision_model:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # TODO: Verify we can use our own ToTensor() function.
        img_transform_func = transforms.Compose([
            Rescale((224, 224)),
            ToTensor(),
            normalize,
        ])

    if vision_model == "resnet18":
        model = models.resnet18(pretrained=True)
    elif vision_model == "resnet50":
        model = models.resnet50(pretrained=True)
    elif vision_model == "resnet101":
        model = models.resnet101(pretrained=True)

    model.fc = nn.Identity()

    return model, img_transform_func


def process_single_example(args, example):
    img_paths = example.img_path_seq
    for img_path in img_paths:
        img_path_name, img_path_ext = os.path.splitext(img_path)
        npy_path = img_path.replace(img_path_ext, 
                                    "_{}.npy".format(args.vision_model))
        if args.delete_files and args.extract_mode == "multi_files":
            if os.path.exists(npy_path):
                if args.verbose:
                    print("Found and delete file: {}".format(npy_path))
                os.remove(npy_path)
        else:
            if os.path.exists(npy_path):
                continue
            img = read_and_transform_img_from_filename(img_path,
                img_transform_func=args.img_transform_func).to(args.device)
            img_batch = img.unsqueeze(0).float()
            img_embd = args.model(img_batch)
            img_embd_npy = img_embd[0].detach().cpu().numpy()
            np.save(npy_path, img_embd_npy)
            if args.verbose:
                print("Save to file: {}".format(npy_path))
    pass


def process_examples(args, examples):
    for example in tqdm.tqdm(examples, total=len(examples)):
        process_single_example(args, example)
    pass


def get_embeddings(args, examples):
    img_names = []
    img_embds = []
    for example in tqdm.tqdm(examples, total=len(examples)):
        img_paths = example.img_path_seq
        for img_path in img_paths:
            img_path_name, img_path_ext = os.path.splitext(img_path)
            npy_path = img_path.replace(img_path_ext, 
                                        "_{}.npy".format(args.vision_model))
            if not os.path.exists(npy_path):
                raise ValueError("No such file {}! You should probably"
                " extract the features in this dataset {}"
                " first.".format(npy_path, args.task_names))
            else:
                img_embd = np.load(npy_path)
                if img_path not in img_names:
                    img_names.append(img_path)
                    img_embds.append(list(img_embd))

    img_embds = np.asarray(img_embds)
    print("Data size: {}".format(img_embds.shape))

    return img_names, img_embds


def bfs_hk(hk):
    root = hk.root
    assignments = {}
    visited = {}

    queue = []
    queue.append(root)
    visited = {}
    
    while queue:
        node = queue.pop(0)
        name = node.name
        visited[name] = True
        assignments[name] = node.get_additional_data()

        for child in node.children:
            if child.name not in visited:
                queue.append(child)

    return assignments


def do_hkmeans(args, examples):
    out_root = "inspection_results"
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    task_str = "_".join(args.task_names)
    num_clusters_str = [str(x) for x in args.num_clusters]
    num_clusters_str = "hkmeans_" + "_".join(num_clusters_str)
    iter_str = "iter{}".format(args.hkmeans_iterations)
    network = args.vision_model
    all_strs = [task_str, network, num_clusters_str, iter_str]
    file_str = "_".join(all_strs)

    if args.input_img_names_json is not None:
        assert task_str in args.input_img_names_json
        img_names = json.load(open(args.input_img_names_json, "r"))
        img_names = img_names["img_names"]
        hk_pkl_file = args.input_img_names_json.replace(".json", ".pkl")

        # hk = init_hkmeans_clusters(data=None, num_clusters=args.num_clusters)
        hk = hierarchical_kmeans.load(hk_pkl_file)
        args.img_names_str = args.input_img_names_json
        construct_visual_token(args, img_names, hk)
        return hk

    img_names, img_embds = get_embeddings(args, examples)

    hk = get_hkmeans_clusters(img_embds, args.num_clusters,
                              iteration=args.hkmeans_iterations)

    hk_str = os.path.join(out_root, file_str+".pkl")
    print("Saving the resulting hkmeans cluster to {}".format(hk_str))
    hk.save(hk_str)

    img_names_str = os.path.join(out_root, file_str+".json")
    print("Saving the resulting image names to {}".format(img_names_str))
    img_names_d = {"img_names": img_names}
    json.dump(img_names_d, open(img_names_str, "w"))

    args.img_names_str = img_names_str

    construct_visual_token(args, img_names, hk)

    return hk


def construct_visual_token(args, img_names, hk):
    all_assignments = bfs_hk(hk)
    new_assignments = {}
    for c in all_assignments:
        if all_assignments[c] is None or len(all_assignments[c]) == 0:
            pass
        else:
            new_assignments[c] = all_assignments[c]
    keys = sorted(list(new_assignments.keys()))
    vt_mapping = {k: i for i, k in enumerate(keys)}

    vt_dict = {}

    for c in vt_mapping:
        img_ids = new_assignments[c]
        for img_id in img_ids:
            vt_dict[img_names[img_id]] = vt_mapping[c]

    print_cnt = 0
    for img_name in vt_dict:
        if print_cnt > 20:
            break
        print(img_name, vt_dict[img_name])
        print_cnt += 1

    npy_name = args.img_names_str.replace(".json", ".npy")
    np.save(npy_name, vt_dict)
    print("Saving the visual token dict to {}".format(npy_name))
        
    return vt_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_names",
        default=None,
        type=str,
        nargs="+",
        required=True,
        help=("The name of the task, see datasets/processors.py"),
    )
    parser.add_argument(
        "--vision_model",
        default="resnet18",
        type=str,
        required=False,
        choices=[
            "resnet18", "resnet50", "resnet101",
        ],
        help=("The vision model."),
    )
    parser.add_argument(
        "--extract_mode",
        default="multi_files",
        type=str,
        required=False,
        choices=[
            "multi_files", "one_file",
        ],
        help=("The mode to extract the images and save."),
    )
    parser.add_argument(
        "--delete_files",
        action="store_true",
        help="If delete the files created accordig to the current settings.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If printing verbose statements.",
    )
    parser.add_argument(
        "--perform_hkmeans",
        action="store_true",
        help="If conduct hkmeans.",
    )
    parser.add_argument(
        "--num_clusters",
        nargs="+",
        type=int,
        default=None,
        help="Number of the hierarchical clusters.",
    )
    parser.add_argument(
        "--hkmeans_iterations",
        type=int,
        default=1,
        help="Number of the hkmeans iters.",
    )
    parser.add_argument(
        "--input_img_names_json",
        type=str,
        default=None,
        help="The input json file which stores the image names.",
    )
    args = parser.parse_args()

    print("Extracting image features of datasets: {}".format(args.task_names))
    print("Using vision model: {}".format(args.vision_model))

    examples = []
    print("-"*50)
    for task_name in args.task_names:
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.extract_mode == "one_file":
        raise NotImplementedError(
            "Extract mode {} not done yet!".format(args.extract_mode))

    if args.perform_hkmeans:
        assert args.num_clusters is not None
        print("Performing hkmeans with clusters: {}".format(args.num_clusters))
        hk = do_hkmeans(args, examples)
    else:
        model, img_transform_func = get_multimodal_utils(args=args)
        model = model.to(args.device)
        args.model = model
        args.img_transform_func = img_transform_func

        process_examples(args, examples)


if __name__ == "__main__":
    main()
