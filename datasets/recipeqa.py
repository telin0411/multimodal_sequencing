import os
import json
import csv
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from .utils import DataProcessor
from .utils import InputPairWiseExample, InputHeadExample, InputAbductiveExample


# TODO: change.
RECIPEQA_DATA_ROOT = "data/recipeQA"


class RecipeQAPairWiseProcessor(DataProcessor):
    """Processor for RecipeQA Steps Dataset, pair-wise data.
    Args:
        data_dir: string. Root directory for the dataset.
        order_criteria: The criteria of determining if a pair is ordered or not.
            "tight" means only strictly consecutive pairs are considered as
            ordered, "loose" means ancestors also count.
        paired_with_image: will only consider sequence that have perfect image
            pairings.
        min_story_length: minimum length of sequence for each.
        max_story_length: maximum length of sequence for each.
    """

    def __init__(self, data_dir=None, order_criteria="tight",
                 paired_with_image=True,
                 min_story_length=5, max_story_length=5,
                 version_text=None,
                 caption_transforms=None, **kwargs):
        """Init."""
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = RECIPEQA_DATA_ROOT
        assert order_criteria in ["tight", "loose"]
        self.order_criteria = order_criteria
        self.paired_with_image = paired_with_image

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        self.caption_transforms = caption_transforms
        self.version_text = version_text

        self.multiref_gt = False

    def get_labels(self):
        """See base class."""
        return ["unordered", "ordered"]  # 0: unordered, 1: ordered.

    def _read_image_paths(self, data_dir=None, split="train"):
        img_paths_dict = {}
        if data_dir is None:
            data_dir = self.data_dir

        split = "*"
        img_dir = os.path.join(data_dir, "images", "images-qa",
                               split, "images-qa")
        all_img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        for img_path_raw in sorted(all_img_paths):
            img_path = img_path_raw.strip()
            img_name = img_path.split("/")[-1].split(".")[0]

            img_name_splits = img_name.split("_")
            if not img_name_splits[-2].isdigit():
                recipe_id = "_".join(img_name.split("_")[:-1])
                step_id = int(img_name.split("_")[-1])
            else:
                recipe_id = "_".join(img_name.split("_")[:-2])
                step_id = int(img_name.split("_")[-2])
                img_id = int(img_name.split("_")[-1])
            if recipe_id not in img_paths_dict:
                img_paths_dict[recipe_id] = {}
            if step_id not in img_paths_dict[recipe_id]:
                img_paths_dict[recipe_id][step_id] = []
            img_paths_dict[recipe_id][step_id].append(img_path_raw)
        return img_paths_dict
        
    def _read_json(self, data_dir=None, split="train"):
        """Reads in json lines to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir
        json_path = os.path.join(data_dir, "texts", split+".json")

        if self.version_text is not None:
            json_path = os.path.join(data_dir, "new_splits", split+"-{}.json".format(self.version_text))
            if not os.path.exists(json_path):
                raise ValueError("File: {} not found!".format(json_path))
        print("Using {}".format(json_path))

        image_paths = self._read_image_paths(data_dir=data_dir, split=split)

        json_file = json.load(open(json_path))
        data = json_file["data"]
        
        story_seqs = []

        used_recipe_ids = {}

        # Each element in a story seq is (text, image) tuple.
        for data_raw in tqdm(data, total=len(data)):
            recipe_id = data_raw["recipe_id"]
            if recipe_id in used_recipe_ids:
                continue
            used_recipe_ids[recipe_id] = True
            context = data_raw["context"]
            image_paths_curr = image_paths[recipe_id]

            story_seq = [recipe_id]

            # Multi-reference GTs.
            if "multiref_gt" in data_raw:
                if not self.multiref_gt: self.multiref_gt = True
            
            for step in context:
                text = step["body"]
                if self.caption_transforms is not None:
                    text = self.caption_transforms.transform(text)
                step_id = int(step["id"])

                if self.paired_with_image:
                    if step_id not in image_paths_curr:
                        # raise ValueError("step_id: {} {}\n{}".format(step_id, recipe_id, image_paths_curr))
                        continue
                    else:
                        # We take the first image for each step.
                        image_path_curr = image_paths_curr[step_id][0]
                        element = (text, image_path_curr)
                else:
                    if step_id not in image_paths_curr:
                        element = (text, None)
                    else:
                        # We take the first image for each step.
                        image_path_curr = image_paths_curr[step_id][0]
                        element = (text, image_path_curr)

                story_seq.append(element)

            if len(story_seq) < self.min_story_length + 1:
                pass
            else:
                story_seq = story_seq[:self.max_story_length+1]
                
                curr_story_seq_len = len(story_seq)
                if self.multiref_gt:
                    story_seq = {
                        "story_seq": story_seq,
                        "multiref_gt": data_raw["multiref_gt"]
                    }

                # TODO: relax this.
                if (curr_story_seq_len >= self.min_story_length + 1
                    and curr_story_seq_len <= self.max_story_length + 1) and not self.multiref_gt:
                    story_seqs.append(story_seq)
                else:
                    story_seqs.append(story_seq)

        print("There are {} valid story sequences in {}".format(
              len(story_seqs), json_path))

        return story_seqs

    def _create_examples(self, lines):
        """Creates examples for the training, dev and test sets."""
        paired_examples = []
        for story_seq in lines:
            if self.multiref_gt:
                multiref_gt = story_seq["multiref_gt"]
                story_seq = story_seq["story_seq"]
            else:
                multiref_gt = None
            story_id = story_seq.pop(0)
            len_seq = len(story_seq)
            for i in range(0, len_seq):
                for j in range(0, len_seq):
                    if i == j:
                        continue
                    if self.order_criteria == "tight":
                        if j == i + 1:
                            label = "ordered"
                        else:
                            label = "unordered"
                    elif self.order_criteria == "loose":
                        if j > i:
                            label = "ordered"
                        else:
                            label = "unordered"
                    guid = "{}_{}{}".format(story_id, i+1, j+1)
                    text_a = story_seq[i][0]
                    text_b = story_seq[j][0]
                    img_path_a = story_seq[i][1]
                    img_path_b = story_seq[j][1]
                    distance = abs(j - i)
                    example = InputPairWiseExample(guid=guid, text_a=text_a,
                                                   text_b=text_b, label=label,
                                                   img_path_a=img_path_a,
                                                   img_path_b=img_path_b,
                                                   distance=distance,
                                                   multiref_gt=multiref_gt)
                    paired_examples.append(example)
        return paired_examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="train")
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="val")
        return self._create_examples(lines)

    def get_test_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="test")
        return self._create_examples(lines)


class RecipeQAAbductiveProcessor(RecipeQAPairWiseProcessor):
    """Processor for RecipeQA Steps Dataset, abductive data.
    Args:
        data_dir: string. Root directory for the dataset.
        pred_method: the method of the predictions, can be binary or
            contrastive 
        paired_with_image: will only consider sequence that have perfect image
            pairings.
        min_story_length: minimum length of sequence for each.
        max_story_length: maximum length of sequence for each.
    """

    def __init__(self, data_dir=None, pred_method="binary",
                 paired_with_image=True,
                 min_story_length=5, max_story_length=5,
                 version_text=None,
                 caption_transforms=None, **kwargs):
        """Init."""
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = RECIPEQA_DATA_ROOT
        assert pred_method in ["binary", "contrastive"]
        self.pred_method = pred_method
        self.paired_with_image = paired_with_image

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        self.caption_transforms = caption_transforms
        self.version_text = version_text

        self.multiref_gt = False

    def _create_examples(self, lines):
        """Creates examples for the training, dev and test sets."""
        abd_examples = []
        for story_seq in lines:
            if self.multiref_gt:
                multiref_gt = story_seq["multiref_gt"]
                story_seq = story_seq["story_seq"]
            else:
                multiref_gt = None
            story_id = story_seq.pop(0)
            len_seq = len(story_seq)
            for i in range(0, len_seq-2):
                all_seq_idx = set(list(range(len_seq)))
                curr_seq_idx = set(list(range(i, i+3)))
                left_seq_idx = list(all_seq_idx - curr_seq_idx)
                curr_seq_idx = list(curr_seq_idx)

                for k in left_seq_idx:
                    abd_idx = [curr_seq_idx[0]] + [k] + [curr_seq_idx[1]]
                    text_h1 = story_seq[abd_idx[0]][0]
                    text_h2 = story_seq[abd_idx[1]][0]
                    text_h3 = story_seq[abd_idx[2]][0]
                    img_path_h1 = story_seq[abd_idx[0]][1]
                    img_path_h2 = story_seq[abd_idx[1]][1]
                    img_path_h3 = story_seq[abd_idx[2]][1]
                    if self.pred_method == "binary":
                        label = "unordered"
                    guid = "{}_{}{}{}".format(story_id, abd_idx[0],
                                              abd_idx[1], abd_idx[2])
                    example = InputAbductiveExample(guid=guid, label=label,
                                                    text_h1=text_h1,
                                                    text_h2=text_h2,
                                                    text_h3=text_h3,
                                                    img_path_h1=img_path_h1,
                                                    img_path_h2=img_path_h2,
                                                    img_path_h3=img_path_h3,
                                                    multiref_gt=multiref_gt)
                    abd_examples.append(example)

                abd_idx = curr_seq_idx
                text_h1 = story_seq[abd_idx[0]]
                text_h2 = story_seq[abd_idx[1]]
                text_h3 = story_seq[abd_idx[2]]
                img_path_h1 = story_seq[abd_idx[0]][1]
                img_path_h2 = story_seq[abd_idx[1]][1]
                img_path_h3 = story_seq[abd_idx[2]][1]
                if self.pred_method == "binary":
                    label = "ordered"
                guid = "{}_{}{}{}".format(story_id, abd_idx[0],
                                          abd_idx[1], abd_idx[2])
                example = InputAbductiveExample(guid=guid, label=label,
                                                text_h1=text_h1,
                                                text_h2=text_h2,
                                                text_h3=text_h3,
                                                img_path_h1=img_path_h1,
                                                img_path_h2=img_path_h2,
                                                img_path_h3=img_path_h3,
                                                multiref_gt=multiref_gt)
                abd_examples.append(example)
        return abd_examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="train")
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="val")
        return self._create_examples(lines)

    def get_test_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="test")
        return self._create_examples(lines)


class RecipeQAGeneralProcessor(RecipeQAPairWiseProcessor):
    """Processor for RecipeQA Steps Dataset, general sorting prediction.
    Args:
        data_dir: string. Root directory for the dataset.
        paired_with_image: will only consider sequence that have perfect image
            pairings.
        min_story_length: minimum length of sequence for each.
        max_story_length: maximum length of sequence for each.
    """

    def __init__(self, data_dir=None, max_story_length=5, pure_class=False,
                 paired_with_image=True, min_story_length=5,
                 version_text=None,
                 caption_transforms=None, **kwargs):
        """Init."""
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = RECIPEQA_DATA_ROOT
        self.max_story_length = max_story_length
        self.pure_class = pure_class
        self.paired_with_image = paired_with_image

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        self.caption_transforms = caption_transforms
        self.version_text = version_text

        self.multiref_gt = False

    def get_labels(self):
        """See base class."""
        if self.pure_class:
            n = self.max_story_length
            fact = 1
            for i in range(1, n+1):
                fact = fact * i
            labels = [0 for i in range(fact)]
            return labels

        return list(range(self.max_story_length))

    def _create_examples(self, lines):
        """Creates examples for the training, dev and test sets."""
        head_examples = []
        for story_seq in lines:
            if self.multiref_gt:
                multiref_gt = story_seq["multiref_gt"]
                story_seq = story_seq["story_seq"]
            else:
                multiref_gt = None
            story_id = story_seq.pop(0)
            len_seq = len(story_seq)
            guid = story_id
            text_seq = [x[0] for x in story_seq]
            img_path_seq = [x[1] for x in story_seq]
            example = InputHeadExample(guid=guid, text_seq=text_seq,
                                       img_path_seq=img_path_seq,
                                       multiref_gt=multiref_gt)
            head_examples.append(example)
        return head_examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="train")
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="val")
        return self._create_examples(lines)

    def get_test_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="test")
        return self._create_examples(lines)


def human_annotated_to_test(data_dir, out_dir=None, train_ratio=0.9,
                            dev_ratio=0.05, test_ratio=0.05):
    random.seed(42)

    # Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--human_annotated_json_files',
        type=str,
        default=None,
        nargs="+",
        help='The jsonl files used for human annotations.'
    )
    parser.add_argument(
        '--human_annotated_version',
        type=str,
        default="human_annot",
        help='The name for the output files.'
    )
    args = parser.parse_args()

    # Read in the human jsonl files.
    assert args.human_annotated_json_files is not None
 
    human_annotated_dats = {}
    for human_annotated_json_file in args.human_annotated_json_files:
        inf = open(human_annotated_json_file, "r")
        cnt = 0
        for line in inf:
            datum = json.loads(line.strip())
            key = datum["guid"]
            human_annotated_dats[key] = datum
            cnt += 1
        pass
    pass

    # Read in data json files.
    json_paths = glob.glob(os.path.join(
        data_dir, "texts", "*.json"))
    line_cnt = 0
    human_line_cnt = 0
    train_data = []
    dev_data = []
    test_data = []
    human_data = []
    # recipe_id_humans = []
    for json_path in sorted(json_paths):
        print("Processing json file: {}".format(json_path))
        json_file = json.load(open(json_path))
        data_curr = json_file["data"]
        for data_raw in tqdm(data_curr, total=len(data_curr)):
            recipe_id = data_raw["recipe_id"]
            if recipe_id in human_annotated_dats:  #  and recipe_id not in recipe_id_humans:
                human_line_cnt += 1
                human_data.append(data_raw)
                # recipe_id_humans.append(recipe_id)
            else:
                line_cnt += 1
                if "train" in json_path:
                    train_data.append(data_raw)
                elif "val" in json_path:
                    dev_data.append(data_raw)
                elif "test" in json_path:
                    test_data.append(data_raw)
    print("Total line counts: {}".format(line_cnt))
    print("Total human line counts: {}".format(human_line_cnt))
    # print(data[0]["recipe_id"])
    # random.shuffle(data)
    # print(data[0]["recipe_id"])

    # Data splits.
    assert train_ratio + dev_ratio + test_ratio == 1.0, ("Split rations do"
        " not sum up to 1!")
    if out_dir is None:
        out_dir = data_dir

    """
    train_cnt = int(float(line_cnt) * train_ratio)
    dev_cnt = int(float(line_cnt) * dev_ratio)
    train_data = data[:train_cnt]
    dev_data = data[train_cnt:train_cnt+dev_cnt]
    test_data = data[train_cnt+dev_cnt:]
    """

    train_urls = [d["recipe_id"] for d in train_data]
    dev_urls = [d["recipe_id"] for d in dev_data]
    test_urls = [d["recipe_id"] for d in test_data]
    human_urls = [d["recipe_id"] for d in human_data]

    for sets in [dev_urls, test_urls, human_urls]:
        for url in sets:
            assert url not in train_urls, "recipe_id: {} is in train!"

    train_urls = list(set(train_urls))
    dev_urls = list(set(dev_urls))
    test_urls = list(set(test_urls))
    human_urls = list(set(human_urls))

    print("RecipeQA Train:  {}".format(len(train_urls)))
    print("RecipeQA Dev:    {}".format(len(dev_urls)))
    print("RecipeQA Test:   {}".format(len(test_urls)))
    test_data += human_data
    print("RecipeQA Test-H: {}".format(len(human_urls)))

    train_data = {"version": 0.9, "data": train_data}
    dev_data = {"version": 0.9, "data": dev_data}
    test_data = {"version": 0.9, "data": test_data}
    human_data = {"version": 0.9, "data": human_data}

    with open(os.path.join(out_dir, "train-{}.json".format(args.human_annotated_version)), "w") as outf:
        json.dump(train_data, outf, indent=4)
    with open(os.path.join(out_dir, "val-{}.json".format(args.human_annotated_version)), "w") as outf:
        json.dump(dev_data, outf, indent=4)
    with open(os.path.join(out_dir, "test-{}.json".format(args.human_annotated_version)), "w") as outf:
        json.dump(test_data, outf, indent=4)
    with open(os.path.join(out_dir, "test-{}_only.json".format(args.human_annotated_version)), "w") as outf:
        json.dump(human_data, outf, indent=4)

    # Leave.
    # exit(-1)


def output_to_tsv(data_dir, out_dir):
    from trainers.caption_utils import CaptionTransformations
    args, task = None, "wikihow"

    caption_transforms = None
    caption_transforms = CaptionTransformations(args, task,
        # caption_transformation_list=["remove_1st", "train_max_sentence_5"])
        caption_transformation_list=["train_max_sentence_5"])
    proc = RecipeQAGeneralProcessor(data_dir=data_dir,
                                    version_text="human_annot",
                                    caption_transforms=caption_transforms)
    caption_transforms = None
    caption_transforms = CaptionTransformations(args, task,
        # caption_transformation_list=["remove_1st", "eval_max_sentence_5"])
        caption_transformation_list=["eval_max_sentence_5"])
    proc_human = RecipeQAGeneralProcessor(data_dir=data_dir,
                                          version_text="human_annot_only",
                                          caption_transforms=caption_transforms)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_examples = proc.get_train_examples()
    dev_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    test_human_examples = proc_human.get_test_examples()

    all_examples = [
        ("train", train_examples),
        ("dev", dev_examples),
        ("test", test_examples),
        ("human_test", test_human_examples)
    ]

    for split, examples in all_examples:
        out_tsv = open(os.path.join(out_dir, "{}.tsv".format(split)), "w")
        if "test" in split:
            out_json = open(os.path.join(out_dir, "{}_examples.json".format(split)), "w")
        
        for example in tqdm(examples, desc="idx"):
            text_seq = example.text_seq
            new_sents = []
            for sent in text_seq:
                tokens = word_tokenize(sent.lower())
                new_sent = " ".join(tokens)
                new_sents.append(new_sent)
            new_text_seq = " <eos> ".join(new_sents)
            out_tsv.write(new_text_seq+"\n")

            d = {
                "url": example.guid
            }
            if "test" in split:
                out_json.write(json.dumps(d)+"\n")
            
        out_tsv.close()
        if "test" in split:
            out_json.close()
        print("Writing files to {}".format(os.path.join(
            out_dir, "{}.tsv".format(split))))

    # Leave.
    exit(-1)


if __name__ == "__main__":
    human_annotated_to_test("data/recipeQA", "data/recipeQA/new_splits")

    output_to_tsv(data_dir="data/recipeQA", out_dir="../prior_works/berson_roc/glue_data_new/recipeqa")

    proc = RecipeQAPairWiseProcessor()
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print(test_examples[0])
    print()

    proc = RecipeQAGeneralProcessor()
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print(test_examples[0])
    rand_idx = np.random.randint(0, len(test_examples))
    selected_example = test_examples[rand_idx]
    print("\nData: {}".format(rand_idx))
    print("-"*50)
    for i in range(len(selected_example.text_seq)):
        print(selected_example.text_seq[i])
        print("-"*50)
    raise
    print()
    
    proc = RecipeQAAbductiveProcessor()
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print(test_examples[0])
    print()
