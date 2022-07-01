import os
import sys
import json
import csv
import glob
import numpy as np
import random
import argparse
from tqdm import tqdm
from .utils import DataProcessor
from .utils import InputPairWiseExample, InputHeadExample, InputAbductiveExample
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# TODO: change.
WIKIHOW_DATA_ROOT = "data/wikihow"

IMAGE_FIELD_NAMES = [
    "image-large",
    "image-src-1",
]


class WikiHowPairWiseProcessor(DataProcessor):
    """Processor for WikiHow Steps Dataset, pair-wise data.
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
                 caption_transforms=None, **kwargs):
        """Init."""
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = WIKIHOW_DATA_ROOT
        assert order_criteria in ["tight", "loose"]
        self.order_criteria = order_criteria
        self.paired_with_image = paired_with_image

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        self.caption_transforms = caption_transforms
        
        if "version_text" in kwargs:
            self.version_text = kwargs["version_text"]
        else:
            self.version_text = None

        self.multiref_gt = False

    def get_labels(self):
        """See base class."""
        return ["unordered", "ordered"]  # 0: unordered, 1: ordered.

    def _read_json(self, data_dir=None, split="train"):
        """Reads in json lines to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        if self.version_text is not None:
            json_path = os.path.join(data_dir, "wikihow-{}-".format(self.version_text)+split+".json")
            if not os.path.exists(json_path):
                raise ValueError("File: {} not found!".format(json_path))
        else:
            json_path = os.path.join(data_dir, "wikihow-"+split+".json")
        print("Using {}".format(json_path))

        line_cnt = 0
        json_file = open(json_path)
        data = []
        for line in json_file:
            d = json.loads(line.strip())
            line_cnt += 1
            data.append(d)

        story_seqs = []
        missing_images = []

        used_wikihow_ids = {}

        # TODO: consistency of human test sets.
        if self.version_text is not None and self.version_text == "human_annot_only_filtered":
            human_check_dict = {}
            human_json = "data/wikihow/wikihow_human_studies_picked.jsonl"
            human_f = open(human_json)
            for line in human_f:
                dd = json.loads(line.strip())
                check_key = dd["steps"][0]["text"].split(".")[0]
                human_check_dict[check_key] = True

        # Each element in a story seq is (text, image) tuple.
        for data_raw in tqdm(data, total=len(data)):
            
            # Form the data id.
            wikihow_url = data_raw["url"]
            title_text = data_raw["title"]
            summary_text = data_raw["summary"]
            # print(wikihow_url)

            wikihow_check_id = "###".join([wikihow_url, title_text])
            wikihow_check_id = wikihow_url

            # TODO: GUID using url and title for now.

            # print(wikihow_url, len(data_raw["sections"]))

            # Multi-reference GTs.
            if "multiref_gt" in data_raw:
                if not self.multiref_gt: self.multiref_gt = True

            for section_id in range(len(data_raw["sections"])):

                # if wikihow_check_id in used_wikihow_ids:
                #     continue
                # used_wikihow_ids[wikihow_check_id] = True

                section_curr = data_raw["sections"][section_id]
                wikihow_page_id = "###".join([wikihow_url, title_text, str(section_id)])
                wikihow_page_id = "###".join([wikihow_url, str(section_id)])
                story_seq = [wikihow_page_id]

                # TODO: consistency of human test sets.
                include_data = True
                if self.version_text is not None and self.version_text == "human_annot_only_filtered":
                    include_data = False

                for step_id in range(len(section_curr["steps"])):
                    step_curr = section_curr["steps"][step_id]
                    step_headline = step_curr["step_headline"]
                    step_text = step_curr["step_text"]["text"]
                    bullet_points = step_curr["step_text"]["bullet_points"]
                    # print(step_headline)
                    # print(step_text)
                    # print(bullet_points)
                    # combined_text = " ".join([step_text] + bullet_points)
                    # if step_headline is not None:
                    #     combined_text = " ".join([step_headline, step_text])
                    #     combined_text = step_headline
                    # else:
                    #     combined_text = step_text
                    # combined_text = step_text
                    combined_text = " ".join([step_text] + bullet_points)

                    if self.version_text is not None and self.version_text == "human_annot_only_filtered":
                        check_str = combined_text.split(".")[0]
                        if check_str in human_check_dict:
                            include_data = True

                    if self.caption_transforms is not None:
                        combined_text = self.caption_transforms.transform(combined_text)
                    
                    element = None
                    if self.paired_with_image:
                        # We take the first image for each step.
                        image_path_curr = None
                        for image_field_key in IMAGE_FIELD_NAMES:
                            if image_field_key in step_curr["step_assets"]:
                                image_path_curr = step_curr["step_assets"][
                                    image_field_key]
                                image_path_curr_new = None
                                if image_path_curr is not None and len(image_path_curr) > 0:
                                    image_path_curr = os.path.join(self.data_dir, image_path_curr)
                                    
                                    if "wikihow.com" not in image_path_curr:
                                        image_path_curr_new = image_path_curr.replace(
                                            "/images/", 
                                            "/www.wikihow.com/images/")
                                    else:
                                        image_path_curr_new = image_path_curr
                                    if not os.path.exists(image_path_curr_new):
                                        image_path_curr_new = image_path_curr.replace(
                                            "/images/", 
                                            "/wikihow.com/images/")
                                        if not os.path.exists(image_path_curr_new):
                                            missing_images.append(wikihow_page_id+"###"+str(step_id))
                                            element = None
                                        else:
                                            element = (combined_text, image_path_curr_new)
                                    else:
                                        element = (combined_text, image_path_curr_new)
                                else:
                                    missing_images.append(wikihow_page_id+"###"+str(step_id))
                                    element = None
                                if image_path_curr_new is not None and os.path.exists(image_path_curr_new):
                                    break
                    else:
                        element = (combined_text, None)

                    if element is not None:
                        story_seq.append(element)

                # TODO: Currently different sections are in different
                # sequences for sorting.
                if len(story_seq) < self.min_story_length + 1:
                    pass
                elif not include_data:
                    pass
                else:
                    story_seq = story_seq[:self.max_story_length+1]
                    
                    curr_story_seq_len = len(story_seq)
                    if self.multiref_gt:
                        story_seq = {
                            "story_seq": story_seq,
                            "multiref_gt": data_raw["multiref_gt"]
                        }

                    # story_seqs.append(story_seq)
                    # TODO: relax this.
                    if (curr_story_seq_len >= self.min_story_length + 1
                        and curr_story_seq_len <= self.max_story_length + 1):
                        story_seqs.append(story_seq)

        print("[WARNING] Number of missing images in {}: {}".format(
            split, len(missing_images)))
        missing_image_paths_f = ("data/wikihow/"
            "missing_images_{}.txt".format(split))
        missing_image_paths_file = open(missing_image_paths_f, "w")
        for missing_image_path in missing_images:
            missing_image_paths_file.write(missing_image_path+"\n")
        missing_image_paths_file.close()
        print("          Saves at: {}".format(missing_image_paths_f))

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
        lines = self._read_json(data_dir=data_dir, split="dev")
        return self._create_examples(lines)

    def get_test_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="test")
        return self._create_examples(lines)


class WikiHowAbductiveProcessor(WikiHowPairWiseProcessor):
    """Processor for WikiHow Steps Dataset, abductive data.
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
                 caption_transforms=None, version_text=None):
        """Init."""
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = WIKIHOW_DATA_ROOT
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

    def get_labels(self):
        """See base class."""
        return ["unordered", "ordered"]  # 0: unordered, 1: ordered.

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
        lines = self._read_json(data_dir=data_dir, split="dev")
        return self._create_examples(lines)

    def get_test_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="test")
        return self._create_examples(lines)


class WikiHowGeneralProcessor(WikiHowPairWiseProcessor):
    """Processor for WikiHow Steps Dataset, general sorting prediction.
    Args:
        data_dir: string. Root directory for the dataset.
        paired_with_image: will only consider sequence that have perfect image
            pairings.
        min_story_length: minimum length of sequence for each.
        max_story_length: maximum length of sequence for each.
    """

    def __init__(self, data_dir=None, max_story_length=5, pure_class=False,
                 paired_with_image=True, min_story_length=5,
                 caption_transforms=None, version_text=None):
        """Init."""
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = WIKIHOW_DATA_ROOT
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
        lines = self._read_json(data_dir=data_dir, split="dev")
        return self._create_examples(lines)

    def get_test_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="test")
        return self._create_examples(lines)


# Get category mappings
def read_in_wikihow_categories(cat_path=None, cat_level=1):
    if cat_path is None:
        json_f = os.path.join(WIKIHOW_DATA_ROOT, "wikihow-categories-output.json")
    else:
        json_f = cat_path
    json_in = open(json_f, "r")
    url2cat = {}
    cat2url = {}
    for line in json_in:
        cat = json.loads(line.strip())
        url = cat["url"]
        categories = cat["categories"]
        if len(categories) - 1 >= cat_level:
            cat_level_desc = categories[cat_level]["category title"]
        elif len(categories) - 1 >= 1:
            cat_level_desc = categories[-1]["category title"]
        else:
            cat_level_desc = "Root"
        url2cat[url] = cat_level_desc
        if cat_level_desc not in cat2url:
            cat2url[cat_level_desc] = []
        cat2url[cat_level_desc].append(url)
    return url2cat, cat2url


if __name__ == "__main__":

    proc = WikiHowPairWiseProcessor(data_dir=WIKIHOW_DATA_ROOT)
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    proc = WikiHowPairWiseProcessor(data_dir=WIKIHOW_DATA_ROOT, version_text="human_annot_only")
    test_examples = proc.get_test_examples()
    print(test_examples[0])
    print()

    proc = WikiHowGeneralProcessor(data_dir=WIKIHOW_DATA_ROOT)
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
    print()
    
    proc = WikiHowAbductiveProcessor(data_dir=WIKIHOW_DATA_ROOT)
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print(test_examples[0])
    print()
