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
from nltk.tokenize import sent_tokenize, word_tokenize


logger = logging.getLogger(__name__)

 
class CaptionTransformations(object):

    def __init__(self, args=None, task=None, caption_transformation_list=None):
        assert task is not None

        self.args = args
        self.task = task

        # TODO: In the future maybe we will have task-specific
        # transformations.
        if "recipeqa" in self.task:
            pass  # Do something here.

        if caption_transformation_list is not None:
            if len(caption_transformation_list) == 0:
                logging.info("No valid caption transformations!")
                return None

        self.transform_funcs = []
        logging.info("Using caption transformations: {}".format(
            caption_transformation_list))
        logging.warning("Notice that caption transformations "
                        "is order sensitive!")
        for transform_method in caption_transformation_list:
            if transform_method == "remove_1st":
                self.transform_funcs.append(self._remove_1st_func)
            elif "max_sentence" in transform_method:
                self.max_sentence = int(transform_method.split("max_sentence_")[-1])
                self.transform_funcs.append(self._cap_sentence_func)
            else:
                raise NotImplementedError("Caption transformation "
                    "method: {} not done yet!".format(transform_method))

        if (self.transform_funcs) == 0:
            logging.info("No valid caption transformations!")
            return None

    def transform(self, captions):
        if type(captions) == str:
            captions = self.transform_single_caption(captions)
            return captions

        new_captions = []
        for caption in captions:
            caption = self.transform_single_caption(caption)
            new_captions.append(caption)

        return new_captions

    def transform_single_caption(self, caption):
        for transform_func in self.transform_funcs:
            caption = transform_func(caption)
        return caption

    def _cap_sentence_func(self, caption):
        sents = sent_tokenize(caption)
        sents_cap = " ".join(sents[:self.max_sentence])
        caption = sents_cap
        return caption

    def _remove_1st_func(self, caption):
        sents = sent_tokenize(caption)
        if len(sents) > 1:
            sents_wo_1st = sents[1:]
            new_caption = " ".join(sents_wo_1st)
            caption = new_caption

        return caption

    
if __name__ == "__main__":
    class dummy_args(object):
        def __init__(self):
            self.device = "cpu"
            self.task_name = "recipeqa"

    pass
