import csv
import glob
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm

from dataclasses import dataclass


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


@dataclass
class InputPairWiseExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        pairID: (Optional) string. Unique identifier for the pair of sentences.
        distance: (Optional) int, distance of the pair.
        img_path_a: (Optional) string. The path to the image for text_a.
        img_path_b: (Optional) string. The path to the image for text_b.
        task_id: (Optional) int. The integer id to each dataset (task).
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    pairID: Optional[str] = None
    distance: Optional[int] = None
    img_path_a: Optional[str] = None
    img_path_b: Optional[str] = None
    task_id: Optional[int] = None
    multiref_gt: Optional[list] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class InputAbductiveExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_h1: string. The untokenized text of the premise sequence. For single
            sequence tasks, only this sequence must be specified.
        text_h2: (Optional) string. The untokenized text of the explain sequence.
            Only must be specified for sequence pair tasks.
        text_h3: (Optional) string. The untokenized text of the entail sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        pairID: (Optional) string. Unique identifier for the pair of sentences.
        img_path_h1: (Optional) string. The path to the image for text_h1.
        img_path_h2: (Optional) string. The path to the image for text_h2.
        img_path_h3: (Optional) string. The path to the image for text_h3.
        task_id: (Optional) int. The integer id to each dataset (task).
    """

    guid: str
    text_h1: str
    text_h2: str
    text_h3: str
    label: Optional[str] = None
    pairID: Optional[str] = None
    img_path_h1: Optional[str] = None
    img_path_h2: Optional[str] = None
    img_path_h3: Optional[str] = None
    task_id: Optional[int] = None
    multiref_gt: Optional[list] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class InputHeadExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_seq: list of strings. The untokenized text of the story.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        pairID: (Optional) string. Unique identifier for the pair of sentences.
        img_path_seq: list of strings. List of image paths corresponding to
            each text item in text_seq.
        task_id: (Optional) int. The integer id to each dataset (task).
    """

    guid: str
    text_seq: list
    label: Optional[str] = None
    pairID: Optional[str] = None
    img_path_seq: Optional[list] = None
    task_id: Optional[int] = None
    multiref_gt: Optional[list] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputPairWiseFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        pairID: (Optional) Unique identifier for the pair of sentences.
        images: (Optional)
        task_id: (Optional) int. The integer id to each dataset (task).
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    pairID: Optional[int] = None
    images: Optional[List[Union[int, float]]] = None
    task_id: Optional[int] = None


class Permutation(object):

   def nextPermutation(self, nums):
      found = False
      i = len(nums)-2
      while i >=0:
         if nums[i] < nums[i+1]:
            found =True
            break
         i-=1
      if not found:
         nums.sort()
      else:
         m = self.findMaxIndex(i+1,nums,nums[i])
         nums[i],nums[m] = nums[m],nums[i]
         nums[i+1:] = nums[i+1:][::-1]
      return nums

   def findMaxIndex(self,index,a,curr):
      ans = -1
      index = 0
      for i in range(index,len(a)):
         if a[i]>curr:
            if ans == -1:
               ans = curr
               index = i
            else:
               ans = min(ans,a[i])
               index = i
      return index
