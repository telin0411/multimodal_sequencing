# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LXRT model."""

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from typing import Any, Dict, Tuple
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_roberta import RobertaClassificationHead
from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except Importtokenization:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

from param import VISUAL_CONFIG


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    is_composition: bool = False

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__(self.vocab_size).to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                key not in default_config_dict
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        return serializable_config_dict

    def to_json_file(self, json_file_path: str, use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def save_pretrained(self, save_directory: str):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info("Configuration saved in {}".format(output_config_file))


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""


class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output


class VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim
        pos_dim = VISUAL_CONFIG.visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        if isinstance(visn_input, tuple):
            feats, boxes = visn_input

            x = self.visn_fc(feats)
            x = self.visn_layer_norm(x)
            y = self.box_fc(boxes)
            y = self.box_layer_norm(y)
            output = (x + y) / 2

            output = self.dropout(output)
            return output
        else:
            feats = visn_input
            x = self.visn_fc(feats)
            x = self.visn_layer_norm(x)
            x = self.dropout(x)
            return x

def _cat_with_none(feat_1, feat_2, dim):
    if feat_1 is None:
        return feat_2
    if feat_2 is None:
        return feat_1
    return torch.cat((feat_1, feat_2), dim=dim)

def _split_with_none(lang_feats, visn_feats, joint_feats):
    if lang_feats is None:
        assert(visn_feats.size(1) == joint_feats.size(1))
        return None, joint_feats
    if visn_feats is None:
        assert(lang_feats.size(1) == joint_feats.size(1))
        return joint_feats, None
    return joint_feats[:, :lang_feats.size(1), :].contiguous(), joint_feats[:, lang_feats.size(1):, :].contiguous()


class LinearPositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.x_position_embedding = nn.Embedding(VISUAL_CONFIG.pos_num, VISUAL_CONFIG.visual_feat_dim)
        self.y_position_embedding = nn.Embedding(VISUAL_CONFIG.pos_num, VISUAL_CONFIG.visual_feat_dim)
        self.hidden_size = VISUAL_CONFIG.visual_feat_dim

    def forward(self, visn_feats, skip_last_layer=True, curr_img_len=None):
        if curr_img_len is None:
            curr_img_len = VISUAL_CONFIG.max_subsample_image_length
        # batch x 2048 x width x height
        if visn_feats.ndim == 3:
            width = (visn_feats.size(1) - 1) // curr_img_len
            width = math.sqrt(width)
            width = int(width)
        else:
            width = visn_feats.size(2)
        width_ids = torch.arange(width, dtype=torch.long, device=visn_feats.device)
        width_ids = width_ids.unsqueeze(0)
        x_embedding = self.x_position_embedding(width_ids).unsqueeze(-2) # 1 x width x 1 x 768

        if visn_feats.ndim == 3:
            height = (visn_feats.size(1) - 1) // curr_img_len
            height = math.sqrt(height)
            height = int(height)
        else:
            height = visn_feats.size(3)
        height_ids = torch.arange(height, dtype=torch.long, device=visn_feats.device)
        height_ids = height_ids.unsqueeze(0)
        y_embedding = self.y_position_embedding(height_ids).unsqueeze(-3) # 1 x 1 x height x 768

        position_embedding = x_embedding + y_embedding # 1 x width x heitht x 768
        position_embedding = position_embedding.view(1, -1, self.hidden_size)
        if curr_img_len > 1 and not skip_last_layer:
            position_embedding = torch.cat([position_embedding] * curr_img_len, dim=1)
            position_embedding = torch.cat([position_embedding[:, 0].unsqueeze(1), position_embedding], dim=1)
        if visn_feats.ndim != 3:
            visn_feats = visn_feats.permute(0, 2, 3, 1).view(visn_feats.size(0), -1, visn_feats.size(1))
        visn_feats += position_embedding
        return visn_feats


class VisualTokenTypeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_story_length = 5
        self.token_type_embedding = nn.Embedding(self.max_story_length, VISUAL_CONFIG.visual_feat_dim)
        self.hidden_size = VISUAL_CONFIG.visual_feat_dim

    def forward(self, visn_feats, skip_last_layer=True, curr_img_len=None):
        if curr_img_len is None:
            curr_img_len = VISUAL_CONFIG.max_subsample_image_length
        # batch x 2048 x width x height
        assert visn_feats.ndim == 3

        if skip_last_layer:
            single_img_len = visn_feats.size(1)

            type_ids = [[i] * single_img_len for i in range(curr_img_len)]
            type_ids = torch.Tensor(type_ids).type_as(visn_feats).long()
            type_ids = type_ids.reshape(-1)
            type_ids = type_ids.unsqueeze(0)

            type_embedding = self.token_type_embedding(type_ids).unsqueeze(-2) # 1 x width x 1 x 768

            type_embedding = type_embedding.view(1, -1, self.hidden_size)
            visn_feats = visn_feats.reshape(int(visn_feats.size(0)/curr_img_len),
                -1, visn_feats.size(2))
            visn_feats += type_embedding
        else:
            single_img_len = visn_feats.size(1) // curr_img_len
            type_ids = np.asarray([0] * visn_feats.size(1))
            start_pos = 1
            for i in range(curr_img_len):
                type_ids[start_pos:start_pos+int(single_img_len)] = i
                start_pos += int(single_img_len)
            type_ids = torch.Tensor(type_ids).type_as(visn_feats).long()
            type_ids = type_ids.unsqueeze(0)

            type_embedding = self.token_type_embedding(type_ids).unsqueeze(-2) # 1 x width x 1 x 768

            type_embedding = type_embedding.view(1, -1, self.hidden_size)
            visn_feats += type_embedding
           
        return visn_feats


def apply_visual_token_type_embedding_to_lang_feats(lang_feats, input_ids,
        cls_id, sep_id, vis_token_type_embedding):
    bz, seq_len = input_ids.size()
    _, _, lang_dim = lang_feats.size()
    for i in range(bz):
        input_id = input_ids[i]
        # print(input_id)
        # print((input_id==cls_id).nonzero().reshape(-1))
        # print((input_id==sep_id).nonzero().reshape(-1))
        type_id = torch.zeros(len(input_id))
        cls_pos = (input_id==cls_id).nonzero().reshape(-1)
        for j in range(len(cls_pos)):
            cls_pos_start = cls_pos[j]
            if j == len(cls_pos) - 1:
                type_id[cls_pos_start:] = j
            else:
                cls_pos_end = cls_pos[j+1]
                type_id[cls_pos_start:cls_pos_end] = j
        type_id = type_id.type_as(input_ids).long()
        type_embedding = vis_token_type_embedding(type_id).unsqueeze(-2)
        type_embedding = type_embedding.view(1, -1, type_embedding.size(-1))
        # print(type_id)
        # print(type_embedding.size())
        # print(lang_feats[i].size())
        lang_feats[i] += type_embedding[0][:, :lang_dim]
        # raise
    return lang_feats


class LXRTEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        if "multimodal_img_part" in kwargs:
            self.multimodal_img_part = kwargs["multimodal_img_part"]
            assert "cls_id" in kwargs
            self.cls_id = kwargs["cls_id"]
        else:
            self.multimodal_img_part = False

        if "multimodal_text_part" in kwargs:
            self.multimodal_text_part = kwargs["multimodal_text_part"]
            assert "cls_id" in kwargs
            self.cls_id = kwargs["cls_id"]
        else:
            self.multimodal_text_part = False

        if "clip_model_name" in kwargs:
            VISUAL_CONFIG.clip_model_name = kwargs["clip_model_name"]

        if "max_story_length" in kwargs:
            VISUAL_CONFIG.max_story_length = kwargs["max_story_length"]
            if "pretraining" in kwargs and kwargs["pretraining"] == True:
                VISUAL_CONFIG.max_subsample_image_length = 5  # Pretraining.
                VISUAL_CONFIG.max_subsample_image_length = 2  # Pretraining.
                VISUAL_CONFIG.max_subsample_text_length = 5 # Pretraining.
                VISUAL_CONFIG.max_subsample_text_length = 2 # Pretraining.
                if (kwargs["multimodal_pretrain_objectives"] is None
                    or len(kwargs["multimodal_pretrain_objectives"]) == 0):
                    VISUAL_CONFIG.max_subsample_image_length = 5
                    VISUAL_CONFIG.max_subsample_text_length = 5
            else:
                VISUAL_CONFIG.max_subsample_image_length = 2  # BERSON input.

        if "cls_id" in kwargs:
            self.cls_id = kwargs["cls_id"]
        else:
            self.cls_id = None

        if "sep_id" in kwargs:
            self.sep_id = kwargs["sep_id"]
        else:
            self.sep_id = None

        # self.skip_last_layer = True
        self.skip_last_layer = False

        self.visualbert_style = VISUAL_CONFIG.visualbert_style

        # Obj-level image embedding layer
        if not self.multimodal_text_part:
            self.visn_fc = VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers

        # Account for models other than bert-base
        self.num_l_layers = config.num_hidden_layers
        
        if self.multimodal_img_part:
            pass  # Do nothing.
        elif self.visualbert_style:
            layers = [BertLayer(config) for _ in range(self.num_l_layers)]
            self.layer = nn.ModuleList(layers)
            print("\n\nVisualBERT style: {} layers".format(len(self.layer)))
        else:
            # Layers
            # Using self.layer instead of self.l_layer to support loading BERT weights.
            print("LXRT encoder with %d l_layers, %d x_layers, and %d r_layers." %
                (self.num_l_layers, self.num_x_layers, self.num_r_layers))
            
            self.layer = nn.ModuleList(
                [BertLayer(config) for _ in range(self.num_l_layers)]
            )
            self.x_layers = nn.ModuleList(
                [LXRTXLayer(config) for _ in range(self.num_x_layers)]
            )
            self.r_layers = nn.ModuleList(
                [BertLayer(config) for _ in range(self.num_r_layers)]
            )

        if not self.multimodal_text_part:
            if VISUAL_CONFIG.use_clip:
                from .visual_transformers import initialize_clip
                self.visual_model = initialize_clip(VISUAL_CONFIG,
                    img_len=VISUAL_CONFIG.max_subsample_image_length,
                    img_only=self.multimodal_img_part)
            elif VISUAL_CONFIG.use_vit:
                from .visual_transformers import initialize_vit
                self.visual_model = initialize_vit(VISUAL_CONFIG)
            
            if VISUAL_CONFIG.use_positional_embedding:
                self.visual_pos = LinearPositionEmbedding(config)
            if VISUAL_CONFIG.use_token_type_embedding:
                self.visual_token_type = VisualTokenTypeEmbedding(config)
            if VISUAL_CONFIG.use_max_pooling:
                self.max_pooling = nn.MaxPool2d(2, stride=2)

    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None, input_ids=None,
                pretraining_objective=None):

        self.no_visual = False
        if type(visn_feats) == tuple:
            if visn_feats[0] is None:
                self.no_visual = True
        else:
            if visn_feats is None:
                self.no_visual = True

        if VISUAL_CONFIG.vilt_style:
            assert(not VISUAL_CONFIG.freeze_clip)
            if VISUAL_CONFIG.use_clip:
                if type(visn_feats) is not tuple:
                    images = visn_feats
                else:
                    images, boxes = visn_feats
                lang_attention_mask = lang_attention_mask.squeeze(1).squeeze(1)
                lang_attention_mask[lang_attention_mask!=0] = float("-inf")

                joint_feats = self.visual_model.visual(images.type(self.visual_model.dtype), skip_last_layer=True, text_embedding = lang_feats, text_mask=lang_attention_mask)
                return _split_with_none(lang_feats, images, joint_feats)
            elif VISUAL_CONFIG.use_vit:
                if type(visn_feats) is not tuple:
                    images = visn_feats
                else:
                    images, boxes = visn_feats
                joint_feats = self.visual_model(
                    images,
                    return_features=True,
                    text_embedding=lang_feats,
                    text_mask=lang_attention_mask)
                return _split_with_none(lang_feats, images, joint_feats)

        if VISUAL_CONFIG.use_clip:
            if not self.multimodal_text_part and not self.no_visual:
                if type(visn_feats) is not tuple:
                    images = visn_feats
                else:
                    images, boxes = visn_feats

                curr_img_len = images.size(0) // lang_feats.size(0)
                visn_feats = self.visual_model.visual(images.type(self.visual_model.dtype), skip_last_layer=self.skip_last_layer, img_len=curr_img_len)

                ### Patch-based pretraining. ###
                if (pretraining_objective is not None
                    and "patch_based_image" in pretraining_objective):
                    img_len = visn_feats.size(1)
                    patch_len = img_len // VISUAL_CONFIG.max_subsample_image_length

                    all_cls_pos = list(range(1, img_len, patch_len))
                    all_cls_pos.pop(0)
                    all_cls_pos.insert(0, 0)

                    bz = lang_feats.size(0)
                    selected_patches_bz_i = []
                    selected_patches_bz_j = []
                    num_sub_samples_curr = np.random.randint(0, patch_len)
                    for i in range(bz):
                        selected_patches_i = []
                        selected_patches_j = []
                        if pretraining_objective == "patch_based_image_swapping":
                            num_sub_samples_curr = np.random.randint(0, patch_len)
                        for j in range(len(all_cls_pos)):
                            if j == len(all_cls_pos) - 1:
                                end_idx = img_len
                            else:
                                end_idx = all_cls_pos[j+1]
                            start_idx = all_cls_pos[j]
                            patch_range = list(range(start_idx, end_idx))
                            # print(patch_range)
                            if True:
                                patch_idx_i = np.random.choice(patch_range, num_sub_samples_curr, replace=False)
                                # print(patch_idx_i, num_sub_samples_curr)
                                patch_idx_j = np.random.choice(patch_range, num_sub_samples_curr, replace=False)
                                # print(patch_idx_j, num_sub_samples_curr)
                                selected_patches_i.append(patch_idx_i)
                                selected_patches_j.append(patch_idx_j)
                        selected_patches_bz_i.append(selected_patches_i)
                        selected_patches_bz_j.append(selected_patches_j)

                    if pretraining_objective == "patch_based_image_swapping":
                        pretraining_objective_labels = []
                        for i in range(bz):
                            swap_prob = np.random.rand()
                            if swap_prob > 0.5:
                                swap_steps = np.random.choice(VISUAL_CONFIG.max_subsample_image_length, 2, replace=False)
                                swap_steps = sorted(swap_steps)
                                # print(swap_steps)
                                swap_x, swap_y = swap_steps[0], swap_steps[1]
                                swap_patches_x = selected_patches_bz_i[i][swap_x]
                                swap_patches_y = selected_patches_bz_i[i][swap_y]
                                # print(swap_patches_x)
                                # print(swap_patches_y)
                                visn_feats_y = visn_feats[i][swap_patches_y]
                                visn_feats[i][swap_patches_y] = visn_feats[i][swap_patches_x]
                                visn_feats[i][swap_patches_x] = visn_feats_y
                                pretraining_objective_labels.append(0)
                            else:
                                pretraining_objective_labels.append(1)
                        pretraining_objective_labels = torch.Tensor(pretraining_objective_labels)
                        pretraining_objective_labels = pretraining_objective_labels.type_as(visn_feats).long()
                        # print(pretraining_objective_labels);raise
                    elif pretraining_objective == "patch_based_image_sequence_predictions":
                        raise NotImplementedError("Not done yet!")
                    else:
                        raise NotImplementedError("Not done yet!")

                elif (pretraining_objective is not None
                      and "mrm_classification" in pretraining_objective):
                    # print(pretraining_objective)
                    mask_num = int(pretraining_objective.split("_")[-1])
                    # print(mask_num)
                    assert visn_feats.ndim == 3
                    # print(visn_feats.size())
                    visn_feat_len = visn_feats.size(1)
                    visn_per_seq_len = visn_feat_len // VISUAL_CONFIG.max_subsample_image_length
                    bz = visn_feats.size(0)

                    pretraining_objective_labels = []
                    pretraining_objective_mask = torch.zeros((bz, visn_feat_len)).long()
                    visn_feats_gt = visn_feats.detach().clone()

                    def overlap_too_much(m1, m2):
                        m1, m2 = sorted(m1), sorted(m2)
                        overlap_cnt = 0
                        for m2_idx in m2:
                            if m2_idx in m1:
                                overlap_cnt += 1
                        overlap_ratio = float(overlap_cnt) / len(m2)
                        if overlap_ratio > 0.6:
                            return True
                        return False

                    for i in range(bz):
                        start_end = list(range(1, visn_feat_len, visn_per_seq_len))
                        curr_batch_gt = None
                        sampled_indices = []
                        for j in start_end:
                            start_idx = j
                            end_idx = j + visn_per_seq_len
                            idx_choices = list(range(start_idx, end_idx))
                            to_mask = np.random.choice(idx_choices, mask_num, replace=False)
                            if len(sampled_indices) == 0:
                                sampled_indices.append(to_mask)
                            else:
                                while overlap_too_much(sampled_indices[-1], to_mask):
                                    to_mask = np.random.choice(idx_choices, mask_num, replace=False)
                                sampled_indices.append(to_mask)
                            to_mask = sorted(to_mask)
                            curr_visn_feats_gt = visn_feats_gt[i, to_mask]
                            if curr_batch_gt is None:
                                curr_batch_gt = curr_visn_feats_gt
                            else:
                                curr_batch_gt = torch.cat([curr_batch_gt, curr_visn_feats_gt], dim=0)

                            # Mask images.
                            visn_feats[i, to_mask] = 0
                            pretraining_objective_mask[i, to_mask] = 1

                        pretraining_objective_labels.append(curr_batch_gt)

                    pretraining_objective_labels = torch.stack(pretraining_objective_labels)
                    pretraining_objective_labels = self.visn_fc(pretraining_objective_labels)
                    # print(pretraining_objective_labels.size())
                    # print(pretraining_objective_mask.size())
                    # print(pretraining_objective_mask[-1])
                    pretraining_objective_labels = (pretraining_objective_labels,
                        pretraining_objective_mask)
                    # raise
                elif pretraining_objective is not None:
                    raise NotImplementedError("Not done yet!")
                ### End of patch-based pretraining. ###

                if "RN" in VISUAL_CONFIG.clip_model_name:
                    if VISUAL_CONFIG.use_max_pooling:
                        visn_feats = self.max_pooling(visn_feats)

                    if VISUAL_CONFIG.use_positional_embedding:
                        visn_feats = self.visual_pos(visn_feats, skip_last_layer=self.skip_last_layer, curr_img_len=curr_img_len)
                    elif visn_feats.ndim == 3:
                        pass  # do nothing.
                    else:
                        visn_feats = visn_feats.permute(0, 2, 3, 1).view(visn_feats.size(0), -1, visn_feats.size(1))

                    if VISUAL_CONFIG.use_token_type_embedding:
                        visn_feats = self.visual_token_type(visn_feats, skip_last_layer=self.skip_last_layer, curr_img_len=curr_img_len)
                    elif visn_feats.ndim == 3:
                        pass  # do nothing.
                    else:
                        visn_feats = visn_feats.permute(0, 2, 3, 1).view(visn_feats.size(0), -1, visn_feats.size(1))

                # Cast back to fp32
                visn_feats = visn_feats.to(dtype=next(self.visn_fc.parameters()).dtype)
            elif VISUAL_CONFIG.use_vit:
                assert not self.multimodal_text_part and not self.no_visual
                if type(visn_feats) is not tuple:
                    images = visn_feats
                else:
                    images, boxes = visn_feats
                visn_feats = self.visual_model(images, return_features=True)
                visn_feats = visn_feats.to(dtype=next(self.visn_fc.parameters()).dtype)
            elif VISUAL_CONFIG.drop_boxes:
                assert not self.multimodal_text_part and not self.no_visual
                if type(visn_feats) is not tuple:
                    images = visn_feats
                else:
                    images, boxes = visn_feats
            
            if VISUAL_CONFIG.sub_sampling:
                assert not self.multimodal_text_part and not self.no_visual
                # visn_feats: batch x seq_len x 768

                sub_feat_num = VISUAL_CONFIG.sub_feat_num
                sampled_index = []
                for i in range(visn_feats.size(0)):
                    sampled_index.append(torch.from_numpy(np.random.choice(visn_feats.size(1), sub_feat_num, replace=False)))
                sampled_index = torch.stack(sampled_index, dim=0).unsqueeze(-1).expand(visn_feats.size(0), sub_feat_num, visn_feats.size(2)).long().to(visn_feats.device)  # batch x sub_feat_num x 768?
                visn_feats = torch.gather(visn_feats, 1, sampled_index)
                
            # Run visual embedding layer
            # Note: Word embedding layer was executed outside this module.
            #       Keep this design to allow loading BERT weights.
            if not self.multimodal_text_part and not self.no_visual:
                visn_feats = self.visn_fc(visn_feats)

        if self.multimodal_img_part and not self.no_visual:
            if pretraining_objective is not None:
                return pretraining_objective_labels, (None, visn_feats)
            return None, visn_feats

        if visn_attention_mask is None and not self.multimodal_text_part and not self.no_visual:
            visn_attention_mask = torch.zeros(visn_feats.size(0), visn_feats.size(1)).to(
                dtype=next(self.visn_fc.parameters()).dtype).to(next(self.visn_fc.parameters()).device)
            visn_attention_mask = visn_attention_mask.unsqueeze(1).unsqueeze(2)

        if VISUAL_CONFIG.visualbert_style:
            
            if False and not self.multimodal_text_part:
                lang_feats = apply_visual_token_type_embedding_to_lang_feats(
                    lang_feats, input_ids, self.cls_id, self.sep_id, 
                    self.visual_token_type.token_type_embedding)
            # print(visn_feats.size())
            # print(input_ids[0], self.cls_id, self.sep_id, lang_feats.size(), input_ids.size());raise
            if not self.multimodal_text_part and not self.no_visual:
                visn_feats = visn_feats.reshape(lang_feats.size(0), -1, visn_feats.size(2))
                visn_attention_mask = visn_attention_mask.reshape(lang_attention_mask.size(0), 1, 1, -1)
                visn_attention_mask = visn_attention_mask[:, 0:1, :, :]

            if self.multimodal_text_part or self.no_visual:
                visn_feats = None
                visn_attention_mask = None

            joint_feats = _cat_with_none(lang_feats, visn_feats, dim=1) #torch.cat((lang_feats, visn_feats), dim=1)
            joint_mask = _cat_with_none(lang_attention_mask, visn_attention_mask, dim=-1)  #torch.cat((lang_attention_mask, visn_attention_mask), dim=-1)
            all_attention_weights = []
            for layer_module in self.layer:
                #if args.get("output_attention", False):
                #    joint_feats, attention_weights = layer_module(joint_feats, joint_mask)
                #    all_attention_weights.append(attention_weights)
                #else:
                joint_feats = layer_module(joint_feats, joint_mask)

            #if args.get("output_attention", False):
            #    return _split_with_none(lang_feats, visn_feats, joint_feats), all_attention_weights
            if pretraining_objective is not None:
                return pretraining_objective_labels, _split_with_none(lang_feats, visn_feats, joint_feats)
            return _split_with_none(lang_feats, visn_feats, joint_feats)
        else:
            # Run language layers
            for layer_module in self.layer:
                lang_feats = layer_module(lang_feats, lang_attention_mask)

            # Run relational layers
            for layer_module in self.r_layers:
                visn_feats = layer_module(visn_feats, visn_attention_mask)

            # Run cross-modality layers
            for layer_module in self.x_layers:
                lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask,
                                                    visn_feats, visn_attention_mask)

        return lang_feats, visn_feats


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        #pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class BertVisualObjHead(nn.Module):
    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Decide the use of visual losses
        visual_losses = visual_losses.split(",")
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict({
            key: nn.Linear(config.hidden_size, VISUAL_CONFIG.visual_loss_config[key][0])
            for key in self.visual_losses
        })

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path == 'bert-base-uncased':
                try:
                    print("The BERT-weight-downloading query to AWS was time-out;" 
                          "trying to download from UNC servers")
                    archive_file = "https://nlp.cs.unc.edu/data/bert/bert-base-uncased.tar.gz"
                    resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
                except EnvironmentError:
                    print("The weight-downloading still crashed with link: %s, "
                          "please check your network connection" % archive_file)
                    return None
            else:
                logger.error(
                        "Model name '{}' was not found in model name list ({}). "
                        "We assumed '{}' was a path or url but couldn't find any file "
                        "associated to this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                            archive_file))
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
            # print(key, key.replace("roberta.", "") in model.state_dict())
        # raise
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        elif not hasattr(model, 'bert') and any(s.startswith('roberta.') for s in state_dict.keys()):
            start_prefix = 'roberta.'
        # """
        elif hasattr(model, 'bert') and any(s.startswith('roberta.') for s in state_dict.keys()):

            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if key == "lm_head.bias":
                    new_key = "cls.predictions.bias"
                elif "lm_head.dense" in key:
                    new_key = key.replace("lm_head.dense", "cls.predictions.transform.dense")
                elif "lm_head.layer_norm" in key:
                    new_key = key.replace("lm_head.layer_norm", "cls.predictions.transform.LayerNorm")
                elif "lm_head.decoder" in key:
                    new_key = key.replace("lm_head.decoder", "cls.predictions.decoder")
                else:
                    new_key = key.replace("roberta", "bert")
                assert new_key in model.state_dict()

                # print(key, new_key, new_key in model.state_dict())
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)

            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # for k, v in state_dict.items():
            #     print(k, k in model.state_dict(), v.shape)
            # for k in state_dict.keys():
            #     print(k, k.replace("roberta", "bert") in model.state_dict(), state_dict[k].shape)
            # print()
            # for k, v in model.state_dict().items():
            #     print(k, v.shape)
        else:
            pass
            # raise NotImplementedError("Needs special care for this model!")
        # """

        load(model, prefix=start_prefix)

        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        WEIGHTS_NAME = "pytorch_model.bin"
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)


class LXRTModel(BertPreTrainedModel):
    """LXRT Model."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTEncoder(config, **kwargs)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

        print(kwargs)

        if "multimodal_img_part" in kwargs:
            self.multimodal_img_part = kwargs["multimodal_img_part"]
            assert "cls_id" in kwargs
            self.cls_id = kwargs["cls_id"]
        else:
            self.multimodal_img_part = False

        if "multimodal_text_part" in kwargs:
            self.multimodal_text_part = kwargs["multimodal_text_part"]
            assert "cls_id" in kwargs
            self.cls_id = kwargs["cls_id"]
        else:
            self.multimodal_text_part = False

        if "multimodal_pretrain_objectives" in kwargs:
            self.multimodal_pretrain = True
            if "image_swapping" in kwargs["multimodal_pretrain_objectives"]:
                self.image_swapping = True
                self.image_swapping_mlp = nn.Linear(config.hidden_size, 2)
            else:
                self.image_swapping = False
        else:
            self.multimodal_pretrain = False

        if "cls_id" in kwargs:
            self.cls_id = kwargs["cls_id"]
        else:
            self.cls_id = None

        if "sep_id" in kwargs:
            self.sep_id = kwargs["sep_id"]
        else:
            self.sep_id = None

        self.topo_sort = False
        if "num_labels" in kwargs and kwargs["num_labels"] is not None:
            config.num_labels = kwargs["num_labels"]
            self.num_labels = kwargs["num_labels"]
            self.classifier = RobertaClassificationHead(config)
            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.init_bert_weights)
            # Loss.
            self.topo_loss = nn.CrossEntropyLoss()
            self.topo_sort = True

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None,
                pretraining_objective=None, labels=None):
        if self.topo_sort:
            img_bz, img_len, img_c, img_h, img_w = visual_feats.size()
            visual_feats = visual_feats.reshape(img_bz*img_len, img_c, img_h, img_w)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # If image-only, language shrinked to only cls token.
        if self.multimodal_img_part:
            input_ids = input_ids[:, 0:1]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, 0:1]
            attention_mask = attention_mask[:, 0:1]

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run LXRT backbone
        if pretraining_objective is not None:
            pretraining_objective_labels, (lang_feats, visn_feats) = self.encoder(
                embedding_output,
                extended_attention_mask,
                visn_feats=visual_feats,
                visn_attention_mask=extended_visual_attention_mask,
                input_ids=input_ids,
                pretraining_objective=pretraining_objective)
        else:
            lang_feats, visn_feats = self.encoder(
                embedding_output,
                extended_attention_mask,
                visn_feats=visual_feats,
                visn_attention_mask=extended_visual_attention_mask,
                input_ids=input_ids,
                pretraining_objective=pretraining_objective)

        if self.multimodal_img_part:
            visn_feats = visn_feats.reshape(visn_feats.size(0),
                                            -1, visn_feats.size(2))
            pooled_output = self.pooler(visn_feats)
            if pretraining_objective is not None:
                return pretraining_objective_labels, (visn_feats, None), pooled_output
            return (visn_feats, None), pooled_output

        pooled_output = self.pooler(lang_feats)

        if self.topo_sort:
            topo_logits = self.classifier(pooled_output.unsqueeze(1))
            reshaped_logits = topo_logits.contiguous().view(-1, self.num_labels)

            if labels is not None:
                topo_loss = self.topo_loss(reshaped_logits, labels)
                return topo_loss, reshaped_logits
            else:
                return (reshaped_logits,)

        if pretraining_objective is not None:
            return pretraining_objective_labels, (lang_feats, visn_feats), pooled_output
        return (lang_feats, visn_feats), pooled_output


class LXRTPretraining(BertPreTrainedModel):
    def __init__(self,
                 config,
                 task_mask_lm=True,
                 task_matched=True,
                 task_obj_predict=True,
                 visual_losses='',
                 task_qa=True,
                 num_answers=2,
                 **kwargs):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_answers = num_answers

        # Use of pre-training tasks
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_matched = task_matched
        self.task_qa = task_qa

        # LXRT backbone
        self.bert = LXRTModel(config, **kwargs)

        self.cls_id = kwargs["cls_id"]
        self.sep_id = kwargs["sep_id"]
        self.pad_id = kwargs["pad_id"]

        if "multimodal_img_part" in kwargs:
            self.multimodal_img_part = kwargs["multimodal_img_part"]
        else:
            self.multimodal_img_part = False

        if "multimodal_text_part" in kwargs:
            self.multimodal_text_part = kwargs["multimodal_text_part"]
        else:
            self.multimodal_text_part = False

        if "mlm_ignore_index" in kwargs:
            self.mlm_ignore_index = kwargs["mlm_ignore_index"]
        else:
            self.mlm_ignore_index = -1

        if "multimodal_pretrain_objectives" in kwargs:
            self.multimodal_pretrain = True

            if "image_swapping" in kwargs["multimodal_pretrain_objectives"]:
                self.image_swapping = True
                self.image_swapping_mlp = nn.Linear(config.hidden_size, 2)
            else:
                self.image_swapping = False

            if "patch_based_image_swapping" in kwargs["multimodal_pretrain_objectives"]:
                self.patch_based_image_swapping = True
                self.patch_based_image_swapping_mlp = nn.Linear(config.hidden_size, 2)
            else:
                self.patch_based_image_swapping = False

            if "image_sequence_predictions" in kwargs["multimodal_pretrain_objectives"]:
                self.image_sequencing = True
                self.image_sequencing_mlp = nn.Linear(config.hidden_size, 2)
            else:
                self.image_sequencing = False

            if "patch_based_image_sequence_predictions" in kwargs["multimodal_pretrain_objectives"]:
                self.patch_based_image_sequencing = True
                self.patch_based_image_sequencing_mlp = nn.Linear(config.hidden_size, 2)
            else:
                self.patch_based_image_sequencing = False

            if "time_contrastive" in kwargs["multimodal_pretrain_objectives"]:
                self.time_contrastive = True
                self.time_contrastive_mlp = nn.Linear(config.hidden_size, config.hidden_size)
            else:
                self.time_contrastive = False

            if "whole_image_sequence_swapping" in kwargs["multimodal_pretrain_objectives"]:
                assert not self.multimodal_img_part
                self.whole_image_sequence_swapping = True
                self.whole_image_sequence_swapping_mlp = nn.Linear(config.hidden_size, 2)
            else:
                self.whole_image_sequence_swapping = False

            if "multimodal_swapping" in kwargs["multimodal_pretrain_objectives"]:
                assert not self.multimodal_img_part
                self.multimodal_swapping = True
                kwargs["multimodal_pretrain_objectives"].pop(kwargs[
                    "multimodal_pretrain_objectives"].index(
                        "multimodal_swapping"))
            else:
                self.multimodal_swapping = False

            self.modality_list = ["multimodal", "text_only", "image_only"]
            if ("multimodal_margin_loss" in kwargs["multimodal_pretrain_objectives"]
                or "margin_loss" in kwargs["multimodal_pretrain_objectives"]):
                assert not self.multimodal_img_part
                self.margin_loss = True
                self.margin_loss_mlp = nn.Linear(config.hidden_size, 1)
                if "multimodal_margin_loss" in kwargs["multimodal_pretrain_objectives"]:
                    assert not self.multimodal_img_part and not self.multimodal_text_part
            else:
                self.margin_loss = False

            if "patch_based_mrm_classification" in kwargs["multimodal_pretrain_objectives"]:
                self.patch_based_mrm_classification = True
                self.pb_mrm_cls_mask_num = 5
                # self.patch_based_mrm_classification_head = BertLMPredictionHead(
                #     config, torch.nn.Embedding(self.pb_mrm_cls_mask_num
                #     * VISUAL_CONFIG.max_subsample_image_length,
                #     config.hidden_size).weight)
                new_config = copy.deepcopy(config)
                new_config.hidden_size = config.hidden_size * 2
                self.patch_based_mrm_classification_head = BertLMPredictionHead(
                    new_config, torch.nn.Embedding(1, new_config.hidden_size).weight)
            else:
                self.patch_based_mrm_classification = False

            self.multimodal_pretrain_objectives = kwargs["multimodal_pretrain_objectives"]

        else:
            self.multimodal_pretrain = False

        # Pre-training heads
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        if self.task_obj_predict and not self.multimodal_text_part:
            self.obj_predict_head = BertVisualObjHead(config, visual_losses)
        if self.task_qa and not self.multimodal_text_part:
            self.answer_head = BertVisualAnswerHead(config, self.num_answers)

        # Weight initialization
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):

        if type(input_ids) == dict:
            batch = input_ids
            input_ids = batch["input_ids"] if "input_ids" in batch else None
            token_type_ids = batch["token_type_ids"] if "token_type_ids" in batch else None
            attention_mask = batch["attention_mask"] if "attention_mask" in batch else None
            masked_lm_labels = batch["masked_lm_labels"] if "masked_lm_labels" in batch else None
            visual_feats = batch["images"] if "images" in batch else None
            # obj_labels = batch["image_target"] if "image_target" in batch else None
            # TODO: add the rest?

        # If image-only, language shrinked to only cls token.
        if self.multimodal_img_part:
            input_ids = input_ids[:, 0:1]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, 0:1]
            attention_mask = attention_mask[:, 0:1]
            masked_lm_labels = None

        pretraining_objective = None

        # Some functions used for pretraining.
        def pad_to_length_with_val(length, val, tensor):
            new_tensor = torch.ones(length).type_as(tensor) * val
            new_tensor[:tensor.size(0)] = tensor
            return new_tensor

        def multimodal_subsampling(lang_feats, visual_feats, indices, pad_length):
            if not self.multimodal_text_part:
                new_visual_feats = visual_feats[indices]
            else:
                new_visual_feats = None

            if not self.multimodal_img_part:
                assert type(lang_feats) == tuple
                assert len(lang_feats) == 4
                input_ids, token_type_ids, attention_mask, masked_lm_labels = lang_feats

                cls_pos = (input_ids==self.cls_id).nonzero().reshape(-1)
                indices_lang = []
                for k in range(len(indices)):
                    idx_curr = indices[k]
                    start_idx = cls_pos[idx_curr]
                    if idx_curr == VISUAL_CONFIG.max_story_length - 1:
                        end_idx = start_idx + input_ids.size(0) // VISUAL_CONFIG.max_story_length
                    else:
                        end_idx = cls_pos[idx_curr+1]
                    indices_lang += list(range(start_idx, end_idx))
                new_input_ids = input_ids[indices_lang]
                new_input_ids = pad_to_length_with_val(
                    pad_length, 0, new_input_ids)
                if token_type_ids is not None:
                    new_token_type_ids = token_type_ids[indices_lang]
                    new_token_type_ids = pad_to_length_with_val(
                        pad_length, 0, new_token_type_ids)
                else:
                    new_token_type_ids = None
                new_attention_mask = attention_mask[indices_lang]
                new_attention_mask = pad_to_length_with_val(
                    pad_length, 0, new_attention_mask)
                new_masked_lm_labels = masked_lm_labels[indices_lang]
                new_masked_lm_labels = pad_to_length_with_val(
                    pad_length, self.mlm_ignore_index, new_masked_lm_labels)
                # print(new_input_ids.size(), new_input_ids)
                # print(new_token_type_ids)
                # print(new_attention_mask.size(), new_attention_mask)
                # print(new_masked_lm_labels.size(), new_masked_lm_labels)
                # raise
                new_lang_feats = (new_input_ids, new_token_type_ids,
                                  new_attention_mask, new_masked_lm_labels)
            else:
                new_lang_feats = None

            return new_lang_feats, new_visual_feats

        # Sequentiality pretraining preparations.
        if self.multimodal_pretrain and len(self.multimodal_pretrain_objectives) > 0:
            assert visual_feats.ndim >= 5
            bz = visual_feats.size(0)
            img_len = visual_feats.size(1)

            obj_this_batch = np.random.choice(self.multimodal_pretrain_objectives,
                                              1, replace=False)[0]
            # print(obj_this_batch)

            assert VISUAL_CONFIG.max_subsample_image_length == VISUAL_CONFIG.max_subsample_text_length

            if self.margin_loss and "margin_loss" in obj_this_batch:
                margin_loss_sample_length = 2

                if not self.multimodal_img_part:
                    pad_length = input_ids.size(1)//VISUAL_CONFIG.max_story_length*margin_loss_sample_length
                    new_input_ids_1 = []
                    new_input_ids_2 = []
                    if token_type_ids is not None:
                        new_token_type_ids_1 = []
                        new_token_type_ids_2 = []
                    else:
                        new_token_type_ids = None
                    new_attention_mask_1 = []
                    new_attention_mask_2 = []
                    new_masked_lm_labels_1 = []
                    new_masked_lm_labels_2 = []
                else:
                    pad_length = None

                if not self.multimodal_text_part:
                    new_visual_feats_1 = []
                    new_visual_feats_2 = []

                margin_target = []
                for i in range(bz):
                    if not self.multimodal_img_part:
                        input_ids_i = input_ids[i]
                        if token_type_ids is not None:
                            token_type_ids_i = token_type_ids[i]
                        else:
                            token_type_ids_i = None
                        attention_mask_i = attention_mask[i]
                        masked_lm_labels_i = masked_lm_labels[i]
                        lang_feats_i = (input_ids_i, token_type_ids_i,
                                        attention_mask_i, masked_lm_labels_i)
                    else:
                        lang_feats_i = (None, None, None, None)
                    
                    if not self.multimodal_text_part:
                        visual_feats_i = visual_feats[i]
                    else:
                        visual_feats_i = None

                    ub = VISUAL_CONFIG.max_story_length
                    ub_i = ub - margin_loss_sample_length
                    ub_j = ub - 1
                    ub_k = ub
                    idx_i = np.random.randint(0, ub_i)
                    idx_j = np.random.randint(idx_i+1, ub_j)
                    idx_k = np.random.randint(idx_j+1, ub_k)
                    assert idx_i < idx_j and idx_j < idx_k
                    
                    indices_1 = [idx_i, idx_j]
                    indices_2 = [idx_i, idx_k]

                    use_reverse_prob = 0.7
                    use_mix_prob = 0.5
                    jk_prob = 0.5
                    if np.random.rand() > use_reverse_prob:
                        if np.random.rand() > use_mix_prob:
                            if np.random.rand() > jk_prob:
                                indices_1 = [idx_i, idx_k]
                                indices_2 = [idx_k, idx_i]
                            else:
                                indices_1 = [idx_i, idx_j]
                                indices_2 = [idx_j, idx_i]
                        else:
                            indices_1 = [idx_j, idx_i]
                            indices_2 = [idx_k, idx_i]

                    # print()
                    # print(indices_1, indices_2);raise
                    # TODO: shuffle this?
                    margin_target.append(1)

                    ret_1 = multimodal_subsampling(lang_feats_i, visual_feats_i,
                                                   indices_1, pad_length)
                    new_lang_feats_i1, new_visual_feats_i1 = ret_1
                    ret_2 = multimodal_subsampling(lang_feats_i, visual_feats_i,
                                                   indices_2, pad_length)
                    new_lang_feats_i2, new_visual_feats_i2 = ret_2

                    if not self.multimodal_text_part:
                        new_visual_feats_1.append(new_visual_feats_i1)
                        new_visual_feats_2.append(new_visual_feats_i2)
                    if not self.multimodal_img_part:
                        (new_input_ids_i1, new_token_type_ids_i1,
                         new_attention_mask_i1, new_masked_lm_labels_i1) \
                         = new_lang_feats_i1
                        new_input_ids_1.append(new_input_ids_i1)
                        if new_token_type_ids_i1 is not None:
                            new_token_type_ids_1.append(new_token_type_ids_i1)
                        new_attention_mask_1.append(new_attention_mask_i1)
                        new_masked_lm_labels_1.append(new_masked_lm_labels_i1)
                        (new_input_ids_i2, new_token_type_ids_i2,
                         new_attention_mask_i2, new_masked_lm_labels_i2) \
                         = new_lang_feats_i2
                        new_input_ids_2.append(new_input_ids_i2)
                        if new_token_type_ids_i2 is not None:
                            new_token_type_ids_2.append(new_token_type_ids_i2)
                        new_attention_mask_2.append(new_attention_mask_i2)
                        new_masked_lm_labels_2.append(new_masked_lm_labels_i2)
                        
                if not self.multimodal_text_part:
                    new_visual_feats_1 = torch.stack(new_visual_feats_1)
                    new_visual_feats_2 = torch.stack(new_visual_feats_2)
                    visual_feats = torch.cat([new_visual_feats_1, new_visual_feats_2], dim=0)

                if not self.multimodal_img_part:
                    new_input_ids_1 = torch.stack(new_input_ids_1)
                    new_input_ids_2 = torch.stack(new_input_ids_2)
                    if token_type_ids is not None:
                        new_token_type_ids_1 = torch.stack(new_token_type_ids_1)
                        new_token_type_ids_2 = torch.stack(new_token_type_ids_2)
                    new_attention_mask_1 = torch.stack(new_attention_mask_1)
                    new_attention_mask_2 = torch.stack(new_attention_mask_2)
                    new_masked_lm_labels_1 = torch.stack(new_masked_lm_labels_1)
                    new_masked_lm_labels_2 = torch.stack(new_masked_lm_labels_2)
                    input_ids = torch.cat([new_input_ids_1, new_input_ids_2], dim=0)
                    if token_type_ids is not None:
                        token_type_ids = torch.cat([new_token_type_ids_1, new_token_type_ids_2], dim=0)
                    attention_mask = torch.cat([new_attention_mask_1, new_attention_mask_2], dim=0)
                    masked_lm_labels = torch.cat([new_masked_lm_labels_1, new_masked_lm_labels_2], dim=0)

                margin_target = torch.Tensor(margin_target).type_as(input_ids)

                # Multimodality
                if "multimodal_margin_loss" == obj_this_batch:
                    curr_modality = np.random.choice(self.modality_list, 1, replace=False)[0]
                    if curr_modality == "image_only":
                        input_ids = input_ids[:, 0:1]
                        if token_type_ids is not None:
                            token_type_ids = token_type_ids[:, 0:1]
                        attention_mask = attention_mask[:, 0:1]
                        masked_lm_labels = masked_lm_labels[:, 0:1]
                    elif curr_modality == "text_only":
                        visual_feats = None
                    else:
                        pass  # Multimodal.

            # Sub-sampling.
            if ((self.image_swapping and obj_this_batch == "image_swapping") or
                (self.image_sequencing and obj_this_batch == "image_sequence_predictions") or
                (self.patch_based_image_swapping and obj_this_batch == "patch_based_image_swapping") or 
                (self.patch_based_mrm_classification and obj_this_batch == "patch_based_mrm_classification") or
                (self.patch_based_image_sequencing and obj_this_batch == "patch_based_image_sequence_predictions")):
                if (VISUAL_CONFIG.max_subsample_image_length == VISUAL_CONFIG.max_subsample_text_length
                    and not self.multimodal_img_part):
                    new_input_ids = []
                    if token_type_ids is not None:
                        new_token_type_ids = []
                    else:
                        new_token_type_ids = None
                    new_attention_mask = []
                    new_masked_lm_labels = []

                new_visual_feats = []
                for i in range(bz):
                    sub_idx = np.random.choice(img_len,
                        VISUAL_CONFIG.max_subsample_image_length, replace=False)
                    sub_idx = sorted(sub_idx)
                    visual_feats_ = visual_feats[i, sub_idx]
                    new_visual_feats.append(visual_feats_)

                    pad_length = input_ids.size(1)//VISUAL_CONFIG.max_story_length*VISUAL_CONFIG.max_subsample_text_length
                    if (VISUAL_CONFIG.max_subsample_image_length == VISUAL_CONFIG.max_subsample_text_length
                        and not self.multimodal_img_part):
                        cls_pos = (input_ids[i]==self.cls_id).nonzero().reshape(-1)
                        sub_idx_lang = []
                        for k in range(len(sub_idx)):
                            sub_idx_curr = sub_idx[k]
                            start_idx = cls_pos[sub_idx_curr]
                            if sub_idx_curr == img_len - 1:
                                end_idx = start_idx + input_ids.size(1) // VISUAL_CONFIG.max_story_length
                            else:
                                end_idx = cls_pos[sub_idx_curr+1]
                            sub_idx_lang += list(range(start_idx, end_idx))
                        input_ids_ = input_ids[i, sub_idx_lang]
                        input_ids_ = pad_to_length_with_val(
                            pad_length, self.pad_id, input_ids_)
                        new_input_ids.append(input_ids_)
                        if token_type_ids is not None:
                            token_type_ids_ = token_type_ids[i, sub_idx_lang]
                            token_type_ids_ = pad_to_length_with_val(
                                pad_length, 0, token_type_ids_)
                            new_token_type_ids.append(token_type_ids_)
                        attention_mask_ = attention_mask[i, sub_idx_lang]
                        attention_mask_ = pad_to_length_with_val(
                            pad_length, 0, attention_mask_)
                        new_attention_mask.append(attention_mask_)
                        masked_lm_labels_ = masked_lm_labels[i, sub_idx_lang]
                        masked_lm_labels_ = pad_to_length_with_val(
                            pad_length, self.mlm_ignore_index, masked_lm_labels_)
                        new_masked_lm_labels.append(masked_lm_labels_)

                new_visual_feats = torch.stack(new_visual_feats)
                visual_feats = new_visual_feats
                img_len = visual_feats.size(1)

                if (VISUAL_CONFIG.max_subsample_image_length == VISUAL_CONFIG.max_subsample_text_length
                    and not self.multimodal_img_part):
                    new_input_ids = torch.stack(new_input_ids)
                    if token_type_ids is not None:
                        new_token_type_ids = torch.stack(new_token_type_ids)
                    new_attention_mask = torch.stack(new_attention_mask)
                    new_masked_lm_labels = torch.stack(new_masked_lm_labels)
                    input_ids = new_input_ids
                    token_type_ids = new_token_type_ids
                    attention_mask = new_attention_mask
                    masked_lm_labels = new_masked_lm_labels

            if self.image_swapping and obj_this_batch == "image_swapping":
                
                # print(torch.sum(visual_feats, (2,3,4)))

                image_swapping_labels = []
                assert img_len >= 2
                for i in range(bz):
                    swap_prob = np.random.rand()
                    if swap_prob > 0.5:
                        swap_steps = np.random.choice(img_len, 2, replace=False)
                        swap_steps = sorted(swap_steps)
                        swap_x, swap_y = swap_steps[0], swap_steps[1]
                        visual_feats_y = visual_feats[i][swap_y]
                        visual_feats[i][swap_y] = visual_feats[i][swap_x]
                        visual_feats[i][swap_x] = visual_feats_y
                        image_swapping_labels.append(0)
                    else:
                        image_swapping_labels.append(1)
                image_swapping_labels = torch.Tensor(image_swapping_labels)
                image_swapping_labels = image_swapping_labels.type_as(input_ids)
                # print(torch.sum(visual_feats, (2,3,4)))

            if self.image_sequencing and obj_this_batch == "image_sequence_predictions":
                assert visual_feats.ndim >= 5
                assert bz > 1

                visual_feats_clone = visual_feats.detach().clone()
                image_sequencing_labels = []
                for i in range(bz):
                    swap_prob = np.random.rand()
                    if swap_prob > 0.5:
                        choices = list(range(bz))
                        choices.pop(i)
                        swap_batch = np.random.choice(choices,
                            1, replace=False)[0]
                        swap_seq_id = np.random.choice(list(range(img_len)),
                            1, replace=False)[0]
                        curr_seq_id = np.random.choice(list(range(img_len)),
                            1, replace=False)[0]
                        visual_feats[i][curr_seq_id] = visual_feats_clone[
                            swap_batch][swap_seq_id]
                        image_sequencing_labels.append(0)
                    else:
                        image_sequencing_labels.append(1)
                image_sequencing_labels = torch.Tensor(image_sequencing_labels)
                image_sequencing_labels = image_sequencing_labels.type_as(input_ids)

            if self.whole_image_sequence_swapping and obj_this_batch == "whole_image_sequence_swapping":
                assert visual_feats.ndim >= 5
                assert bz > 1

                visual_feats_clone = visual_feats.detach().clone()
                whole_image_sequence_swapping_labels = []
                # print(torch.sum(visual_feats, (2,3,4)))
                for i in range(bz):
                    swap_prob = np.random.rand()
                    if swap_prob > 0.5:
                        choices = list(range(bz))
                        choices.pop(i)
                        swap_batch = np.random.choice(choices,
                            1, replace=False)[0]
                        visual_feats[i] = visual_feats_clone[swap_batch]
                        whole_image_sequence_swapping_labels.append(0)
                    else:
                        whole_image_sequence_swapping_labels.append(1)
                # print(whole_image_sequence_swapping_labels)
                # print(torch.sum(visual_feats, (2,3,4)));raise
                whole_image_sequence_swapping_labels = torch.Tensor(whole_image_sequence_swapping_labels)
                whole_image_sequence_swapping_labels = whole_image_sequence_swapping_labels.type_as(input_ids)

            if ((self.patch_based_image_swapping
                 or self.patch_based_image_sequencing
                 or self.patch_based_mrm_classification)
                and "patch_based" in obj_this_batch):
                pretraining_objective = obj_this_batch
                if "mrm_classification" in obj_this_batch:
                    assert not self.multimodal_text_part
                    pretraining_objective = "{}_{}".format(
                        obj_this_batch, self.pb_mrm_cls_mask_num)
                pass

            if ("image_swapping" in obj_this_batch and self.multimodal_swapping
                and not self.multimodal_img_part and VISUAL_CONFIG.max_subsample_image_length == VISUAL_CONFIG.max_subsample_text_length):
                lang_swapped_labels = []
                new_input_ids = []
                if token_type_ids is not None:
                    new_token_type_ids = []
                else:
                    new_token_type_ids = None
                new_attention_mask = []
                new_masked_lm_labels = []
                pad_length = input_ids.size(1)

                self.lang_swap_prob = 0.75

                for i in range(bz):
                    if_lang_swap = np.random.rand()
                    if if_lang_swap > self.lang_swap_prob:

                        first_pad_pos = int((input_ids[i]!=self.pad_id).nonzero().reshape(-1)[-1].cpu().numpy())

                        input_ids_ = input_ids[i][:first_pad_pos+1]
                        if token_type_ids is not None:
                            token_type_ids_ = token_type_ids[:first_pad_pos+1]
                        attention_mask_ = attention_mask[i][:first_pad_pos+1]
                        masked_lm_labels_ = masked_lm_labels[i][:first_pad_pos+1]

                        cls_pos = (input_ids_==self.cls_id).nonzero().reshape(-1)
                        cls_pos = cls_pos.detach().cpu().tolist()
                        len_cls = len(cls_pos)
                        idx_to_swap = sorted(np.random.choice(len_cls, 2, replace=False))
                        cls_pos.append(first_pad_pos+1)

                        curr_indices = list(range(VISUAL_CONFIG.max_subsample_text_length))
                        curr_indices_idx_i_tmp = curr_indices[idx_to_swap[0]]
                        curr_indices[idx_to_swap[0]] = curr_indices[idx_to_swap[1]]
                        curr_indices[idx_to_swap[1]] = curr_indices_idx_i_tmp

                        start_end_idx = []
                        for j in curr_indices:
                            start_end_idx.append((cls_pos[j], cls_pos[j+1]))
                        # print(curr_indices)
                        # print(start_end_idx)

                        input_ids_curr_ = []
                        if token_type_ids is not None:
                            token_type_curr_ = []
                        attention_mask_curr_ = []
                        masked_lm_labels_curr_ = []

                        for start_end in start_end_idx:
                            start, end = start_end
                            input_ids_curr_.append(input_ids_[start:end])
                            if token_type_ids is not None:
                                token_type_ids_curr_.append(token_type_ids_[start:end])
                            attention_mask_curr_.append(attention_mask_[start:end])
                            masked_lm_labels_curr_.append(masked_lm_labels_[start:end])
                        input_ids_curr_ = torch.cat(input_ids_curr_, dim=-1)
                        if token_type_ids is not None:
                            token_type_ids_curr_ = torch.cat(token_type_ids_curr_, dim=-1)
                        attention_mask_curr_ = torch.cat(attention_mask_curr_, dim=-1)
                        masked_lm_labels_curr_ = torch.cat(masked_lm_labels_curr_, dim=-1)
                        # print(input_ids[i])
                        # print(input_ids_curr_);raise

                        input_ids_ = pad_to_length_with_val(
                            pad_length, self.pad_id, input_ids_curr_)
                        if token_type_ids is not None:
                            token_type_ids_ = pad_to_length_with_val(
                                pad_length, 0, token_type_ids_curr_)
                        attention_mask_ = pad_to_length_with_val(
                            pad_length, 0, attention_mask_curr_)
                        masked_lm_labels_ = pad_to_length_with_val(
                            pad_length, self.mlm_ignore_index, masked_lm_labels_curr_)

                        lang_swapped_labels.append(0)
                    else:
                        input_ids_ = input_ids[i]
                        if token_type_ids is not None:
                            token_type_ids_ = token_type_ids[i]
                        attention_mask_ = attention_mask[i]
                        masked_lm_labels_ = masked_lm_labels[i]

                        lang_swapped_labels.append(1)

                    new_input_ids.append(input_ids_)
                    if token_type_ids is not None:
                        new_token_type_ids.append(token_type_ids_)
                    new_attention_mask.append(attention_mask_)
                    new_masked_lm_labels.append(masked_lm_labels_)

                new_input_ids = torch.stack(new_input_ids)
                if token_type_ids is not None:
                    new_token_type_ids = torch.stack(new_token_type_ids)
                new_attention_mask = torch.stack(new_attention_mask)
                new_masked_lm_labels = torch.stack(new_masked_lm_labels)
                input_ids = new_input_ids
                token_type_ids = new_token_type_ids
                attention_mask = new_attention_mask
                masked_lm_labels = new_masked_lm_labels
                lang_swapped_labels = torch.Tensor(lang_swapped_labels).type_as(
                    masked_lm_labels).long()
            ####
        ####

        if visual_feats is not None and visual_feats.ndim >= 5:
            cc, hh, ww = visual_feats.size(2), visual_feats.size(3), visual_feats.size(4)
            visual_feats = visual_feats.reshape(-1, cc, hh, ww)

        # print(input_ids.size(), token_type_ids, attention_mask.size())
        # print(visual_feats.size())
        if pretraining_objective is None:
            (lang_output, visn_output), pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask,
                visual_feats=(visual_feats, pos),
                pretraining_objective=pretraining_objective,
            )
        else:
            pretraining_objective_labels, (lang_output, visn_output), pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask,
                visual_feats=(visual_feats, pos),
                pretraining_objective=pretraining_objective,
            )
        # print(lang_output.size());raise
        # print(lang_output.size(), visn_output.size(), pooled_output.size());raise
        # print(lang_output.size(), visn_output, pooled_output.size());raise

        # XXX: If multimodal_img_part, than the lang_output become the actual
        # visn_output, and then visn_output is None.
        if self.multimodal_img_part:
            # print(lang_output.size(), visn_output, pooled_output.size());raise
            pass

        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa and not self.multimodal_text_part:
            answer_score = self.answer_head(pooled_output)
        else:
            # This answer_score would not be used anywhere,
            # just to keep a constant return function signature.
            answer_score = pooled_output[0][0]

        total_loss = 0.
        loss_fct = CrossEntropyLoss(ignore_index=self.mlm_ignore_index)
        losses = ()

        # Sequentiality pretraining losses.
        # print(image_swapping_labels)
        # print(lang_output.size(), visn_output.size(), pooled_output.size());raise
        if self.multimodal_pretrain and len(self.multimodal_pretrain_objectives) > 0:

            if ("image_swapping" in obj_this_batch and self.multimodal_swapping
                and not self.multimodal_img_part and VISUAL_CONFIG.max_subsample_image_length == VISUAL_CONFIG.max_subsample_text_length):
                if "image_swapping" == obj_this_batch:
                    image_swapping_labels *= lang_swapped_labels
                elif "patch_based_image_swapping" == obj_this_batch:
                    pretraining_objective_labels *= lang_swapped_labels
                pass  ####

            if self.image_swapping and obj_this_batch == "image_swapping":
                image_swapping_logits = self.image_swapping_mlp(pooled_output)
                image_swapping_loss_fct = CrossEntropyLoss()
                image_swapping_loss = image_swapping_loss_fct(
                    image_swapping_logits, image_swapping_labels)
                total_loss += image_swapping_loss
                losses += (image_swapping_loss.detach(),)

            if self.image_sequencing and obj_this_batch == "image_sequence_predictions":
                image_sequencing_logits = self.image_sequencing_mlp(pooled_output)
                image_sequencing_loss_fct = CrossEntropyLoss()
                image_sequencing_loss = image_sequencing_loss_fct(
                    image_sequencing_logits, image_sequencing_labels)
                total_loss += image_sequencing_loss
                losses += (image_sequencing_loss.detach(),)

            if self.whole_image_sequence_swapping and obj_this_batch == "whole_image_sequence_swapping":
                whole_image_sequence_swapping_logits = self.whole_image_sequence_swapping_mlp(pooled_output)
                whole_image_sequence_swapping_loss_fct = CrossEntropyLoss()
                whole_image_sequence_swapping_loss = whole_image_sequence_swapping_loss_fct(
                    whole_image_sequence_swapping_logits, whole_image_sequence_swapping_labels)
                total_loss += whole_image_sequence_swapping_loss
                losses += (whole_image_sequence_swapping_loss.detach(),)

            if ((self.patch_based_image_swapping or self.patch_based_image_sequencing)
                and "patch_based_image" in obj_this_batch):
                if self.patch_based_image_swapping:
                    patch_based_objective_logits = self.patch_based_image_swapping_mlp(pooled_output)
                elif self.patch_based_image_sequencing:
                    patch_based_objective_logits = self.patch_based_image_sequence_predictions(pooled_output)
                patch_based_objective_loss_fct = CrossEntropyLoss()
                patch_based_objective_labels = pretraining_objective_labels
                patch_based_objective_loss = patch_based_objective_loss_fct(
                    patch_based_objective_logits, patch_based_objective_labels)
                total_loss += patch_based_objective_loss
                losses += (patch_based_objective_loss.detach(),)

            if "patch_based_mrm_classification" in obj_this_batch:
                if self.multimodal_img_part:
                    visn_output = lang_output
                mask_patch_gt, mask_patch_mask = pretraining_objective_labels
                # print(mask_patch_gt.size(), mask_patch_mask.size())
                # visn_output_mrm = visn_output.unsqueeze(1)
                shuffle_labels = []
                shuffled_mask_patch_gt = []
                visn_output_concat = []
                bz = visn_output.size(0)
                mrm_loss = 0
                for i in range(bz):
                    indices = np.arange(self.pb_mrm_cls_mask_num *
                        VISUAL_CONFIG.max_subsample_image_length)
                    np.random.shuffle(indices)
                    mask_patch_gt_curr = mask_patch_gt[i][indices]
                    shuffled_mask_patch_gt.append(mask_patch_gt_curr)
                    indices_argsort = np.argsort(indices)
                    shuffle_labels.append(list(indices_argsort))
                    visn_output_curr = visn_output[i]
                    mask_patch_mask_curr = mask_patch_mask[i]
                    mask_patch_indices = (mask_patch_mask_curr==1).nonzero().reshape(-1).long()
                    visn_output_masked = visn_output_curr[mask_patch_indices]

                    visn_output_qa = []
                    for j in range(visn_output_masked.size(0)):
                        visn_output_masked_j = visn_output_masked[j].unsqueeze(0)
                        visn_output_masked_j = visn_output_masked_j.repeat(
                            visn_output_masked.size(0), 1)
                        visn_output_masked_j = torch.cat([visn_output_masked_j, mask_patch_gt_curr], dim=-1)
                        visn_output_qa.append(visn_output_masked_j)

                    visn_output_qa = torch.stack(visn_output_qa)
                    visn_output_qa_scores = self.patch_based_mrm_classification_head(visn_output_qa)
                    visn_output_qa_scores = visn_output_qa_scores.squeeze()
                    visn_output_qa_labels = torch.Tensor(indices_argsort).type_as(visn_output_qa_scores).long()
                    visn_output_qa_loss = loss_fct(visn_output_qa_scores, visn_output_qa_labels)
                    mrm_loss += visn_output_qa_loss

                mrm_scale = 0.2
                total_loss += mrm_scale * mrm_loss
                # print(total_loss, mrm_loss);raise
                losses += (mrm_scale * mrm_loss.detach(),)

            if self.margin_loss and "margin_loss" in obj_this_batch:
                margin_logits = self.margin_loss_mlp(pooled_output)
                margin_bz = margin_logits.size(0)
                margin_length = margin_bz // 2
                margin_logits_1 = margin_logits[:margin_length]
                margin_logits_2 = margin_logits[margin_length:]
                margin_loss_fct = torch.nn.MarginRankingLoss(margin=1.0)
                margin_loss_val = margin_loss_fct(margin_logits_1,
                    margin_logits_2, margin_target)
                total_loss += margin_loss_val
                # print(margin_loss_val);raise
                losses += (margin_loss_val.detach(),)
                ####
                lang_prediction_scores = lang_prediction_scores[:margin_length]
                masked_lm_labels = masked_lm_labels[:margin_length]

            if self.time_contrastive and obj_this_batch == "time_contrastive":
                seq_len = lang_output.size(1)
                curr_len = VISUAL_CONFIG.max_subsample_image_length
                curr_len = VISUAL_CONFIG.max_story_length
                patch_len = seq_len // curr_len

                all_cls_pos = list(range(1, seq_len, patch_len))
                all_cls_pos.pop(0)
                all_cls_pos.insert(0, 0)

                anchors = []
                positives = []
                negatives = []

                bz = lang_output.size(0)
                for i in range(bz):
                    choice_idx = list(range(curr_len))
                    anchor_idx = np.random.choice(choice_idx, 1, replace=False)[0]
                    pos_indices = []
                    if anchor_idx - 1 >= 0:
                        pos_indices.append(anchor_idx - 1)
                    if anchor_idx + 1 < curr_len:
                        pos_indices.append(anchor_idx + 1)
                    positive_idx = np.random.choice(pos_indices, 1, replace=False)[0]
                    neg_indices = []
                    for j in range(curr_len):
                        if abs(j - anchor_idx) >= 2:
                            neg_indices.append(j)
                    negative_idx = np.random.choice(neg_indices, 1, replace=False)[0]
                    if not self.multimodal_img_part:
                        all_cls_pos = (input_ids[i]==self.cls_id).nonzero().reshape(-1)
                    anchor_idx = all_cls_pos[anchor_idx]
                    positive_idx = all_cls_pos[positive_idx]
                    negative_idx = all_cls_pos[negative_idx]
                    # print(all_cls_pos)
                    # print(anchor_idx, pos_indices, positive_idx, neg_indices, negative_idx);raise
                    anchors.append(lang_output[i][anchor_idx])
                    positives.append(lang_output[i][positive_idx])
                    negatives.append(lang_output[i][negative_idx])

                anchors = torch.stack(anchors)
                positives = torch.stack(positives)
                negatives = torch.stack(negatives)

                triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
                time_contrastive_loss = triplet_loss(anchors, positives, negatives)
                total_loss += time_contrastive_loss
                # print(total_loss);raise
                losses += (time_contrastive_loss.detach(),)
                
            ####
            ####
        ####
        # masked_lm_labels = None
        if masked_lm_labels is not None and self.task_mask_lm:
            # print(input_ids[0])
            # print(masked_lm_labels[0])
            # raise
            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
            losses += (masked_lm_loss.detach(),)

        if matched_label is not None and self.task_matched:
            raise NotImplementedError("Do not use this!")
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            losses += (matched_loss.detach(),)

        if obj_labels is not None and self.task_obj_predict:
            raise NotImplementedError("Do not use this!")
            loss_fcts = {
                'l2': SmoothL1Loss(reduction='none'),
                'ce': CrossEntropyLoss(ignore_index=-1, reduction='none')
            }
            total_visn_loss = 0.
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in VISUAL_CONFIG.visual_losses:
                label, mask_conf = obj_labels[key]
                output_dim, loss_fct_name, label_shape, weight = VISUAL_CONFIG.visual_loss_config[key]
                visn_loss_fct = loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(
                    visn_prediction_scores.view(-1, output_dim),
                    label.view(*label_shape),
                )
                if visn_loss.dim() > 1:     # Regression Losses
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
                losses += (visn_loss.detach(),)
            total_loss += total_visn_loss

        if ans is not None and self.task_qa:
            raise NotImplementedError("Do not use this!")
            answer_loss = loss_fct(
                answer_score.view(-1, self.num_answers),
                ans.view(-1)
            )  
            # Since this Github version pre-trains with QA loss from the beginning,
            # I exclude "*2" here to match the effect of QA losses.
            # Previous: (loss *0) for 6 epochs, (loss *2) for 6 epochs.   (Used 10 instead of 6 in EMNLP paper)
            # Now     : (loss *1) for 12 epochs
            #
            # * 2       # Multiply by 2 because > half of the data will not have label
            total_loss += answer_loss
            losses += (answer_loss.detach(),)

        # print(obj_this_batch, total_loss, total_loss-masked_lm_loss)

        return total_loss, torch.stack(losses).unsqueeze(0), answer_score.detach()


class LXRTFeatureExtraction(BertPreTrainedModel):
    """
    BERT model for classification.
    """
    def __init__(self, config, mode='lxr', **kwargs):
        """

        :param config:
        :param mode:  Number of visual layers
        """
        super().__init__(config)
        self.bert = LXRTModel(config)
        self.mode = mode
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None,
                visual_attention_mask=None):
        feat_seq, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                            visual_feats=visual_feats,
                                            visual_attention_mask=visual_attention_mask)
        if 'x' == self.mode:
            return pooled_output
        elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
            return feat_seq, pooled_output
        elif 'l' in self.mode or 'r' in self.mode:
            return feat_seq

