# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/uclanlp/visualbert
# which was cleaned up and adapted for MMF.

import os
import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from mmf.modules.hf_layers import BertEncoderJit, BertLayerJit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.torchscript import getattr_torchscriptable
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertConfig,
    BertForPreTraining,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)
from transformers.modeling_roberta import RobertaClassificationHead
from trainers.input_utils import get_detailed_input_feats

from models.heatmap_module import HeatMapOutput
from models.pointer_module import PointerOutput


class VisualBERTBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=512,
        embedding_strategy="plain",
        bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.embedding_strategy = embedding_strategy
        config.bypass_transformer = bypass_transformer
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.embeddings = BertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoderJit(config)
        # self.pooler = BertPooler(config)
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = BertLayerJit(config)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # Python builtin next is currently not supported in Torchscript
        if not torch.jit.is_scripting():
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
        )

        if (
            self.bypass_transformer
            and visual_embeddings is not None
            and input_ids is not None
            and hasattr(self, "additional_layer")
        ):
            assert (
                not self.output_hidden_states
            )  # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[
                :, :, :text_length, :text_length
            ]

            encoded_layers = self.encoder(
                text_embedding_output, text_extended_attention_mask
            )
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(
                new_input, extended_attention_mask
            )
            # pooled_output = self.pooler(final_sequence_output[0])
            pooled_output = final_sequence_output[0]
            return final_sequence_output[0], pooled_output, []

        elif (
            visual_embeddings is not None
            and input_ids is None
        ):

            img_length = visual_embeddings.size(1)
            img_embedding_output = embedding_output[:, :img_length, :]

            img_extended_attention_mask = extended_attention_mask[
                :, :, :img_length, :img_length
            ]

            encoded_layers = self.encoder(
                img_embedding_output, img_extended_attention_mask
            )
            sequence_output = encoded_layers[0]
            # pooled_output = self.pooler(sequence_output)
            pooled_output = sequence_output
            attn_data_list: List[Tensor] = []

            if not torch.jit.is_scripting():
                if self.output_attentions:
                    attn_data_list = encoded_layers[1:]
            else:
                assert (
                    not self.output_attentions
                ), "output_attentions not supported in script mode"

            return sequence_output, pooled_output, attn_data_list

        else:
            encoded_layers = self.encoder(embedding_output, extended_attention_mask)
            sequence_output = encoded_layers[0]
            # pooled_output = self.pooler(sequence_output)
            pooled_output = sequence_output
            attn_data_list: List[Tensor] = []

            if not torch.jit.is_scripting():
                if self.output_attentions:
                    attn_data_list = encoded_layers[1:]
            else:
                assert (
                    not self.output_attentions
                ), "output_attentions not supported in script mode"

            return sequence_output, pooled_output, attn_data_list


class VisualBERTForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        # self.bert_config = BertConfig.from_dict(
        #     OmegaConf.to_container(self.config, resolve=True)
        # )
        self.bert_config = config
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    "pretrained_models/bert", "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.vocab_size = self.bert.config.vocab_size

        # TODO: Once omegaconf fixes int keys issue, bring this back
        # See https://github.com/omry/omegaconf/issues/149
        # with omegaconf.open_dict(self.config):
        #     # Add bert config such as hidden_state to our main config
        #     self.config.update(self.bert.config.to_dict())
        if self.bert_model_name is None:
            bert_masked_lm = BertForPreTraining(self.bert.config)
        else:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                self.config.bert_model_name,
                config=self.bert.config,
                cache_dir=os.path.join(
                    "pretrained_models/bert", "distributed_{}".format(-1)
                ),
            )
        self.cls = deepcopy(bert_masked_lm.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.init_weights()

        # Parameters for objectives.
        self.itm_ot_lambda = 0.1
        self.swapping_based_nsp_prob = 0.5

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them
            instead.
        """
        self.bert._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        downstream_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        if "itm" in self.config.multimodal_pretrain_objectives:
            itm_outputs = self._itm_swapping_based(visual_embeddings,
                                                   img_pos_feat=None,
                                                   input_ids=input_ids)
            visual_embeddings, img_pos_feat, itm_targets = itm_outputs

        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            image_text_alignment,
        )

        output_dict: Dict[str, Tensor] = {}
        if not torch.jit.is_scripting():
            if self.output_attentions:
                output_dict["attention_weights"] = attention_weights

            if self.output_hidden_states:
                output_dict["sequence_output"] = sequence_output
                output_dict["pooled_output"] = pooled_output
        else:
            assert not (
                self.output_attentions or self.output_hidden_states
            ), "output_attentions or output_hidden_states not supported in script mode"

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        if masked_lm_labels is not None:
            output_dict["logits"] = prediction_scores
            prediction_scores_t = prediction_scores[:, :input_ids.size(1), :]
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.contiguous().view(-1, self.vocab_size),
                masked_lm_labels.contiguous().view(-1),
            )
            output_dict["masked_lm_loss"] = masked_lm_loss
            output_dict["loss"] = masked_lm_loss

            if "itm" in self.config.multimodal_pretrain_objectives:
                seq_relationship_score = seq_relationship_score[:, 0, :]
                itm_loss = F.cross_entropy(seq_relationship_score, itm_targets,
                                           # reduction='none')
                                           )
                output_dict["itm_loss"] = itm_loss
                output_dict["loss"] += itm_loss

        return output_dict

    def _itm_swapping_based(self, images, img_pos_feat=None,
                            input_ids=None, compute_loss=True):
        # Perform swapping-based multimodal alignment objective.
        bz, img_len = images.size(0), images.size(1)
        images_if_swapped = torch.zeros(bz, img_len)
        swapping_based_nsp_labels = []
        new_images = []
        if "itm" in self.config.multimodal_pretrain_objectives:
            for i in range(bz):
                image_ = images[i].clone()  # L x D
                image_lenwise_sum = torch.sum(image_, dim=-1)
                # TODO: Since our visual mask token is 0.
                # non_zero_images = image_lenwise_sum.nonzero().t()[0]
                non_zero_images = torch.nonzero(image_lenwise_sum, as_tuple=False).t()[0]

                if len(non_zero_images) == 0:
                    swapping_based_nsp_labels.append(1)
                    continue

                sample_batch_idx = i + 1
                if i == bz - 1:
                   sample_batch_idx = 0
                image_cands_ = images[sample_batch_idx]
                image_cands_lenwise_sum = torch.sum(image_cands_, dim=-1)
                # non_zero_image_cands_ = image_cands_lenwise_sum.nonzero().t()[0]
                non_zero_image_cands_ = torch.nonzero(image_cands_lenwise_sum, as_tuple=False).t()[0]
                if len(non_zero_image_cands_) == 0:
                    swapping_based_nsp_labels.append(1)
                    continue

                non_zero_image_cands_ = non_zero_image_cands_.detach().cpu().numpy().astype(int)
                non_zero_images = non_zero_images.detach().cpu().numpy().astype(int)

                # TODO: Prevent swapping the already swapped images.
                non_zero_image_cands_ = set(list(non_zero_image_cands_))
                images_if_swapped_i = torch.nonzero(
                    images_if_swapped[
                        sample_batch_idx],
                        as_tuple=False).t()[0].detach().cpu().numpy().astype(int)
                images_if_swapped_i = set(list(images_if_swapped_i))
                non_zero_image_cands_ -= images_if_swapped_i
                non_zero_image_cands_ = list(non_zero_image_cands_)
                if len(non_zero_image_cands_) == 0:
                    swapping_based_nsp_labels.append(1)
                    continue

                chose_index = np.random.choice(non_zero_image_cands_)
                swapped_index = np.random.choice(non_zero_images)

                # Probability of swapping.
                if_swap = np.random.rand()
                if if_swap > self.swapping_based_nsp_prob:
                    # image_[swapped_index] = image_cands_[chose_index]
                    image_[swapped_index] = image_cands_[swapped_index]
                    swapping_based_nsp_labels.append(0)
                    images_if_swapped[i][swapped_index] = 1
                    if self.config.include_num_img_regional_features is not None:
                        img_regional_features[i][swapped_index] = \
                            img_regional_features[
                                sample_batch_idx][chose_index]
                else:
                    swapping_based_nsp_labels.append(1)

                # images[i] = image_
                new_images.append(image_)

            images = torch.stack(new_images)
            swapping_based_nsp_labels = torch.Tensor(
                swapping_based_nsp_labels).type_as(input_ids)
        elif "whole_itm" in self.config.multimodal_pretrain_objectives:
            for i in range(bz):
                sample_batch_idx = i + 1
                if i == bz - 1:
                   sample_batch_idx = 0

                if_swap = np.random.rand()
                if if_swap > self.swapping_based_nsp_prob:
                    image_ = images[sample_batch_idx]
                    swapping_based_nsp_labels.append(0)
                else:
                    swapping_based_nsp_labels.append(1)
                new_images.append(image_)

            images = torch.stack(new_images)
            swapping_based_nsp_labels = torch.Tensor(
                swapping_based_nsp_labels).type_as(input_ids)
            
        return images, img_pos_feat, swapping_based_nsp_labels


class VisualBERTForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        if "pooler_strategy" not in self.config.__dict__:
            self.pooler_strategy = "default"
        else:
            self.pooler_strategy = self.config.pooler_strategy
        # self.pooler_strategy = self.config.get("pooler_strategy", "default")

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        # self.bert_config = BertConfig.from_dict(
        #     OmegaConf.to_container(self.config, resolve=True)
        # )
        self.bert_config = config
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    "pretrained_models/bert", "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        # self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        if self.config.training_head_type == "nlvr2":
            self.bert.config.hidden_size *= 2

        # self.classifier = nn.Sequential(
        #     BertPredictionHeadTransform(self.bert.config),
        #     nn.Linear(self.bert.config.hidden_size, self.config.num_labels),
        # )
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

        # Loss.
        self.sort_loss = nn.CrossEntropyLoss()

        # Heatmap predictions.
        self.hierarchical_version = config.hierarchical_version
        if (self.hierarchical_version != "v0"
            and "p" not in self.hierarchical_version):
            self.heatmap = HeatMapOutput(config)
            print(self.heatmap)
        elif "p" in self.hierarchical_version:
            self.pointer = PointerOutput(config)
            print(self.pointer)

        # Parameters for objectives.
        self.itm_ot_lambda = 0.1
        self.swapping_based_nsp_prob = 0.5

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        downstream_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        itm_targets = None
        if self.config.hl_include_objectives is not None:
            if "itm" in self.config.hl_include_objectives:
                itm_outputs = self._itm_swapping_based(visual_embeddings,
                                                       img_pos_feat=None,
                                                       input_ids=input_ids)
                visual_embeddings, img_pos_feat, itm_targets = itm_outputs

        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            visual_embeddings_type,
            image_text_alignment,
        )

        if self.training_head_type == "nlvr2":
            # 2B * H => B * 2H
            b, h = pooled_output.size()
            pooled_output = torch.cat(
                [pooled_output[: b // 2], pooled_output[b // 2 :]], dim=1
            )

        output_dict: Dict[str, Tensor] = {}
        if not torch.jit.is_scripting():
            if self.output_attentions:
                output_dict["attention_weights"] = attention_weights

            if self.output_hidden_states:
                output_dict["sequence_output"] = sequence_output
                output_dict["pooled_output"] = pooled_output
        else:
            assert not (
                self.output_attentions or self.output_hidden_states
            ), "output_attentions or output_hidden_states not supported in script mode"

        if self.pooler_strategy == "vqa":
            # In VQA2 pooling strategy, we use representation from second last token
            index_to_gather = input_mask.sum(1) - 2
            pooled_output = torch.gather(
                sequence_output,
                1,
                index_to_gather.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
            )

        # pooled_output = self.dropout(pooled_output)

        # Heatmap handlings.
        if (self.hierarchical_version != "v0"
            and "p" not in self.hierarchical_version):
            heatmap_batch = {
                "input_ids": input_ids,
                "labels": downstream_labels,
            }
            heatmap_outputs = self.heatmap(heatmap_batch, sequence_output,
                                           itm_repr=(pooled_output, itm_targets))
            if downstream_labels is not None:
                heatmap_loss, heatmap_logits = heatmap_outputs
                output_dict["scores"] = heatmap_logits
                output_dict["logits"] = heatmap_logits
                output_dict["loss"] = heatmap_loss
            else:
                heatmap_logits = heatmap_outputs[0]
                output_dict["scores"] = heatmap_logits
                output_dict["logits"] = heatmap_logits
                output_dict["loss"] = None
            return output_dict

        elif "p" in self.hierarchical_version:
            pointer_batch = {
                "input_ids": input_ids,
                "labels": downstream_labels,
            }
            pointer_outputs = self.pointer(pointer_batch, sequence_output,
                                           itm_repr=(pooled_output, itm_targets))
            if downstream_labels is not None:
                pointer_loss, pointer_logits = pointer_outputs
                output_dict["scores"] = pointer_logits
                output_dict["logits"] = pointer_logits
                output_dict["loss"] = pointer_loss
            else:
                pointer_logits = pointer_outputs[0]
                output_dict["scores"] = pointer_logits
                output_dict["logits"] = pointer_logits
                output_dict["loss"] = None
            return output_dict
            
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict["scores"] = reshaped_logits
        output_dict["logits"] = reshaped_logits

        if downstream_labels is not None:
            sort_loss = self.sort_loss(reshaped_logits, downstream_labels)
            output_dict["loss"] = sort_loss
        else:
            output_dict["loss"] = None

        output_dict["sequence_output"] = sequence_output
        output_dict["pooled_output"] = pooled_output
        return output_dict

    def _itm_swapping_based(self, images, img_pos_feat=None,
                            input_ids=None, compute_loss=True):
        # Perform swapping-based multimodal alignment objective.
        bz, img_len = images.size(0), images.size(1)
        images_if_swapped = torch.zeros(bz, img_len)
        swapping_based_nsp_labels = []
        new_images = []
        for i in range(bz):
            image_ = images[i].clone()  # L x D
            image_lenwise_sum = torch.sum(image_, dim=-1)
            # TODO: Since our visual mask token is 0.
            # non_zero_images = image_lenwise_sum.nonzero().t()[0]
            non_zero_images = torch.nonzero(image_lenwise_sum, as_tuple=False).t()[0]

            if len(non_zero_images) == 0:
                swapping_based_nsp_labels.append(1)
                continue

            sample_batch_idx = i + 1
            if i == bz - 1:
               sample_batch_idx = 0
            image_cands_ = images[sample_batch_idx]
            image_cands_lenwise_sum = torch.sum(image_cands_, dim=-1)
            # non_zero_image_cands_ = image_cands_lenwise_sum.nonzero().t()[0]
            non_zero_image_cands_ = torch.nonzero(image_cands_lenwise_sum, as_tuple=False).t()[0]
            if len(non_zero_image_cands_) == 0:
                swapping_based_nsp_labels.append(1)
                continue

            non_zero_image_cands_ = non_zero_image_cands_.detach().cpu().numpy().astype(int)
            non_zero_images = non_zero_images.detach().cpu().numpy().astype(int)

            # TODO: Prevent swapping the already swapped images.
            non_zero_image_cands_ = set(list(non_zero_image_cands_))
            images_if_swapped_i = torch.nonzero(
                images_if_swapped[
                    sample_batch_idx],
                    as_tuple=False).t()[0].detach().cpu().numpy().astype(int)
            images_if_swapped_i = set(list(images_if_swapped_i))
            non_zero_image_cands_ -= images_if_swapped_i
            non_zero_image_cands_ = list(non_zero_image_cands_)
            if len(non_zero_image_cands_) == 0:
                swapping_based_nsp_labels.append(1)
                continue

            chose_index = np.random.choice(non_zero_image_cands_)
            swapped_index = np.random.choice(non_zero_images)

            # Probability of swapping.
            if_swap = np.random.rand()
            if if_swap > self.swapping_based_nsp_prob:
                image_[swapped_index] = image_cands_[chose_index]
                swapping_based_nsp_labels.append(0)
                images_if_swapped[i][swapped_index] = 1
                if self.config.include_num_img_regional_features is not None:
                    img_regional_features[i][swapped_index] = \
                        img_regional_features[
                            sample_batch_idx][chose_index]
            else:
                swapping_based_nsp_labels.append(1)

            # images[i] = image_
            new_images.append(image_)

        images = torch.stack(new_images)
        swapping_based_nsp_labels = torch.Tensor(
            swapping_based_nsp_labels).type_as(input_ids)
        return images, img_pos_feat, swapping_based_nsp_labels


@registry.register_model("visual_bert")
class VisualBERT(BaseModel):
    def __init__(self, config, vision_model=None, tokenizer=None,
                 multimodal_text_part=False, multimodal_img_part=False,
                 additional_config=None):
        # Change the head type.
        if additional_config is not None:
            if "training_head_type" in additional_config.__dict__:
                config.training_head_type = additional_config.training_head_type
        super().__init__(config)
        self.config = config
        self.training_head_type: str = self.config.training_head_type

        # Vision model.
        self.vision_model = vision_model
        if self.vision_model is not None:
            # Remove the final FC layer.
            try:
                self.num_img_dim = self.vision_model.fc.in_features
                self.vision_model.fc = nn.Identity()
            except:
                # Detectron2 models
                self.num_img_dim = additional_config.v_feature_size
                self.config.visual_embedding_dim = self.num_img_dim
            self.freeze_vision_model = False
        self.multimodal_text_part = multimodal_text_part
        self.multimodal_img_part = multimodal_img_part

        # Tokenizer.
        self.tokenizer = tokenizer

        # Update the config
        # self.config.__dict__.update(additional_config.__dict__)
        additional_config.__dict__.update(self.config)
        print(self.config)
        self.config = additional_config

    @classmethod
    def config_path(cls):
        return "configs/models/visual_bert/pretrain.yaml"

    def build(self):
        if self.training_head_type == "pretraining":
            if "v_feature_size" in self.config.__dict__:
                self.config.visual_embedding_dim = self.config.v_feature_size
            self.model = VisualBERTForPretraining(self.config)
        else:
            self.model = VisualBERTForClassification(self.config)

        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def flatten(
        self,
        sample_list: Dict[str, Tensor],
        to_be_flattened: List[str],
        to_be_flattened_dim: List[str],
    ) -> Dict[str, Tensor]:
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])
        return sample_list

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        sample_list["visual_embeddings_type"] = torch.zeros_like(
            sample_list["image_mask"]
        )
        attention_mask = torch.cat(
            (sample_list["input_mask"], sample_list["image_mask"]), dim=-1
        )
        sample_list["attention_mask"] = attention_mask

        if self.training_head_type == "pretraining":
            assert sample_list["masked_lm_labels"].size(-1) == sample_list[
                "input_mask"
            ].size(-1)
            new_lm_labels = torch.ones_like(attention_mask) * -1
            size_masked_lm_labels = sample_list["masked_lm_labels"].size()
            assert len(size_masked_lm_labels) == 2
            new_lm_labels[
                : size_masked_lm_labels[0], : size_masked_lm_labels[1]
            ] = sample_list["masked_lm_labels"]
            sample_list["masked_lm_labels"] = new_lm_labels

        return sample_list

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "token_type_ids", "input_mask", "image_mask"]
        to_be_flattened_dim = ["visual_embeddings"]

        if self.training_head_type == "pretraining":
            to_be_flattened.append("masked_lm_labels")

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def update_sample_list_based_on_head(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        print(sample_list.keys())
        bert_input_ids = sample_list["input_ids"]
        # bert_input_mask = sample_list["input_mask"]
        bert_input_mask = sample_list["attention_mask"]
        # bert_input_type_ids = sample_list["segment_ids"]
        bert_input_type_ids = sample_list["token_type_ids"]

        if self.training_head_type == "nlvr2":
            if not torch.jit.is_scripting():
                bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
                bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
                bert_input_type_ids = torch.cat(
                    [bert_input_type_ids, bert_input_type_ids]
                )

                # image input
                img0 = getattr(sample_list, "img0", {})
                image_feat_variable_0 = getattr(img0, "image_feature_0", None)
                img1 = getattr(sample_list, "img1", {})
                image_feat_variable_1 = getattr(img1, "image_feature_0", None)
                image_feat_variable = torch.cat(
                    [image_feat_variable_0, image_feat_variable_1]
                )

                image_info = getattr(img0, "image_info_0", {})
                image_dim_variable_0 = getattr(image_info, "max_features", None)
                image_info = getattr(img1, "image_info_0", {})
                image_dim_variable_1 = getattr(image_info, "max_features", None)
                image_dim_variable = torch.cat(
                    [image_dim_variable_0, image_dim_variable_1]
                )
            else:
                raise RuntimeError("nlvr2 head doesn't support scripting as of now")
        else:

            if not torch.jit.is_scripting():
                image_info = getattr(sample_list, "image_info_0", {})
                image_dim_variable = getattr(image_info, "max_features", None)
                image_feat_variable = getattr(sample_list, "image_feature_0", None)
            else:
                image_feat_variable = sample_list["image_feature_0"]
                image_dim_variable = None

        if image_dim_variable is None:
            image_dim_variable = sample_list["image_feature_0"].new_full(
                size=(image_feat_variable.size(0), 1),
                fill_value=image_feat_variable.size(1),
            )

        sample_list["visual_embeddings"] = image_feat_variable
        sample_list["image_dim"] = image_dim_variable
        sample_list["input_ids"] = bert_input_ids
        sample_list["input_mask"] = bert_input_mask
        sample_list["token_type_ids"] = bert_input_type_ids
        return sample_list

    def add_custom_params(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings = sample_list["visual_embeddings"]
        image_dim = sample_list["image_dim"]

        if self.training_head_type == "pretraining":
            # pretraining labels
            sample_list["masked_lm_labels"] = sample_list["lm_label_ids"]
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        image_mask = torch.arange(
            visual_embeddings.size(-2), device=visual_embeddings.device
        ).expand(visual_embeddings.size()[:-1])
        if len(image_dim.size()) < len(image_mask.size()):
            image_dim = image_dim.unsqueeze(-1)
            assert len(image_dim.size()) == len(image_mask.size())
        image_mask = image_mask < image_dim
        sample_list["image_mask"] = image_mask.long()

        return sample_list

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def get_proper_sample_list_based_on_head(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        batch = sample_list
        bz, txt_len = batch["input_ids"].size()
        new_batch = get_detailed_input_feats(batch, self.tokenizer, self.config)
        batch["position_ids"] = new_batch["position_ids"]
        batch["attn_masks"] = new_batch["attn_masks"] 
        # batch["gather_index"] = new_batch["gather_index"]

        # Visual embedding.
        if self.vision_model is not None and batch["images"].ndim > 3:
            images = batch["images"]
            bz, img_len, C, H, W = images.size()
            images = torch.reshape(images, (bz*img_len, C, H, W)).float()
            images = self.vision_model(images)
            if type(images) == tuple:
                images, img_regional_features = images
                img_regional_features = torch.reshape(
                    img_regional_features,
                    (bz, img_len, self.config.include_num_img_regional_features,  # -1,
                    self.num_img_dim))
            if self.freeze_vision_model:
                images = images.detach()

            images = torch.reshape(images, (bz, img_len, self.num_img_dim))

            if self.config.include_num_img_regional_features is not None:
                img_regional_features = img_regional_features.float()
                new_images = []
                for i in range(bz):
                    curr_img = images[i]
                    curr_img_region_ = []
                    for j in range(img_len):
                        curr_seq_img = curr_img[j].unsqueeze(0)
                        curr_regional_features = img_regional_features[i, j, :, :]
                        curr_new_img = torch.cat([curr_seq_img, curr_regional_features])
                        curr_img_region_.append(curr_new_img)
                    curr_img_region_ = torch.cat(curr_img_region_)
                    new_images.append(curr_img_region_)
                new_images = torch.stack(new_images)
                images = new_images

            batch["img_feat"] = images
            batch["visual_embeddings"] = images
        else:
            batch["img_feat"] = None
        batch['img_pos_feat'] = None

        if batch["img_feat"] is not None:
            batch["visual_embeddings_type"] = torch.zeros(bz,
                batch["img_feat"].size(1)).type_as(batch["input_ids"])

        batch["token_type_ids"] = torch.zeros(bz,
            batch["input_ids"].size(1)).type_as(batch["input_ids"])

        # Attention masks handling.
        # TODO Make sure the followings.
        if (not self.multimodal_text_part and not self.multimodal_img_part
            and not self.config.img_text_paired_coattention):
            additional_attn = torch.ones(bz, batch["img_feat"].size(1)).type_as(
                batch["attn_masks"])
            batch["attn_masks"] = torch.cat([batch["attn_masks"],
                                             additional_attn], dim=-1)
            additional_type_ids = torch.ones(bz, batch["img_feat"].size(1)).type_as(
                batch["input_ids"])
            # batch["token_type_ids"] = torch.cat([batch["token_type_ids"],
            #     additional_type_ids], dim=-1)

        if self.multimodal_text_part:
            batch["images"] = None
            batch["img_feat"] = None
            batch["visual_embeddings"] = None
            batch["visual_embeddings_type"] = None
        if self.multimodal_img_part:
            batch["input_ids"] = None
            batch["attn_masks"] = torch.ones(bz, batch["img_feat"].size(1)).type_as(
                batch["attn_masks"])
            # batch["token_type_ids"] = torch.ones(bz, batch["img_feat"].size(1)).type_as(
            #     batch["input_ids"])

        batch["input_mask"] = batch["attn_masks"]
        batch["attention_mask"] = batch["attn_masks"]

        batch = {x: batch[x].to(batch['attn_masks'].device)
                 if batch[x] is not None else batch[x] for x in batch}

        batch["dataset_name"] = "recipeqa"
        batch["dataset_type"] = "qa"

        return batch

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if torch.jit.is_scripting():
            assert (
                "image_feature_0" in sample_list
            ), "Key 'image_feature_0' is required in TorchScript model"

        # sample_list = self.update_sample_list_based_on_head(sample_list)
        # sample_list = self.add_custom_params(sample_list)
        # sample_list = self.flatten_for_bert(sample_list)
        # sample_list = self.add_post_flatten_params(sample_list)
        sample_list = self.get_proper_sample_list_based_on_head(sample_list)
        if self.training_head_type != "pretraining":
            if "labels" in sample_list:
                sample_list["downstream_labels"] = sample_list["labels"]
            else:
                sample_list["downstream_labels"] = None

        # print(sample_list["input_ids"].size())
        # print(sample_list["input_mask"].size())
        # print(sample_list["attention_mask"].size())
        # print(sample_list["token_type_ids"].size())
        # print(sample_list["visual_embeddings"].size())
        # print(sample_list["visual_embeddings_type"].size())

        output_dict = self.model(
            sample_list["input_ids"],
            sample_list["input_mask"],
            sample_list["attention_mask"],
            sample_list["token_type_ids"],
            sample_list["visual_embeddings"],
            sample_list["visual_embeddings_type"],
            getattr_torchscriptable(sample_list, "image_text_alignment", None),
            getattr_torchscriptable(sample_list, "masked_lm_labels", None),
            getattr_torchscriptable(sample_list, "downstream_labels", None),
        )

        if self.training_head_type == "pretraining":
            if not torch.jit.is_scripting():
                loss_key = "{}/{}".format(
                    sample_list["dataset_name"], sample_list["dataset_type"]
                )
                output_dict["losses"] = {}
                output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                    "masked_lm_loss"
                )
            else:
                raise RuntimeError("Pretraining head can't be used in script mode.")

        loss, logits = output_dict["loss"], output_dict["logits"]
        if loss is not None:
            return loss, logits

        if self.config.wrapper_model_type == "berson":
            return output_dict["sequence_output"], output_dict["pooled_output"]

        return (logits, )
        # return output_dict

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
