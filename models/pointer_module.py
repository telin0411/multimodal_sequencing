import os
import math
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch import nn
import torch.nn.functional as F

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertForMaskedLM,
    RobertaForCausalLM,
)
from transformers.file_utils import is_sklearn_available, requires_sklearn
from torch.nn import LayerNorm as BertLayerNorm
from .beam import Beam

logger = logging.getLogger(__name__)


class PointerOutput(nn.Module):
    
    def __init__(self, config):
        super(PointerOutput, self).__init__()

        self.config = config
        config.beam_size = None

        if config.hierarchical_version == "p1":
            self.lstm_decoder = LSTMDecoder(config.hidden_size,
                                            config.max_story_length)
            self.lstm_pointer = LSTMPointerModule(self.lstm_decoder,
                                                  config.beam_size)
        else:
            if config.hidden_size == 1024:
                decoder_config_path = "pretrained_models/roberta/large/decoder_config.json"
            elif config.hidden_size == 768:
                decoder_config_path = "pretrained_models/roberta/base/decoder_config.json"
            else:
                raise ValueError

            assert os.path.exists(decoder_config_path)

            decoder_config = AutoConfig.from_pretrained(decoder_config_path)
            # We are not using the ordinary vocabulary.
            decoder_config.out_features = config.hidden_size
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True

            causal_lm = RobertaForCausalLM(decoder_config)
            self.decoder = causal_lm.roberta.encoder

            self.index_classifier = SimpleClassifier(
                config.hidden_size, config.hidden_size,
                config.max_story_length, 0.5
            )

        # TODO HL aux objectives.
        self.hl_include_objectives = config.hl_include_objectives
        logging.info("Aux using: {}".format(self.hl_include_objectives))
        if self.hl_include_objectives is not None:
            assert type(self.hl_include_objectives) == list
            if ("binary" in self.hl_include_objectives
                or "pairwise" in self.hl_include_objectives):
                self.hl_bin_pred_layer = SimpleClassifier(
                    config.hidden_size * 1,
                    config.hidden_size,
                    1, 0.5
                )
                self.hl_bin_pred_crit = torch.nn.CrossEntropyLoss()
                self.hl_bin_sparse_prob = 1.5  # 1.0
                if self.hl_bin_sparse_prob < 1.0:
                    self.hl_bin_sparse_pos = []

            if "binary_cross_modal" in self.hl_include_objectives:
                raise NotImplementedError("Not done yet!")
                self.hl_binx_pred_layer = SimpleClassifier(
                    config.hidden_size * 1 + config.hidden_size,
                    config.hidden_size,
                    2, 0.5
                )
                self.hl_binx_pred_crit = torch.nn.CrossEntropyLoss()
                self.hl_binx_sparse_prob = 1.5  # 1.0
                if self.hl_binx_sparse_prob < 1.0:
                    self.hl_binx_sparse_pos = []

            if "head" in self.hl_include_objectives:
                self.hl_head_pred_layer = SimpleClassifier(
                    config.hidden_size * 1,
                    config.hidden_size,
                    1, 0.5
                )
                self.hl_head_pred_crit = torch.nn.CrossEntropyLoss()

            if "mlm" in self.hl_include_objectives:
                raise NotImplementedError("Not done yet!")
                self.mlm_loss_fct = CrossEntropyLoss(
                    ignore_index=config.mlm_ignore_index)

            if "itm" in self.hl_include_objectives:
                if "swapping_based_nsp" in config.__dict__:
                    self.swapping_based_nsp = config.swapping_based_nsp
                    self.swapping_based_nsp_prob = 0.5
                    self.itm_loss_fct = torch.nn.CrossEntropyLoss()
                else:
                    raise ValueError("No `swapping_based_nsp` in config.")
                self.seq_relationship = nn.Linear(config.hidden_size, 2)
            else:
                self.swapping_based_nsp = False

            if "cross_modal_dependence" in self.hl_include_objectives:
                raise NotImplementedError("Can't perform "
                    "`cross_modal_dependence` with pointer network yet!")
                self.cross_modal_dependence_prediction = SimpleClassifier(
                    config.hidden_size, config.hidden_size * 1,
                    self.config.max_story_length, 0.5
                )
                self.cross_modal_dependence_loss = torch.nn.MSELoss()

            if "pointer_pairwise_ranking" in self.hl_include_objectives:
                raise NotImplementedError("Can't perform "
                    "`pointer_pairwise_ranking` with pointer network yet!")
                self.hm_pw_ranking_loss = torch.nn.MarginRankingLoss(margin=0.2)

        # TODO Higher-level layers.
        self.additional_hl = False
        pass

        # TODO More random stuff.
        self.fusion_method = "mul"
        self.pointer_late_fusion = False

        # Losses:
        # TODO KL-Div.
        self.use_kl_div = False

        self.pointer_sort_loss = torch.nn.CrossEntropyLoss()

        # TODO: If perform loss and inference in a looping setup.
        self.for_loop = False

    def forward(self, batch, sequence_output, itm_repr=None):
        """
            batch (dict): Dict of inputs.
        """
        input_ids = batch["input_ids"]
        bz, text_len = input_ids.size()

        cls_repr_pos = []
        for i in range(bz):
            cls_repr_pos_i = torch.nonzero(
                input_ids[i]==self.config.cls_id, as_tuple=False)
            cls_repr_pos.append(cls_repr_pos_i)

        cls_pointer = []
        # print(cls_repr_pos[0])
        # print(batch["input_ids"][0])
        # print(sequence_output.size())
        # print(self.hl_include_objectives);raise

        # TODO See if we need to use additional hl layers.
        if self.additional_hl:
            pass

        # Obtaining linguistics and visual outputs.
        if type(sequence_output) != tuple:
            sequence_output_t = sequence_output[:, :text_len]
            sequence_output_v = sequence_output[:, text_len:]
        else:
            sequence_output_t, sequence_output_v = sequence_output
            raise NotImplementedError("Not done yet!")

        # TODO: Auxiliary predictions.
        if self.hl_include_objectives is not None:
            hl_aux_predictions = [None] * len(self.hl_include_objectives)
            if "head" in self.hl_include_objectives:
                hl_aux_head_predictions = []
            if ("pairwise" in self.hl_include_objectives
                or "binary" in self.hl_include_objectives):
                hl_aux_bin_predictions = []
            if "binary_cross_modal" in self.hl_include_objectives:
                hl_aux_binx_predictions = []
            if "cross_modal_dependence" in self.hl_include_objectives:
                hl_aux_x_dep_predictions = []

        for i in range(bz):
            if not self.additional_hl:
                cls_repr_pos_i = cls_repr_pos[i].squeeze()
                cls_repr_i = sequence_output[i][cls_repr_pos_i]
                if self.fusion_method == "img_only" or self.pointer_late_fusion:
                    raise NotImplementedError("Not done yet!")
            else:
                raise NotImplementedError("Not done yet!")

            if self.pointer_late_fusion:
                raise NotImplementedError("Not done yet!")

            if self.hl_include_objectives is not None:
                if len(self.hl_include_objectives) > 0:
                    for hl_aux_objective in self.hl_include_objectives:
                        if "head" == hl_aux_objective:
                            hl_aux_head_prediction_curr = self.hl_head_pred_layer(cls_repr_i)
                            hl_aux_head_prediction_curr = hl_aux_head_prediction_curr.squeeze()
                            hl_aux_head_predictions.append(hl_aux_head_prediction_curr)
                        elif ("pairwise" == hl_aux_objective
                              or "binary" == hl_aux_objective):
                            hl_aux_bin_predictions_tmp = []
                            for seq_i in range(len(cls_repr_pos_i)):
                                for seq_j in range(seq_i+1, len(cls_repr_pos_i)):
                                    cls_repr_seq_i = cls_repr_i[seq_i]
                                    cls_repr_seq_j = cls_repr_i[seq_j]
                                    cls_repr_seq_ij = torch.stack(
                                        [cls_repr_seq_i, cls_repr_seq_j])
                                    hl_aux_bin_prediction_curr = self.hl_bin_pred_layer(cls_repr_seq_ij)
                                    hl_aux_bin_prediction_curr = hl_aux_bin_prediction_curr.squeeze()
                                    hl_aux_bin_predictions_tmp.append(hl_aux_bin_prediction_curr)
                            hl_aux_bin_predictions_tmp = torch.stack(hl_aux_bin_predictions_tmp)

                            if self.hl_bin_sparse_prob < 1.0:
                                hl_bin_sparse_len = int(
                                    len(hl_aux_bin_predictions_tmp)
                                        * self.hl_bin_sparse_prob)
                                hl_bin_sparse_pos_tmp = np.random.choice(
                                    np.arange(hl_aux_bin_predictions_tmp.size(0)),
                                              hl_bin_sparse_len)
                                # TODO: Temporary!!!
                                hl_aux_bin_predictions_tmp = hl_aux_bin_predictions_tmp[
                                    hl_bin_sparse_pos_tmp]
                                # hl_aux_bin_predictions_tmp = hl_aux_bin_predictions_tmp[:3]
                                self.hl_bin_sparse_pos.append(hl_bin_sparse_pos_tmp)

                            hl_aux_bin_predictions.append(hl_aux_bin_predictions_tmp)

                        elif "binary_cross_modal" == hl_aux_objective:
                            hl_aux_binx_predictions_tmp = []
                            for seq_i in range(len(cls_repr_pos_i)):
                                for seq_j in range(seq_i+1, len(cls_repr_pos_i)):
                                    # Text modality.
                                    cls_repr_seq_i = cls_repr_i[seq_i]
                                    # Image modality.
                                    if self.config.include_num_img_regional_features  is not None:
                                        img_seq_j = (1 + self.config.include_num_img_regional_features) * seq_j
                                        raise NotImplementedError("Not debugged yet!")
                                    else:
                                        img_seq_j = seq_j
                                    cls_repr_seq_j = sequence_output_v[i][img_seq_j]
                                    # Stacking.
                                    cls_repr_seq_ij = torch.cat(
                                        [cls_repr_seq_i, cls_repr_seq_j], dim=-1)
                                    hl_aux_binx_prediction_curr = self.hl_binx_pred_layer(cls_repr_seq_ij)
                                    hl_aux_binx_prediction_curr = hl_aux_binx_prediction_curr.squeeze()
                                    hl_aux_binx_predictions_tmp.append(hl_aux_binx_prediction_curr)
                            hl_aux_binx_predictions_tmp = torch.stack(hl_aux_binx_predictions_tmp)

                            if self.hl_binx_sparse_prob < 1.0:
                                hl_binx_sparse_len = int(
                                    len(hl_aux_binx_predictions_tmp)
                                        * self.hl_binx_sparse_prob)
                                hl_binx_sparse_pos_tmp = np.random.choice(
                                    np.arange(hl_aux_binx_predictions_tmp.size(0)),
                                              hl_binx_sparse_len)
                                # TODO: Temporary!!!
                                hl_aux_binx_predictions_tmp = hl_aux_binx_predictions_tmp[
                                    hl_binx_sparse_pos_tmp]
                                # hl_aux_binx_predictions_tmp = hl_aux_binx_predictions_tmp[:3]
                                self.hl_binx_sparse_pos.append(hl_binx_sparse_pos_tmp)

                            hl_aux_binx_predictions.append(hl_aux_binx_predictions_tmp)

                        elif "cross_modal_dependence" == hl_aux_objective:
                            hl_aux_x_dep_predictions_tmp = []
                            for img_seq_i in range(len(cls_repr_pos_i)):
                                # Image modality.
                                if self.config.include_num_img_regional_features  is not None:
                                    img_seq_i_real = (1 + self.config.include_num_img_regional_features) * img_seq_i
                                    raise NotImplementedError("Not debugged yet!")
                                else:
                                    img_seq_i_real = img_seq_i
                                img_repr_seq_i = sequence_output_v[i][img_seq_i_real]
                                hl_aux_x_dep_predictions_tmp.append(img_repr_seq_i)
                            hl_aux_x_dep_predictions_tmp = torch.stack(hl_aux_x_dep_predictions_tmp)
                            hl_aux_x_dep_predictions_tmp = self.cross_modal_dependence_prediction(
                                hl_aux_x_dep_predictions_tmp)
                            hl_aux_x_dep_predictions.append(hl_aux_x_dep_predictions_tmp)

                pass  # End of auxiliary predictions.

            # Concatenate the pointer representations.
            cls_pointer.append(cls_repr_i)

        cls_pointer = torch.stack(cls_pointer)
        self.cls_pointer = cls_pointer

        # Feed to the decoder.
        if self.config.hierarchical_version == "p1":
            encoder_out = cls_pointer
            encoder_cls = sequence_output_t[:, 0, :]
            ptr_outs = self.lstm_pointer(encoder_out, encoder_cls, batch["labels"])
            pointer_sorted_outputs, pointer_loss = ptr_outs
        else:
            pointer_sorted_outputs = torch.zeros(bz, self.config.max_story_length,
                                                 dtype=torch.long)
            pointer_loss = 0
            teacher_force_ratio = 0.5

            if not self.for_loop:
                encoder_hidden_states = cls_pointer
                decoder_inputs_embeds = cls_pointer
                decoder_outputs = self.decoder(
                    encoder_hidden_states=encoder_hidden_states,
                    hidden_states=decoder_inputs_embeds
                )
                decoder_repr = decoder_outputs[0]
                
                use_self_repr = False

                if not use_self_repr:
                    decoder_predictions = self.index_classifier(decoder_repr)
                else:
                    # print("cls_pointer_T", cls_pointer_T.size())
                    # print("decoder_repr", decoder_repr.size())
                    cls_pointer_T = torch.transpose(cls_pointer, 1, 2).detach()
                    decoder_predictions = torch.matmul(decoder_repr, cls_pointer_T)
                # print("decoder_predictions", decoder_predictions.size())
                pointer_sorted_outputs = F.softmax(decoder_predictions, dim=1).argmax(-1)
                pointer_loss = self.pointer_sort_loss(decoder_predictions, batch["labels"])
            else:
                encoder_hidden_states = cls_pointer
                decoder_inputs_embeds = sequence_output_t[:, 0, :].unsqueeze(1)
                for t in range(self.config.max_story_length):
                    # TODO: Causal attention mask?
                    # print(decoder_inputs_embeds.size())
                    decoder_outputs = self.decoder(
                        encoder_hidden_states=encoder_hidden_states,
                        # inputs_embeds=decoder_inputs_embeds,
                        hidden_states=decoder_inputs_embeds
                    )
                    decoder_repr = decoder_outputs[0][:, t, :]
                    decoder_predictions = self.index_classifier(decoder_repr)
                    index_predictions = F.softmax(decoder_predictions, dim=1).argmax(1)

                    # Pick next index
                    # If teacher force the next element will we the ground truth
                    # otherwise will be the predicted value at current timestep.
                    teacher_force = random.random() < teacher_force_ratio
                    idx = batch["labels"][:, t] if teacher_force else index_predictions

                    next_decoder_inputs_embeds = []
                    for b in range(bz):
                        curr_pred_idx = index_predictions[b]
                        next_decoder_inputs_embeds.append(cls_pointer[:, curr_pred_idx, :])
                    next_decoder_inputs_embeds = torch.stack(next_decoder_inputs_embeds, dim=1)
                    decoder_inputs_embeds = torch.cat(
                        # [decoder_inputs_embeds, decoder_repr.unsqueeze(1)], dim=1
                        [decoder_inputs_embeds, next_decoder_inputs_embeds], dim=1,
                    )

                    # Compute loss.
                    pointer_loss_t = self.pointer_sort_loss(decoder_predictions,
                                                            batch["labels"][:, t])
                    pointer_loss += pointer_loss_t
                    pointer_sorted_outputs[:, t] = index_predictions

        # print(pointer_sorted_outputs)
        # print(pointer_loss);raise

        # pointer_sorted_outputs = pointer_sorted_outputs.cpu().numpy()

        # Obtain auxiliary predictions.
        if self.hl_include_objectives is not None:
            if len(self.hl_include_objectives) > 0:
                for hl_aux_objective in self.hl_include_objectives:
                    aux_obj_list_idx = self.hl_include_objectives.index(hl_aux_objective)

                    if "head" == hl_aux_objective:
                        hl_aux_head_predictions = torch.stack(hl_aux_head_predictions)
                        hl_aux_predictions[aux_obj_list_idx] = hl_aux_head_predictions
                    elif ("pairwise" == hl_aux_objective
                          or "binary" == hl_aux_objective):
                        hl_aux_bin_predictions = torch.stack(hl_aux_bin_predictions)
                        hl_aux_predictions[aux_obj_list_idx] = hl_aux_bin_predictions
                    elif "binary_cross_modal" == hl_aux_objective:
                        hl_aux_binx_predictions = torch.stack(hl_aux_binx_predictions)
                        hl_aux_predictions[aux_obj_list_idx] = hl_aux_binx_predictions
                    elif "cross_modal_dependence" == hl_aux_objective:
                        hl_aux_x_dep_predictions = torch.stack(hl_aux_x_dep_predictions)
                        hl_aux_predictions[aux_obj_list_idx] = hl_aux_x_dep_predictions
            pass ####

        if "labels" in batch and batch["labels"] is not None:
            # The main pointer loss.
            loss = pointer_loss

            # TODO: Deal with auxiliary objectives.
            if self.hl_include_objectives is not None:
                if len(self.hl_include_objectives) > 0:
                    for h in range(len(self.hl_include_objectives)):
                        hl_aux_objective = self.hl_include_objectives[h]

                        if "pointer_pairwise_ranking" == hl_aux_objective:
                            hm_pw_target = []
                            hm_pw_input_1 = []
                            hm_pw_input_2 = []
                            for b in range(len(batch["labels"])):
                                label_ = list(batch["labels"][b].cpu().numpy())
                                # print(label_)
                                hm_pw_target_tmp = []
                                hm_pw_input_1_tmp = []
                                hm_pw_input_2_tmp = []
                                for seq_i in range(len(label_)):
                                    pos_i = label_[seq_i]
                                    if seq_i+1 >= len(label_):
                                        break
                                    pos_j = label_[seq_i+1]
                                    anchor = logits[b][pos_i][pos_j]
                                    for seq_j in range(len(label_)):
                                        if seq_j - seq_i == 1:  # Positive
                                            hm_pw_target_tmp.append(1)
                                        else:
                                            hm_pw_target_tmp.append(-1)
                                        pos_i = label_[seq_i]
                                        pos_j = label_[seq_j]
                                        # print(pos_i, pos_j, hm_pw_target_tmp[-1])
                                        pointer_ij = logits[b][pos_i][pos_j]
                                        hm_pw_input_1_tmp.append(anchor)
                                        hm_pw_input_2_tmp.append(pointer_ij)

                                hm_pw_target_tmp = torch.Tensor(
                                    hm_pw_target_tmp).type_as(batch["labels"])
                                hm_pw_input_1_tmp = torch.stack(hm_pw_input_1_tmp)
                                hm_pw_input_2_tmp = torch.stack(hm_pw_input_2_tmp)

                                hm_pw_target.append(hm_pw_target_tmp)
                                hm_pw_input_1.append(hm_pw_input_1_tmp)
                                hm_pw_input_2.append(hm_pw_input_2_tmp)
                            
                            hm_pw_target = torch.stack(hm_pw_target)
                            hm_pw_input_1 = torch.stack(hm_pw_input_1)
                            hm_pw_input_2 = torch.stack(hm_pw_input_2)
                            # print(hm_pw_target.size(), hm_pw_input_1.size(),
                            #       hm_pw_input_2.size())

                            pointer_pairwise_ranking_loss = self.hm_pw_ranking_loss(
                                hm_pw_input_1, hm_pw_input_2, hm_pw_target)
                            # print(pointer_pairwise_ranking_loss)

                            loss += pointer_pairwise_ranking_loss

                        elif "mlm_wo_loss" == hl_aux_objective:
                            pass

                        elif "mlm" == hl_aux_objective:
                            masked_lm_labels = batch["masked_lm_labels"]
                            masked_lm_loss = self.mlm_loss_fct(
                                linguistic_prediction.view(-1,
                                    self.config.vocab_size),
                                masked_lm_labels.view(-1),
                            )
                            loss += 0.05 * masked_lm_loss

                        elif "itm" == hl_aux_objective:
                            assert itm_repr is not None, "No itm representation!"
                            pooled_output, itm_targets = itm_repr
                            seq_relationship_prediction = self.seq_relationship(
                                pooled_output)
                            swapping_based_nsp_loss = self.itm_loss_fct(
                                seq_relationship_prediction,
                                itm_targets,
                            )
                            loss += 0.1 * swapping_based_nsp_loss

                        elif "head" == hl_aux_objective:
                            head_labels = batch["labels"][:, 0]
                            hl_aux_head_loss = self.hl_head_pred_crit(
                                hl_aux_predictions[h], head_labels)
                            loss += hl_aux_head_loss

                        elif ("pairwise" == hl_aux_objective
                              or "binary" == hl_aux_objective):
                            hl_bin_predictions = hl_aux_predictions[h]
                            bz, seq_len = batch["labels"].size()
                            for b in range(bz):
                                label_curr = batch["labels"][b]
                                label_index = torch.argsort(label_curr)
                                hl_bin_label = torch.zeros(
                                    hl_bin_predictions.size()[1]).type_as(
                                        batch["labels"])

                                bin_idx = 0
                                if self.hl_bin_sparse_prob < 1.0:
                                    hl_bin_label = []
                                for seq_i in range(seq_len):
                                    for seq_j in range(seq_i+1, seq_len):
                                        if label_index[seq_i] < label_index[seq_j]:
                                            if self.hl_bin_sparse_prob < 1.0:
                                                hl_bin_label.append(1)
                                            else:
                                                hl_bin_label[bin_idx]  = 1
                                        else:
                                            if self.hl_bin_sparse_prob < 1.0:
                                                hl_bin_label.append(0)
                                        bin_idx += 1

                                if self.hl_bin_sparse_prob < 1.0:
                                    hl_bin_label = torch.Tensor(hl_bin_label).type_as(batch["labels"])
                                    # TODO: Change this!!!
                                    # hl_bin_label = hl_bin_label[:3]
                                    hl_bin_label = hl_bin_label[self.hl_bin_sparse_pos[b]]

                                hl_aux_bin_loss_curr = self.hl_bin_pred_crit(hl_bin_predictions[b], hl_bin_label)
                                loss += hl_aux_bin_loss_curr
                                # loss += 0.1 * hl_aux_bin_loss_curr
                            pass
                                    
                        elif "binary_cross_modal" == hl_aux_objective:
                            hl_binx_predictions = hl_aux_predictions[h]
                            bz, seq_len = batch["labels"].size()
                            for b in range(bz):
                                label_curr = batch["labels"][b]
                                label_index = torch.argsort(label_curr)
                                hl_binx_label = torch.zeros(
                                    hl_binx_predictions.size()[1]).type_as(
                                        batch["labels"])

                                binx_idx = 0
                                if self.hl_binx_sparse_prob < 1.0:
                                    hl_binx_label = []
                                for seq_i in range(seq_len):
                                    for seq_j in range(seq_i+1, seq_len):
                                        if label_index[seq_i] < label_index[seq_j]:
                                            if self.hl_binx_sparse_prob < 1.0:
                                                hl_binx_label.append(1)
                                            else:
                                                hl_binx_label[binx_idx]  = 1
                                        else:
                                            if self.hl_binx_sparse_prob < 1.0:
                                                hl_binx_label.append(0)
                                        binx_idx += 1

                                if self.hl_binx_sparse_prob < 1.0:
                                    hl_binx_label = torch.Tensor(hl_binx_label).type_as(batch["labels"])
                                    # TODO: Change this!!!
                                    # hl_bin_label = hl_bin_label[:3]
                                    hl_binx_label = hl_binx_label[self.hl_binx_sparse_pos[b]]

                                hl_aux_binx_loss_curr = self.hl_binx_pred_crit(hl_binx_predictions[b], hl_binx_label)
                                loss += hl_aux_binx_loss_curr
                                    
                            pass

                        elif "cross_modal_dependence" == hl_aux_objective:
                            hl_binx_predictions = hl_aux_predictions[h]
                            cross_modal_dependence_preds = torch.sigmoid(
                                hl_binx_predictions)
                            cross_modal_dependence_logits = torch.transpose(
                                cross_modal_dependence_preds, 1, 2)
                            cross_modal_dependence_loss = self.cross_modal_dependence_loss(
                                pointer_labels, cross_modal_dependence_logits)
                            loss += cross_modal_dependence_loss

                pass  # End of all auxiliary losses. 

            return loss, pointer_sorted_outputs

        return (pointer_sorted_outputs, )


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


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


class LSTMAttention(nn.Module):
  def __init__(self, hidden_size, units):
    super(LSTMAttention, self).__init__()
    self.W1 = torch.nn.Linear(hidden_size, units, bias=False)
    self.W2 = torch.nn.Linear(hidden_size, units, bias=False)
    self.V =  torch.nn.Linear(units, 1, bias=False)

  def forward(self, encoder_out, decoder_hidden):
    # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    # decoder_hidden: (BATCH, HIDDEN_SIZE)

    # Add time axis to decoder hidden state
    # in order to make operations compatible with encoder_out
    # decoder_hidden_time: (BATCH, 1, HIDDEN_SIZE)
    decoder_hidden_time = decoder_hidden.unsqueeze(1)

    # uj: (BATCH, ARRAY_LEN, ATTENTION_UNITS)
    # Note: we can add the both linear outputs thanks to broadcasting
    uj = self.W1(encoder_out) + self.W2(decoder_hidden_time)
    uj = torch.tanh(uj)

    # uj: (BATCH, ARRAY_LEN, 1)
    uj = self.V(uj)

    # Attention mask over inputs
    # aj: (BATCH, ARRAY_LEN, 1)
    aj = F.softmax(uj, dim=1)

    # di_prime: (BATCH, HIDDEN_SIZE)
    di_prime = aj * encoder_out
    di_prime = di_prime.sum(1)
    
    return di_prime, uj.squeeze(-1)


class LSTMDecoder(nn.Module):
  def __init__(self, 
               hidden_size: int,
               attention_units: int = 10):
    super(LSTMDecoder, self).__init__()
    self.lstm = torch.nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
    self.attention = LSTMAttention(hidden_size, attention_units)

  def forward(self, x, hidden, encoder_out):
    # x: (BATCH, 1, 1) 
    # hidden: (1, BATCH, HIDDEN_SIZE)
    # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    
    # Get hidden states (not cell states) 
    # from the first and unique LSTM layer
    ht = hidden[0][0] # ht: (BATCH, HIDDEN_SIZE)

    # di: Attention aware hidden state -> (BATCH, HIDDEN_SIZE)
    di, att_w = self.attention(encoder_out, ht)
    
    # Append attention aware hidden state to our input
    # x: (BATCH, 1, 1 + HIDDEN_SIZE)
    x = torch.cat([di.unsqueeze(1), x], dim=2)
    
    # Generate the hidden state for next timestep

    _, hidden = self.lstm(x.contiguous(), hidden)
    return hidden, att_w


class LSTMPointerModule(nn.Module):

  def __init__(self, 
               decoder: nn.Module,
               beam_size=None):
    super(LSTMPointerModule, self).__init__()
    self.decoder = decoder
    self.beam_size = beam_size

  def forward(self, encoder_out, encoder_cls, y,
              teacher_force_ratio=.5):

    out = encoder_out
    hs = encoder_cls
    hs = (hs.unsqueeze(0).contiguous(), hs.unsqueeze(0).contiguous())
    # Accum loss throughout timesteps
    loss = 0

    # Save outputs at each timestep
    # outputs: (ARRAY_LEN, BATCH)
    outputs = torch.zeros(out.size(0), out.size(1)).type_as(out)
    
    # First decoder input is always 0
    # dec_in: (BATCH, 1, 1)
    # dec_in = torch.zeros(out.size(0), 1, 1).type_as(out)
    # dec_in = encoder_out[:, 0, :].unsqueeze(1)
    dec_in = encoder_cls.unsqueeze(1)

    target_len = out.size(1)

    if self.beam_size is not None:
        prev_beam = Beam(self.beam_size)
        prev_beam.candidates = [[]]
        prev_beam.scores = [0]
        f_done = (lambda x: len(x) == target_len)
        hyp_list = []
        valid_size = self.beam_size
    
    for t in range(target_len):
      hs, att_w = self.decoder(dec_in, hs, out)
      predictions = F.softmax(att_w, dim=1).argmax(1)

      # Pick next index
      # If teacher force the next element will we the ground truth
      # otherwise will be the predicted value at current timestep
      # teacher_force = random.random() < teacher_force_ratio
      # idx = y[:, t] if teacher_force else predictions
      # dec_in = torch.stack([x[b, idx[b].item()] for b in range(x.size(0))])
      # dec_in = dec_in.view(out.size(0), 1, 1).type(torch.float)
      dec_in = torch.stack([encoder_out[j, predictions[j], :]
                           for j in range(len(predictions))])
      dec_in = dec_in.unsqueeze(1)

      if self.beam_size is not None:
          pred_prob = F.softmax(att_w, dim=1).max(1)
          next_beam = Beam(valid_size)
          done_list, remain_list = next_beam.step(pred_prob, prev_beam, f_done)
          hyp_list.extend(done_list)
          valid_size -= len(done_list)

      # Add cross entropy loss (F.log_softmax + nll_loss)
      loss += F.cross_entropy(att_w, y[:, t])
      outputs[:, t] = predictions

    # Weight losses, so every element in the batch 
    # has the same 'importance' 
    batch_loss = loss / y.size(0)

    return outputs, batch_loss
