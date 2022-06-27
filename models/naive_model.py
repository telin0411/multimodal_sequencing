import os
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
)
from transformers.file_utils import is_sklearn_available, requires_sklearn

logger = logging.getLogger(__name__)


class NaiveMultimodalModel(nn.Module):
    
    def __init__(self, args, language_model=None, vision_model=None):
        super(NaiveMultimodalModel, self).__init__()

        self.args = args
        self.language_model = language_model
        self.vision_model = vision_model
        if self.vision_model is not None:
            self.vision_model = vision_model.to(args.device)

            # Remove the final FC layer.
            if True:
                num_img_dim = self.vision_model.fc.in_features
                self.vision_model.fc = nn.Identity()

            img_project_layer = nn.Sequential(
                nn.Linear(num_img_dim, self.language_model.config.hidden_size),
                nn.ReLU(inplace=True),
            )
            self.img_project_layer = img_project_layer.to(args.device)

    def forward(self, inputs):
        """
            inputs (dict): Dict of inputs.
        """
        if self.args.multimodal_img_part:
            inputs["input_ids"] = torch.zeros(inputs["input_ids"].size(0),
                1).type_as(inputs["input_ids"])
            inputs["attention_mask"] = torch.zeros(inputs["input_ids"].size(0),
                1).type_as(inputs["attention_mask"])

        # Transform the input ids to input embeddings.
        if self.vision_model is not None:
            if self.args.model_type == "roberta":
                embedding_layer = self.language_model.roberta.embeddings
                inputs_embeds = embedding_layer(inputs["input_ids"])
            else:
                raise NotImplementedError("Not handling {} yet!".format(self.args.model_type))
            
            del inputs["input_ids"]
            inputs["inputs_embeds"] = inputs_embeds

        bz, token_len = inputs["attention_mask"].size()

        if "images" in inputs:
            bz, img_len, C, H, W = inputs["images"].size()

            img_attention_mask = torch.ones(bz, img_len).long()
            img_attention_mask = img_attention_mask.to(self.args.device)

            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"], img_attention_mask
            ], dim=-1)

        # TODO Deal with token type ids for multimodal inputs.
        if inputs["token_type_ids"] is not None:
            pass

        # Deal with the images.
        if "images" in inputs:
            # To B x C x H x W
            img_seq = torch.unbind(inputs["images"], dim=1)
            img_embeds = []
            for i in range(len(img_seq)):
                img_curr = img_seq[i].float()  # TODO Verify this.
                img_curr_embed = self.vision_model(img_curr)
                img_curr_embed = self.img_project_layer(img_curr_embed)
                img_embeds.append(img_curr_embed)
            img_embeds = torch.stack(img_embeds, dim=1)
            inputs["inputs_embeds"] = torch.cat([
                inputs["inputs_embeds"], img_embeds
            ], dim=1)

            del inputs["images"]

        if "pure_decode" in self.args.task_type:
            outputs = self.language_model(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                decoder_input_ids=inputs["labels"],
                labels=inputs["labels"],
                return_dict=True,
            )
        else:
            outputs = self.language_model(**inputs)

        return outputs


    def save_pretrained(self, save_directory):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.language_model.config.architectures = [model_to_save.__class__.__name__]

        state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        self.keys_to_never_save = None
        if self.keys_to_never_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self.keys_to_never_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        if getattr(self.language_model.config, "xla_device", False) and is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            if xm.is_master_ordinal():
                # Save configuration file
                model_to_save.config.save_pretrained(save_directory)
            # xm.save takes care of saving only from master
            xm.save(state_dict, output_model_file)
        else:
            model_to_save.language_model.config.save_pretrained(save_directory)
            torch.save(state_dict, output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))
