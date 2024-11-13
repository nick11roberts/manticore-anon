import os
import torch
import torch.nn as nn
import json
import os
import torch
import torch.nn as nn

from functools import partial
from collections import namedtuple
from transformers.modeling_outputs import (
    CausalLMOutput,
    TokenClassifierOutput,
    SequenceClassifierOutputWithPast,
)
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer

class MadWrapper(nn.Module):
    def __init__(self, mad_model):
        super().__init__()
        self.mad_model = mad_model

    def forward(
        self,
        input_ids, 
        labels=None,
    ):

        logits = self.mad_model(input_ids)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.mad_model.vocab_size), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )
        