import torch
import torch.nn as nn

from collections import namedtuple
from typing import Optional, List
from based.models.gpt import GPTLMHeadModel
from .partial_runner import PartialRunner


class PartialRunnerForGPTLMHeadModel(nn.Module, PartialRunner):
    def __init__(self, model: GPTLMHeadModel):
        super().__init__()
        self.model = model
        self.L = len(self.model.transformer.layers)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        ##############################
        position_ids=None,
        inference_params=None,
        attention_mask=None,
        stream=None,
    ):
        raise NotImplementedError

    #     """
    #     "position_ids" is just to be compatible with Transformer generation. We don't use it.
    #     num_last_tokens: if > 0, only return the logits for the last n tokens
    #     """

    #     self.validate_args(input_ids, hidden_states, layer_indices)

    #     # NOTE simulates running on just layer_indices of self.model.transformer
    #     # using either input_ids or hidden_states as input, depending on whether
    #     # layer_indices starts with the first module or not,
    #     # and returning a hidden state or an output, depending on whether
    #     # layer_indices ends with the last module or not

    #     # if input_ids is not None:
    #     #     hidden_states = self.model.backbone.embedding(input_ids)

    #     if layer_indices is None:
    #         layer_indices = range(len(self.model.backbone.layers))

    #     # for layer in self.model.backbone.layers[layer_indices]:
    #     for layer_i in layer_indices:
    #         layer = self.model.backbone.layers[layer_i]
    #         hidden_states, residual = layer(
    #             hidden_states, None, inference_params=inference_params
    #         )
    #         hidden_states = hidden_states + residual

    #     return hidden_states
