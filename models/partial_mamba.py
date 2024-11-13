import torch
import torch.nn as nn

from collections import namedtuple
from typing import Optional, List
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from .partial_runner import PartialRunner


class PartialRunnerForMambaLMHeadModel(nn.Module, PartialRunner):
    def __init__(self, model: MambaLMHeadModel):
        super().__init__()
        self.model = model
        self.L = len(self.model.backbone.layers)
        self.d_model = self.model.config.d_model

    def forward_intermediate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        ##############################
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if layer_indices is None:
            layer_indices = range(self.L)
        self.validate_args(input_ids, hidden_states, layer_indices)

        # NOTE simulates running on just layer_indices of self.model.transformer
        # using either input_ids or hidden_states as input, depending on whether
        # layer_indices starts with the first module or not,
        # and returning a hidden state or an output, depending on whether
        # layer_indices ends with the last module or not

        # if input_ids is not None:
        #     hidden_states = self.model.backbone.embedding(input_ids)

        # for layer in self.model.backbone.layers[layer_indices]:
        for layer_i in layer_indices:
            layer = self.model.backbone.layers[layer_i]
            hidden_states, residual = layer(
                hidden_states, None, inference_params=inference_params
            )
            hidden_states = hidden_states + residual

        # If we use the last layer, apply the final layer norm
        # TODO not sure yet where to put this
        if layer_indices[-1] == self.L - 1:
            hidden_states = self.model.backbone.norm_f(
                hidden_states.to(dtype=self.model.backbone.norm_f.weight.dtype)
            )

        return hidden_states
