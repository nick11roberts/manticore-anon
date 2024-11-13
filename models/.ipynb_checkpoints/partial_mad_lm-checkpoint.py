import torch
import torch.nn as nn

from collections import namedtuple
from typing import Optional, List
from mad.model import LanguageModel as MadLM
from .partial_runner import PartialRunner


class PartialRunnerForMadLM(nn.Module, PartialRunner):
    def __init__(self, model: MadLM):
        super().__init__()
        self.model = model
        self.L = len(self.model.model) # is a module list of the backbone
        self.model.unembed = self.model.unembed[:1] # delete the final linear head; we'll add a single one in in the manticore wrapper if needed
        
    def forward_intermediate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        ##############################
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if layer_indices is None:
            layer_indices = range(self.L)
        self.validate_args(input_ids, hidden_states, layer_indices)
        assert hidden_states is not None
        # NOTE simulates running on just layer_indices of self.model.transformer
        # using either input_ids or hidden_states as input, depending on whether
        # layer_indices starts with the first module or not,
        # and returning a hidden state or an output, depending on whether
        # layer_indices ends with the last module or not

        # if input_ids is not None:
        #     hidden_states = self.model.backbone.embedding(input_ids)

        # for layer in self.model.backbone.layers[layer_indices]:
        for layer_i in layer_indices:
            layer = self.model.model[layer_i] # each is nn.Sequential(norm, layer)
            hidden_states = layer(
                hidden_states
            ) + hidden_states # residuals needs to be manually specified

        # If we use the last layer, apply the final layer norm
        # TODO not sure yet where to put this
        if layer_indices[-1] == self.L - 1:
            hidden_states = self.model.unembed[0]( # model.unembed is nn.Sequential(norm, linear)
                hidden_states.to(dtype=self.model.unembed[0].weight.dtype) # is it necessary for MAD model?
            )

        return hidden_states
