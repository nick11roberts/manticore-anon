'''
Wrappers for each partial model class specialized for each task
Similar to those in Manticore
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from transformers.models.gpt_neo import GPTNeoForCausalLM
from transformers.modeling_outputs import CausalLMOutput, TokenClassifierOutput
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.utils.generation import GenerationMixin

# from based.models.gpt import GPTLMHeadModel
from models.partial_mamba import PartialRunnerForMambaLMHeadModel
from models.partial_gpt_neo import PartialRunnerForGPTNeoForCausalLM
# from models.partial_based import PartialRunnerForGPTLMHeadModel

from typing import List
from collections import namedtuple
from functools import partial

class PartialWrapperForICLRegression(nn.Module):
    def __init__(self, 
                 partial_runner,
                 n_regr_dims: int,
                 train_metric_fn,
                ):
        super().__init__()
        self.partial_runner = partial_runner
        if isinstance(self.partial_runner, PartialRunnerForMambaLMHeadModel):
            self.model_type = "mamba"
            self.d_model = partial_runner.model.config.d_model
        elif isinstance(self.partial_runner, PartialRunnerForGPTNeoForCausalLM):
            self.model_type = "gptneo"
            self.d_model = partial_runner.model.transformer.embed_dim
        # TODO: based
        else:
            raise NotImplementedError
        
        self.iclregr_dims = n_regr_dims
        self.iclregr_train_metric = train_metric_fn

        self._read_in = nn.Linear(
            n_regr_dims,
            self.d_model,
        )

        self._read_out = nn.Linear(
            self.d_model,
            1,
        )

    def forward(self, 
                input_ids, # are actually continoous inputs, not tokens
                labels=None,
                position_ids=None,
                **kwargs
               ):
        pred = self._read_in(input_ids)
        if self.model_type=="mamba":
            pred = self.partial_runner.forward_intermediate(
                hidden_states=pred,
                position_ids=position_ids,
                **kwargs,
            )
        elif self.model_type=="gptneo":
            past_length = 0
            input_shape = pred.size()
        
            position_ids = torch.arange(
                past_length,
                input_shape[1] + past_length,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0)
 
            position_embeds = (
                self.partial_runner.model.transformer.wpe(position_ids)
            ) 
            pred = pred + position_embeds
            pred = self.partial_runner.forward_intermediate(
                hidden_states = pred,
                position_ids=position_ids,
                **kwargs,
            )
        # TODO: based
        else:
            raise NotImplementedError
        pred = self._read_out(pred)
        pred = pred[:, -1, 0] # last token on x_query

        loss = None
        if labels is not None:
            loss_fct = self.iclregr_train_metric
            loss = loss_fct(pred, labels)

        # TODO: most of these are regression not classification, is there a better output class to wrap in?
        return TokenClassifierOutput(
            loss=loss,
            logits=pred,
        )
    
    @staticmethod
    def _combine(
        xs_b, ys_b
    ):  # not necessary if input data is already combined-formatted
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs