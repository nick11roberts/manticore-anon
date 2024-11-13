from mad.configs import MADConfig, MADModelConfig
from mad.model.layers import (
    Attention,
    LinearAttention,
    GatedLinearAttention,
    HyenaOperator, MultiHeadHyenaOperator, HyenaExpertsOperator,
    Mamba,
    Mlp, SwiGLU, MoeMlp,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from transformers.models.gpt_neo import GPTNeoForCausalLM
from transformers.modeling_outputs import (
    CausalLMOutput,
    TokenClassifierOutput,
    SequenceClassifierOutputWithPast,
)
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.utils.generation import GenerationMixin

# from based.models.gpt import GPTLMHeadModel
from models.partial_mamba import PartialRunnerForMambaLMHeadModel
from models.partial_gpt_neo import PartialRunnerForGPTNeoForCausalLM
from models.partial_mad_lm import PartialRunnerForMadLM

# from models.partial_based import PartialRunnerForGPTLMHeadModel
from typing import List
from collections import namedtuple
from functools import partial
import math

# TODO: check, is this necessary correct for all our model classes?
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def disable_grad(m):
    for i, param in enumerate(m.parameters()):
        param.requires_grad = False


def enable_grad(m):
    for i, param in enumerate(m.parameters()):
        param.requires_grad = True


class MixedOp(nn.Module):
    def __init__(self, k, fixed=False, alpha_mamba=None):
        super().__init__()
        self.k = k
        self.alpha_mamba = alpha_mamba
        if alpha_mamba is not None:
            self.alphas = nn.Parameter(
                torch.Tensor([alpha_mamba, 1 - alpha_mamba]), requires_grad=False
            )
        else:
            self.alphas = nn.Parameter(torch.zeros(self.k), requires_grad=not fixed)

    def forward(self, xs: List[torch.Tensor]):
        assert len(xs) == self.k
        x = torch.stack(xs, -1)
        if self.alpha_mamba is not None:
            self.alphas.requires_grad = False
            # print(self.alphas)
            # quit()
            return x @ self.alphas
        else:
            return x @ F.softmax(self.alphas, dim=0)
        # return xs[0]  # TODO remove
        # return x @ F.gumbel_softmax(self.alphas, dim=0, tau=1.0, hard=True)

class ManticoreConfig(PretrainedConfig):
    model_type = "manticore"

    def __init__(
        self,
        d_model: int = 768,
        n_mixtures: int = 4,
        fixed_alphas: bool = False,
        use_projectors: bool = True,
        identity_init_projectors: bool = True,
        trunc_layers: int = 0,
        problem_type: str = "single_label_classification",
        alpha_mamba: float = None,
        projector_type: str = "default",
        # projector_gpt: bool = False,
        **kwargs,
    ):
        """
        n_mixtures: how many segments each model gets chopped into
        trunc_layers: truncate each model to at most this many layers
            - Can use in conjunction with n_mixtures; if trunc_layers==n_mixtures, each MixedOp is 1 layer of each model
            - Above use-case mainly for pretraining?
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_mixtures = n_mixtures
        self.fixed_alphas = fixed_alphas
        self.use_projectors = use_projectors
        self.identity_init_projectors = identity_init_projectors
        self.trunc_layers = trunc_layers
        self.problem_type = problem_type
        # self.alpha_mamba = alpha_mamba
        self.projector_type = projector_type
        # self.projector_gpt = projector_gpt

class LinearProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        nn.init.eye_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


class LinearSkipProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(x) + x


class TwoLayerProjector(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=None):
        super().__init__()
        if d_hidden == None:
            d_hidden = d_in
        self.proj1 = nn.Linear(d_in, d_hidden)
        self.a = nn.GELU()
        self.proj2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = self.proj1(x)
        x = self.a(x)
        x = self.proj2(x)
        return x


class TwoLayerSkipProjector(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=None):
        super().__init__()
        if d_hidden == None:
            d_hidden = d_in
        self.proj1 = nn.Linear(d_in, d_hidden)
        self.a = nn.GELU()
        self.proj2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        y = self.proj1(x)
        y = self.a(y)
        y = self.proj2(y)
        return x + y
projector_types = {
    "default": nn.Linear,
    "LinearProjector": LinearProjector,
    "LinearSkipProjector": LinearSkipProjector,
    "TwoLayerProjector": TwoLayerProjector,
    "TwoLayerSkipProjector": TwoLayerSkipProjector,
}

'''
Example code to load a MAD model:
```
from mad.configs import MADConfig, MADModelConfig
model_config = MADModelConfig()
model_config.update_from_kwargs(
    {
        "layers":['hyena', 'swiglu', 'hyena', 'swiglu'],
        "backbone": 'language-model',
        "dim": 128,
    }
)
model = model_config.build_model_from_registry().to("cuda")
```
'''
class ManticoreMadForTask(PreTrainedModel, GenerationMixin):
    config_class = ManticoreConfig
    """ Built on top of MambaLMHeadModel
    TODOs: 
    -Implement based in forward pass
    -Make mixture handling more flexible/generalizable?; 
        -currently places in code that need manual updating whenever a new model is added to the mixture are marked with $ in comments
    """

    def __init__(
        self,
        config: ManticoreConfig,
        vocab_size: int,
        mad_models: list,
        # based: GPTLMHeadModel,
        pretrain=False,
    ):
        """
        pretrain: Whether or not we are going to pretrain the model from scratch; False for finetuning with projectors
        config.trunc_layers: if >0,
        """
        super().__init__(config)
        self.config = config

        
        # Model dims $
        self.mad_dims = [m.dim for m in mad_models]
        self.vocab_size = vocab_size
        
        assert self.config.use_projectors or (np.all( np.array(self.mad_dims)==self.mad_dims[0] ))
        self.d_model = self.config.d_model if self.config.use_projectors else self.mad_dims[0]
        # self.d_model = self.config.d_model # TODO choose max

        # Init the partial models $
        self.mad_models = nn.ModuleList( [PartialRunnerForMadLM(m) for m in mad_models] )
        self.mad_models[0].model.reinit_embedding(dim=self.d_model, vocab_size=vocab_size) # reinit first embedding to d_model shape and use it as the embedding for the entire manticore model; delete other mad models' embeddings to save space
        for i in range(1, len(self.mad_models)):
            if hasattr(self.mad_models[i].model, "token_embeds"):
                del self.mad_models[i].model.token_embeds 
            if hasattr(self.mad_models[i].model, "position_embeds"):
                del self.mad_models[i].model.position_embeds 
        # Mamba
        # self.mamba = PartialRunnerForMambaLMHeadModel(mamba)
        # # GPT-Neo
        # self.gptneo = PartialRunnerForGPTNeoForCausalLM(gptneo)
        # Based
        # self.based = PartialRunnerForGPTLMHeadModel(based)

        if self.config.trunc_layers > 0:  # truncate the model layers if necessary $
            tl = self.config.trunc_layers
            for partial_model in self.mad_models:
                if tl > partial_model.L:
                    print(
                        "WARNING: Truncation is greater than full some model depth(s); \nThis can potentially mess up the intended interaction between truncation and n_mixtures and break the split indices, so check the original number of layers in each model before setting truncation"
                    )
                partial_model.L = min(tl, partial_model.L)

                # truncate the layers within each model; needs to handled on an individual basis
                partial_model.model = partial_model.model[:tl]
                # TODO when based implemented # self.based.model.... = self.based.model...[:tl]

        # Create layer index splits $
        # NOTE this can throw an error if the lengths aren't divisible by n_mixtures, and this always needs to come after any potential layer truncating
        self.mad_inds = [ np.split(np.arange(m.L), self.config.n_mixtures) for m in self.mad_models ]
        # self.gptneo_inds = np.split(np.arange(self.gptneo.L), self.config.n_mixtures)
        # self.based_inds = np.split(np.arange(self.based.L), self.config.n_mixtures)

        # NAS component
        # if self.config.alpha_mamba is not None:
        #     assert self.config.n_mixtures == 1
        self.mixtures = nn.ModuleList(
            [
                MixedOp(k= len(self.mad_models), alpha_mamba=None)
                for _ in range(self.config.n_mixtures)
            ]  # k changes depending on how many mixtures $
        )
        if self.config.fixed_alphas:
            disable_grad(self.mixtures)

        # Projectors $
        self.mad_ins = nn.ModuleList( # (n_mixtures, num_mad_models)
            [
                nn.ModuleList([
                    projector_types[self.config.projector_type](
                        self.config.d_model, partial_model.model.dim
                    )
                    if self.config.use_projectors
                    else nn.Identity()
                    for partial_model in self.mad_models
                ])
                for _ in range(self.config.n_mixtures)
            ]
        )
        self.mad_outs = nn.ModuleList( # (n_mixtures, num_mad_models)
            [
                nn.ModuleList([
                    projector_types[self.config.projector_type](
                        partial_model.model.dim, self.config.d_model
                    )
                    if self.config.use_projectors
                    else nn.Identity()
                    for partial_model in self.mad_models
                ])
                for _ in range(self.config.n_mixtures)
            ]
        )

    def disable_alphas(self):
        disable_grad(self.mixtures)

    def enable_alphas(self):
        enable_grad(self.mixtures)

    def disable_finetuning(self):
        disable_grad(self.mad_models)

    def enable_finetuning(self):
        enable_grad(self.mad_models)

    def disable_projectors(self):
        disable_grad(self.mad_ins)
        disable_grad(self.mad_outs)

    def enable_projectors(self):
        enable_grad(self.mad_ins)
        enable_grad(self.mad_outs)

    def disable_pretraining(self):
        self.disable_projectors()
        self.disable_finetuning()

    def enable_pretraining(self):
        self.enable_projectors()
        self.enable_finetuning()

    def load_projectors(self, projector_checkpoint):
        # model_proj = ManticoreForCausalLM.from_pretrained(
        #     projector_checkpoint,
        #     mamba=self.mamba.model,
        #     gptneo=self.gptneo.model,
        #     token="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",
        # )
        # self.mamba_ins = model_proj.mamba_ins
        # self.mamba_outs = model_proj.mamba_outs
        # print(f"Projectors loaded from {projector_checkpoint}")
        raise NotImplementedError

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.model.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward_backbone(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        labels=None,
        **kwargs,
    ):
        """
        Almost the complete forward pass, sans the LM/token class/whatever head
        Child classes will have their final layer & actual forward pass suited for the particular task
        """
        """TODO: implement based """
        ############### Use GPT-Neo's embedding layer #############
        # hidden_states = self.mamba.model.backbone.embedding(input_ids)
        assert (input_ids is not None)
        hidden_states = self.mad_models[0].model.embed(input_ids, position_ids) # first mad model's embedding
        ###########################################################

        for i in range(self.config.n_mixtures):
            ############### One mixture level ############### $
            # print(hidden_states)
            hidden_states_list = \
                [
                    self.mad_outs[i][j](
                        self.mad_models[j].forward_intermediate(
                            hidden_states = self.mad_ins[i][j](hidden_states),
                            position_ids = position_ids,
                            layer_indices = self.mad_inds[j][i],
                            inference_params=inference_params,
                            num_last_tokens=num_last_tokens,
                        )
                    )
                    for j in range(len(self.mad_models))
                ]
            # print(hidden_states_list)
            hidden_states = self.mixtures[i](hidden_states_list)
            #################################################

        # hidden_states = self.gptneo.model.transformer.ln_f(hidden_states)

        # print("hidden_states", hidden_states)
        # quit()
        # hidden_states = self.mamba.model.backbone.norm_f(
        #     hidden_states.to(dtype=self.mamba.model.backbone.norm_f.weight.dtype)
        # )
        # TODO need to add the norm function back in

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        return hidden_states

    def forward(**kwargs):
        raise NotImplementedError


class ManticoreMadForCausalLM(ManticoreMadForTask):
    def __init__(
        self,
        config: ManticoreConfig,
        vocab_size:int,
        mad_models: list,        
        pretrain=False,
    ):
        super().__init__(config=config, vocab_size=vocab_size, mad_models=mad_models, pretrain=pretrain)
        # self.vocab_size=vocab_size
        self.lm_head = nn.Linear(self.d_model, vocab_size)
    
    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        labels=None,
        **kwargs,
    ):
        hidden_states = self.forward_backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            labels=labels,
            **kwargs,
        )

        # NOTE TODO change this to GPT-Neo
        # lm_logits = self.mamba.model.lm_head(hidden_states)
        # lm_logits = self.gptneo.model.lm_head(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        ###################################################

        # Loss computation and HF output from GPT-Neo
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        # NOTE there's also a router version. Look into how this works?
        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
        )

class ManticoreMadForTokenClassification(ManticoreMadForTask):
    def __init__(
        self,
        config: ManticoreConfig,
        vocab_size:int,
        num_classes:int, 
        mad_models: list,        
        pretrain=False,
    ):
        super().__init__(config=config, vocab_size=vocab_size, mad_models=mad_models, pretrain=pretrain)
        self.vocab_size=vocab_size
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.d_model, num_classes)
    
    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        labels=None,
        **kwargs,
    ):
        hidden_states = self.forward_backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            labels=labels,
            **kwargs,
        )
        logits = self.classifier(hidden_states)
        ###################################################

        # Loss computation and HF output
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )







