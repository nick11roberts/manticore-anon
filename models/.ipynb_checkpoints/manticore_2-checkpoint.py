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
from models.partial_gpt_neox import PartialRunnerForGPTNeoXForCausalLM

# from models.partial_based import PartialRunnerForGPTLMHeadModel
from models.partial_runner import PartialRunner
from typing import List
from collections import namedtuple
from functools import partial
import math


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
# Also used for mamba
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


class GatedMixedOp(nn.Module):
    def __init__(
        self, k, fixed=False, alpha_mamba=None, retraining=False, 
        dash=False, gaea=False
    ):
        super().__init__()
        self.k = k
        self.alpha_mamba = alpha_mamba
        self.retraining = retraining
        self.dash = dash
        self.gaea = gaea
        assert not (dash and gaea), "Cannot specify both a GAEA and DASH style update"
        if alpha_mamba is not None:
            self.alphas = nn.Parameter(
                torch.Tensor([alpha_mamba, 1 - alpha_mamba]), requires_grad=False
            )
        else:
            if self.gaea:
                ## if we're using gaea, then we maintain the simplex values directly in the alphas, not the logits
                self.alphas = nn.Parameter(torch.ones(self.k) / self.k, requires_grad=not fixed)
            else:
                self.alphas = nn.Parameter(torch.zeros(self.k), requires_grad=not fixed)

    def forward(self, Wxs: List[torch.Tensor], xs: List[torch.Tensor]):
        assert len(Wxs) == self.k

        if self.alpha_mamba is not None:
            self.alphas.requires_grad = False

        alphas = self.get_alphas()

        return sum(
            [
                alphas[i] * (((1 - alphas[i]) * Wxs[i]) + (alphas[i] * xs[i]))
                for i in range(self.k)
            ]
        )

    def get_alphas(self):
        if self.alpha_mamba is not None or self.gaea: ## if we're using gaea, then we maintain the simplex values directly in the alphas, not the logits
            return self.alphas
        else:
            if self.dash:  # and not self.retraining:
                return F.gumbel_softmax(
                    self.alphas, dim=0, tau=1, hard=False, eps=1e-10
                )
            else:
                return F.softmax(self.alphas, dim=0)


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
        alpha_mamba=None,
        alpha_mamba2=None,
        projector_type: str = "default",
        projector_gpt: bool = False,
        embedding_source: str = "gptneo",
        vocab_size: int = 63599,
        retraining=False,
        dash=False,
        gaea=False,
        **kwargs,
    ):
        """
        n_mixtures: how many segments each model gets chopped into
        trunc_layers: truncate each model to at most this many layers
            - Can use in conjunction with n_mixtures; if trunc_layers==n_mixtures, each GatedMixedOp is 1 layer of each model
            - Above use-case mainly for pretraining?
        """
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.n_mixtures = int(n_mixtures)
        self.fixed_alphas = bool(fixed_alphas)
        self.use_projectors = bool(use_projectors)
        self.identity_init_projectors = bool(identity_init_projectors)
        self.trunc_layers = int(trunc_layers)
        self.problem_type = str(problem_type)
        self.alpha_mamba = alpha_mamba
        self.alpha_mamba2 = alpha_mamba2
        self.projector_type = str(projector_type)
        self.projector_gpt = bool(projector_gpt)
        self.embedding_source = str(embedding_source)
        self.vocab_size = int(vocab_size)
        self.retraining = bool(retraining)
        self.dash = bool(dash)
        self.gaea = bool(gaea)


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


def partial_class(m):
    if m.__class__.__name__ == "MambaLMHeadModel":
        return PartialRunnerForMambaLMHeadModel(m)
    elif m.__class__.__name__ == "GPTNeoForCausalLM":
        return PartialRunnerForGPTNeoForCausalLM(m)
    elif m.__class__.__name__ == "GPTNeoXForCausalLM":
        return PartialRunnerForGPTNeoXForCausalLM(m)
    elif "PartialRunner" in m.__class__.__name__:
        return m
    else:
        raise NotImplementedError


class ManticoreForTask(PreTrainedModel, GenerationMixin):
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
        models: List[PreTrainedModel],
        pretrain=False,
    ):
        """
        pretrain: Whether or not we are going to pretrain the model from scratch; False for finetuning with projectors
        config.trunc_layers: if >0,
        """
        super().__init__(config)
        self.config = config

        # Init the partial models $
        self.models = nn.ModuleList([partial_class(m) for m in models])

        # Model dims $
        self.ds = [m.d_model for m in self.models]  # TODO implement for this
        self.config.d_model = max(self.ds)

        # Create layer index splits $
        # NOTE this can throw an error if the lengths aren't divisible by n_mixtures
        self.layer_inds_all = [
            np.split(np.arange(m.L), self.config.n_mixtures) for m in self.models
        ]

        # NAS component $
        alpha_mambas = [self.config.alpha_mamba, self.config.alpha_mamba2]
        if self.config.alpha_mamba is not None and self.config.alpha_mamba2 is None:
            assert self.config.n_mixtures == 1
            assert len(self.models) == 2
        elif (
            self.config.alpha_mamba is not None and self.config.alpha_mamba2 is not None
        ):
            assert self.config.n_mixtures == 2
            assert len(self.models) == 2
        self.mixtures = nn.ModuleList(
            [
                GatedMixedOp(
                    k=len(self.models),
                    alpha_mamba=None, #alpha_mambas[m],
                    retraining=self.config.retraining,
                    dash=self.config.dash,
                    gaea=self.config.gaea,
                )
                for m in range(self.config.n_mixtures)
            ]
        )

        if self.config.fixed_alphas:
            disable_grad(self.mixtures)

        # Projectors $
        self.model_ins_all = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        (
                            projector_types[self.config.projector_type](
                                self.config.d_model, self.ds[i]
                            )
                            if self.config.use_projectors
                            else nn.Identity()
                        )
                        for _ in range(self.config.n_mixtures)
                    ]
                )
                for i in range(len(models))
            ]
        )
        self.model_outs_all = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        (
                            projector_types[self.config.projector_type](
                                self.ds[i], self.config.d_model
                            )
                            if self.config.use_projectors
                            else nn.Identity()
                        )
                        for _ in range(self.config.n_mixtures)
                    ]
                )
                for i in range(len(models))
            ]
        )

        # Embeddings $
        self.embeddings_all = nn.ModuleList(
            [
                nn.Embedding(self.config.vocab_size, self.ds[i])
                for i in range(len(models))
            ]
        )
        for i in range(len(self.embeddings_all)):
            nn.init.normal_(self.embeddings_all[i].weight, std=0.02)

    def disable_alphas(self):
        disable_grad(self.mixtures)

    def enable_alphas(self):
        enable_grad(self.mixtures)

    def disable_finetuning(self):
        for model in self.models:
            disable_grad(model)

    def enable_finetuning(self):
        for model in self.models:
            enable_grad(model)

    def disable_embeddings(self):
        disable_grad(self.embeddings_all)

    def enable_embeddings(self):
        enable_grad(self.embeddings_all)

    def disable_projectors(self):
        disable_grad(self.model_ins_all)
        disable_grad(self.model_outs_all)

    def enable_projectors(self):
        enable_grad(self.model_ins_all)
        enable_grad(self.model_outs_all)

    def disable_pretraining(self):
        self.disable_projectors()
        self.disable_finetuning()

    def enable_pretraining(self):
        self.enable_projectors()
        self.enable_finetuning()

    def load_projectors(self, projector_checkpoint, compat=False):
        if compat:
            from models.manticore_compat import (
                ManticoreForCausalLM as ManticoreForCausalLM_compat,
            )

            model_proj = ManticoreForCausalLM_compat.from_pretrained(
                projector_checkpoint,
                mamba=self.models[0].model,
                gptneo=self.models[1].model,
                token="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",
            )
            self.model_ins_all[0] = model_proj.mamba_ins
            self.model_ins_all[1] = model_proj.gptneo_ins

            self.model_outs_all[0] = model_proj.mamba_outs
            self.model_outs_all[1] = model_proj.gptneo_outs

            # self.mamba_outs = model_proj.mamba_outs
            print(f"Projectors loaded from {projector_checkpoint}")
        else:
            model_proj = ManticoreForCausalLM.from_pretrained(
                projector_checkpoint,
                models=self.models,
                token="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",
            )
            self.model_ins_all = model_proj.model_ins_all
            self.model_outs_all = model_proj.model_outs_all
            print(f"Projectors loaded from {projector_checkpoint}")

    def load_alphas(self, supernet):
        self.mixtures = supernet.mixtures
        self.disable_alphas()
        print(f"Alphas loaded from supernet")

    def pad_to_max(self, a):
        return F.pad(a, (0, self.config.d_model - a.shape[-1]), "constant", 0)

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
        ############### Embedding layer #############
        past_length = 0
        input_shape = input_ids.size()

        position_ids = torch.arange(
            past_length,
            input_shape[1]
            + past_length,  # (B,L) if proper input ids, (B,L,D) if actually embeddings
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0)

        mixed_embeds = []
        for m in range(len(self.models)):
            inputs_embeds = self.embeddings_all[m](input_ids)
            inputs_embeds = self.pad_to_max(inputs_embeds)
            mixed_embeds.append(self.mixtures[0].get_alphas()[m] * inputs_embeds)

        hidden_states = sum(mixed_embeds)

        for i in range(self.config.n_mixtures):
            ############### One mixture level ###############
            _hidden_states_all = []
            hidden_states_all = []
            for m in range(len(self.models)):
                hidden_states_m = self.model_ins_all[m][i](hidden_states)
                hidden_states = self.pad_to_max(hidden_states)
                hidden_states_m = self.pad_to_max(hidden_states_m)
                hidden_states_m = (
                    (1 - self.mixtures[i].get_alphas()[m]) * hidden_states_m
                ) + (self.mixtures[i].get_alphas()[m] * hidden_states)
                hidden_states_m = hidden_states_m[..., : self.ds[m]]

                _hidden_states_m = self.models[m].forward_intermediate(
                    input_ids=None,
                    hidden_states=hidden_states_m,
                    layer_indices=self.layer_inds_all[m][i],
                    position_ids=position_ids,
                    inference_params=inference_params,
                    num_last_tokens=num_last_tokens,
                )
                hidden_states_m = self.model_outs_all[m][i](_hidden_states_m)

                _hidden_states_m = self.pad_to_max(_hidden_states_m)
                hidden_states_m = self.pad_to_max(hidden_states_m)

                _hidden_states_all.append(_hidden_states_m)
                hidden_states_all.append(hidden_states_m)

            hidden_states = self.mixtures[i](hidden_states_all, _hidden_states_all)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        return hidden_states


class ManticoreForSequenceClassification(ManticoreForTask):
    def __init__(
        self,
        config: ManticoreConfig,
        num_classes: int,
        mamba: MambaLMHeadModel,
        gptneo: GPTNeoForCausalLM,
        # based: GPTLMHeadModel,
        pretrain=False,
    ):
        super().__init__(config=config, mamba=mamba, gptneo=gptneo, pretrain=pretrain)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.config.d_model if self.config.use_projectors else self.d_gptneo,
            self.num_classes,
            bias=False,  # ?
        )

    def forward(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        # labels=None,
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
        batch_size, sequence_length = input_ids.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = (
                torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            )
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        # TODO Zhiqi -- add problem type into the config
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_classes == 1:
                    self.config.problem_type = "regression"
                elif self.num_classes > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_classes == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_classes), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
        )


class ManticoreForTokenClassification(ManticoreForTask):
    def __init__(
        self,
        config: ManticoreConfig,
        num_classes: int,
        mamba: MambaLMHeadModel,
        gptneo: GPTNeoForCausalLM,
        # based: GPTLMHeadModel,
        pretrain=False,
    ):
        super().__init__(config=config, mamba=mamba, gptneo=gptneo, pretrain=pretrain)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.config.d_model if self.config.use_projectors else self.d_gptneo,
            self.num_classes,
            bias=False,  # ?
        )

    def forward(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        # labels=None,
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


class ManticoreForLEGO(ManticoreForTokenClassification):
    def __init__(
        self,
        config: ManticoreConfig,
        mamba: MambaLMHeadModel,
        gptneo: GPTNeoForCausalLM,
        # based: GPTLMHeadModel,
        pretrain=False,
    ):
        super().__init__(
            config=config,
            num_classes=1,
            mamba=mamba,
            gptneo=gptneo,
            # based=based,
            pretrain=pretrain,
        )

    def forward(
        self,
        input_ids,
        labels,
        order,
        n_var=12,
        var_pred=6,
        append_var_tokens=True,  # LEGO parameters
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        # labels=None,
        **kwargs,
    ):
        pred = self.forward_backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            labels=labels,
            **kwargs,
        )

        inv_order = order.permute(0, 2, 1).to(input_ids.device)
        pred = self.classifier(pred)  # unordered
        """ASSUMES GPT TOKENIZER; NO EXTRA STARTING NOR ENDING TOKEN"""
        vars_slice = (
            np.s_[:, 0::5, :] if not append_var_tokens else np.s_[:, -n_var:, :]
        )  # ASSUMES GPT TOKENIZER; NO EXTRA STARTING NOR ENDING TOKEN

        pred = torch.bmm(inv_order, pred[vars_slice]).squeeze(-1)

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = None
        if labels is not None:
            loss = (
                sum(
                    [
                        criterion(pred[:, idx], labels[:, idx].float())
                        for idx in range(var_pred)
                    ]
                )
                / var_pred
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=pred,
        )


class ManticoreForICLRegression(ManticoreForTokenClassification):

    def __init__(
        self,
        config: ManticoreConfig,
        n_regr_dims: int,
        train_metric_fn,
        mamba: MambaLMHeadModel,
        gptneo: GPTNeoForCausalLM,
        # based: GPTLMHeadModel,
        pretrain=False,
    ):
        super().__init__(
            config=config,
            num_classes=1,
            mamba=mamba,
            gptneo=gptneo,
            # based=based,
            pretrain=pretrain,
        )
        assert n_regr_dims is not None and train_metric_fn is not None
        self.iclregr_n_dim = n_regr_dims
        # self.iclregr_task_name = task_name
        self.iclregr_train_metric = train_metric_fn

        self._read_in = nn.Linear(
            n_regr_dims,
            # self.config.d_model if self.config.use_projectors else self.d_gptneo,
            self.d_gptneo,  # read_in -> embedder (identity) + pos_embed ->
        )
        """
        float numerical inputs, not tokens; hacky...
        If the way ManticoreForTask (base class) handles embedding forward changes, THEN THIS WILL HAVE TO CHANGE TOO
            Basically we just need some way to ignore the embedder that's used in ManticoreForTask $$
        """
        self.mamba.model.backbone.embedding = (
            nn.Identity()
        )  # float numerical inputs, not tokens; hacky...
        self.gptneo.model.transformer.wte = nn.Identity()

        self._read_out = nn.Linear(
            self.config.d_model if self.config.use_projectors else self.d_gptneo,
            1,
        )

    def forward(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        # labels=None,
        **kwargs,
    ):
        pred = self._read_in(input_ids)
        pred = self.forward_backbone(
            input_ids=pred,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            labels=labels,
            **kwargs,
        )  # (B, L, D)
        pred = pred[:, -1, :]  # last token on x query; (B,D)
        pred = self._read_out(pred)  # (B,1)
        pred = pred.squeeze(-1)  # (B,)

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


class ManticoreForCausalLM(ManticoreForTask):
    def __init__(
        self,
        config: ManticoreConfig,
        models: List[PreTrainedModel],
        # based: GPTLMHeadModel,
        pretrain=False,
    ):
        super().__init__(config=config, models=models, pretrain=pretrain)

        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(self.ds[m], self.config.vocab_size, bias=False)
                for m in range(len(self.models))
            ]
        )

        self.tie_weights()

    def tie_weights(self):
        for m in range(len(self.models)):
            self.lm_heads[m].weight = self.embeddings_all[m].weight

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

        lm_logits_all = []
        for m in range(len(self.models)):
            lm_logits_m = self.lm_heads[m](hidden_states[..., : self.ds[m]])
            lm_logits_all.append(self.mixtures[0].get_alphas()[m] * lm_logits_m)
        lm_logits = sum(lm_logits_all)

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
