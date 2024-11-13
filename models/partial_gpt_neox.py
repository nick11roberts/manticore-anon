import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXModel, GPTNeoXForCausalLM
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPast,
    CausalLMOutput,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from typing import Optional, Tuple, List, Union
from .partial_runner import PartialRunner


class PartialRunnerForGPTNeoXForCausalLM(nn.Module, PartialRunner):
    def __init__(self, model: GPTNeoXForCausalLM):
        super().__init__()
        self.model = model
        self.L = len(self.model.gpt_neox.layers)
        self.d_model = self.model.gpt_neox.config.hidden_size

    def forward_intermediate_tranformer(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        ####
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.gpt_neox.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.gpt_neox.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.gpt_neox.config.use_return_dict
        )
        use_cache = (
            use_cache if use_cache is not None else self.model.gpt_neox.config.use_cache
        )

        # NOTE
        if input_ids is not None:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time"
                )
            elif input_ids is not None:
                self.model.gpt_neox.warn_if_padding_and_no_attention_mask(
                    input_ids, attention_mask
                )
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds"
                )
        else:
            input_shape = hidden_states.size()[:2]
            device = hidden_states.device

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple(
                [None] * self.model.gpt_neox.config.num_hidden_layers
            )
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else hidden_states.device
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            if self.model.gpt_neox._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(
                    dtype=self.model.gpt_neox.dtype
                )  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(
                    self.model.gpt_neox.dtype
                ).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.model.gpt_neox.get_head_mask(
            head_mask, self.model.gpt_neox.config.num_hidden_layers
        )

        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_in(input_ids)

        if layer_indices[0] == 0:
            hidden_states = self.model.gpt_neox.emb_dropout(hidden_states)

        if self.model.gpt_neox.gradient_checkpointing and self.model.gpt_neox.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        # for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
        for i in layer_indices:
            layer, layer_past = self.model.gpt_neox.layers[i], past_key_values[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if (
                self.model.gpt_neox.gradient_checkpointing
                and self.model.gpt_neox.training
            ):
                outputs = self.model.gpt_neox._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    None,
                    output_attentions,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        if layer_indices[-1] == len(self.model.gpt_neox.layers) - 1:
            hidden_states = self.model.gpt_neox.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def forward_intermediate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        #####
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        # NOTE simulates running on just layer_indices of self.model.transformer
        # using either input_ids or hidden_states as input, depending on whether
        # layer_indices starts with the first module or not,
        # and returning a hidden state or an output, depending on whether
        # layer_indices ends with the last module or not

        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if layer_indices is None:
            layer_indices = range(self.L)
        self.validate_args(input_ids, hidden_states, layer_indices)

        outputs = self.forward_intermediate_tranformer(
            input_ids=input_ids,
            hidden_states=hidden_states,
            layer_indices=layer_indices,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        return hidden_states
