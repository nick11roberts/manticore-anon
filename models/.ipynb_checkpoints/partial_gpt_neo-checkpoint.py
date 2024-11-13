import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers.models.gpt_neo import GPTNeoModel, GPTNeoForCausalLM
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPast,
    CausalLMOutput,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from typing import Optional, Tuple, List, Union
from .partial_runner import PartialRunner


class PartialRunnerForGPTNeoForCausalLM(nn.Module, PartialRunner):
    def __init__(self, model: GPTNeoForCausalLM):
        super().__init__()
        self.model = model
        self.L = len(self.model.transformer.h)

    def forward_intermediate_tranformer(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        ####
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.transformer.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.transformer.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else self.model.transformer.config.use_cache
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.transformer.config.use_return_dict
        )

        # NOTE
        if input_ids is not None:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time"
                )
            elif input_ids is not None:
                self.model.transformer.warn_if_padding_and_no_attention_mask(
                    input_ids, attention_mask
                )
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds"
                )

            device = input_ids.device if input_ids is not None else inputs_embeds.device
        else:
            input_shape = hidden_states.size()[:2]
            device = hidden_states.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.model.transformer.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.model.transformer.get_head_mask(
            head_mask, self.model.transformer.config.num_layers
        )

        # NOTE
        if input_ids is not None:
            if inputs_embeds is None:
                inputs_embeds = self.model.transformer.wte(input_ids)
            position_embeds = self.model.transformer.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, hidden_states, past_length
        )

        if token_type_ids is not None:
            token_type_embeds = self.model.transformer.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.model.transformer.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if (
            self.model.transformer.gradient_checkpointing
            and self.model.transformer.training
        ):
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        # for i, (block, layer_past) in enumerate(
        #     zip(self.model.transformer.h, past_key_values)
        # ):
        for i in layer_indices:
            block, layer_past = self.model.transformer.h[i], past_key_values[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if (
                self.model.transformer.gradient_checkpointing
                and self.model.transformer.training
            ):
                outputs = self.model.transformer._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        # hidden_states = self.model.transformer.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def forward_intermediate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        #####
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        transformer_outputs = self.forward_intermediate_tranformer(
            input_ids=input_ids,
            hidden_states=hidden_states,
            layer_indices=layer_indices,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        hidden_states = transformer_outputs[0]
        # if layer_indices[-1] != len(self.model.transformer.h) - 1:
        return hidden_states

        # lm_logits = self.model.lm_head(hidden_states)

        # loss = None
        # if labels is not None:
        #     # move labels to correct device to enable model parallelism
        #     labels = labels.to(lm_logits.device)
        #     # Compute loss in fp32 to match with mesh-tf version
        #     # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        #     lm_logits = lm_logits.to(torch.float32)

        #     # Shift so that tokens < n predict n
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(
        #         shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        #     )

        #     lm_logits = lm_logits.to(hidden_states.dtype)
        #     loss = loss.to(hidden_states.dtype)

        # if not return_dict:
        #     output = (lm_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        # return CausalLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=transformer_outputs.past_key_values,
        #     hidden_states=transformer_outputs.hidden_states,
        #     attentions=transformer_outputs.attentions,
        # )
