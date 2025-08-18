# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_hymeta import HymetaConfig
from fla.modules import FusedCrossEntropyLoss, ShortConvolution
from fla.modules.activations import swish
from .activations import swiglu_linear
from .moe import GLU, SparseMoeBlock
from fla.modules import RMSNorm
from .cache import Cache
from .attention import FlashAttention
from .linear_attention import HLinearAttention
logger = logging.get_logger(__name__)

# class GLU(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         intermediate_size: Optional[int] = None,
#         hidden_act: str = 'swish'
#     ) -> GLU:
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = ACT2FN[hidden_act]

#     def forward(self, hidden_state):
#         gate = self.gate_proj(hidden_state)
#         y = self.up_proj(hidden_state)
#         return swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)


class IntraHybridAttention(nn.Module):
    def __init__(self, config: HymetaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        self.vanilla_attn = FlashAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            use_swa=(layer_idx not in config.full_attn_layers),
            sliding_window=config.sliding_window,
            num_meta_tokens=config.num_meta_tokens,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            layer_idx=layer_idx,
        )
        self.linear_attn = HLinearAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_meta_tokens=config.num_meta_tokens,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            layer_idx=layer_idx,
        )

        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.norm1 = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.norm2 = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lower_bound: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        input_meta_tokens: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        out_attn, _, past_key_values = self.vanilla_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_meta_tokens=input_meta_tokens,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs
        )

        out_linear, _, past_key_values = self.linear_attn(
            hidden_states=hidden_states,
            lower_bound=lower_bound,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_meta_tokens=input_meta_tokens,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs
        )

        hidden_states = (self.norm1(out_attn) + self.norm2(out_linear)) / 2
        hidden_states = self.out_proj(hidden_states)

        return hidden_states, None, past_key_values

class HybridBlock(nn.Module):
    def __init__(self, config: HymetaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.attn = IntraHybridAttention(config=config, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        # self.mlp = GLU(
        #     hidden_size=config.hidden_size,
        #     intermediate_size=config.intermediate_size,
        #     hidden_act=config.hidden_act,
        # )
        if layer_idx in config.dense_mlp_layers:
            
            self.mlp = GLU(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
        else:
            self.block_sparse_moe = SparseMoeBlock(config)
            shared_intermediate = getattr(config, 'shared_intermediate_size', 0)
            self.shared_moe = False
            if shared_intermediate > 0:
                self.shared_moe = True
                self.shared_mlp = GLU(config.hidden_size, config.intermediate_size)
                # self.coefficient = torch.nn.Linear(config.hidden_size, 1, bias=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        lower_bound: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        input_meta_tokens: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            lower_bound=lower_bound,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_meta_tokens=input_meta_tokens,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs
        )
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        if self.layer_idx in self.config.dense_mlp_layers:
            hidden_states = self.mlp(hidden_states)
        else:
            moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)
            if self.shared_moe :
                output_mlp = self.shared_mlp(hidden_states)
                hidden_states = moe_hidden_states + output_mlp
            else :
                hidden_states = moe_hidden_states
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs


class HybridPreTrainedModel(PreTrainedModel):

    config_class = HymetaConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['HybridBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class HybridModel(HybridPreTrainedModel):

    def __init__(self, config: HymetaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_lower_bound = config.use_lower_bound
        self.num_meta_tokens = config.num_meta_tokens

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([HybridBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))
        if config.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(torch.randn(config.num_meta_tokens, config.hidden_size))

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`GLAModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and not isinstance(past_key_values, Cache):
                past_key_values = Cache.from_legacy_cache(past_key_values)
        
        input_meta_tokens = (self.num_meta_tokens > 0) and (past_key_values is None or past_key_values.get_seq_length() == 0)
        if input_meta_tokens:
            meta_tokens = repeat(self.meta_tokens, 'n d -> b n d', b = batch_size)
            inputs_embeds = torch.cat((meta_tokens, inputs_embeds), dim=1)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.use_lower_bound else None
                
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    lower_bound,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    input_meta_tokens,
                    use_cache,
                    output_attentions,
                    cache_position,
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    lower_bound=lower_bound,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    input_meta_tokens=input_meta_tokens,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if input_meta_tokens:
            hidden_states = hidden_states[:, self.num_meta_tokens:, :]

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class HymetaForCausalLM(HybridPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HybridModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ):
        # complete attention_mask according to meta_tokens
        if self.config.num_meta_tokens > 0:
            attention_mask = torch.cat([torch.ones(input_ids.shape[0], self.config.num_meta_tokens, \
                                                   device=attention_mask.device), attention_mask], dim=1)

        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            if not isinstance(past_key_values, Cache):
                past_key_values = Cache.from_legacy_cache(past_key_values, input_ids.shape[1] - 1)
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}
        
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'cache_position': cache_position,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(logits.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
