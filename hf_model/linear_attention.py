# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN

# from fla.models.utils import Cache
from fla.modules import ShortConvolution
from fla.modules.activations import swish
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
# from .norms import RMSNorm
from fla.modules import RMSNorm

from .cache import Cache


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)    


class HLinearAttention(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        num_meta_tokens: int = 128,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        clamp_max: Optional[float] = 0.95,
        layer_idx: int = None,
    ) -> HLinearAttention:
        super().__init__()

        self.mode = mode
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.num_meta_tokens = num_meta_tokens
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.key_dim = hidden_size
        self.head_dim = self.hidden_size // num_heads
        self.clamp_max = clamp_max
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.num_key_value_heads * self.head_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.num_key_value_heads * self.head_dim, conv_size, activation='silu')

        self.q_feature_map = ACT2FN['relu']

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        lower_bound: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        input_meta_tokens: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = None
        # if past_key_values is not None and len(past_key_values) > self.layer_idx:
        if past_key_values is not None and past_key_values[self.layer_idx]['recurrent_state'] is not None:
            last_state = past_key_values[self.layer_idx]
            attention_mask = attention_mask[:, -1:] if attention_mask is not None else None
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q = self.q_conv1d(q, attention_mask, conv_state_q)
            k = self.k_conv1d(k, attention_mask, conv_state_k)
            v = self.v_conv1d(v, attention_mask, conv_state_v)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # dealing with left-padding
        if attention_mask is not None:
            if input_meta_tokens:
                v1 = v[:, self.num_meta_tokens:, ...]
                v[:, self.num_meta_tokens:, ...] = v1.mul_(attention_mask.unsqueeze(-1))
            else:
                v = v.mul_(attention_mask.unsqueeze(-1))
            
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_key_value_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_key_value_heads)
        
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q = self.q_feature_map(q)
        # improve precision
        k = k.float()

        # the lower bound for the first layer is zero
        if lower_bound is None or self.layer_idx == 0:
            k = k.sigmoid()
            k = torch.clamp(k, max=self.clamp_max)
            g = (1 - k).log()
        else:
            k = k.sigmoid()
            lower_bound = rearrange(lower_bound, '(h d) -> 1 h 1 d', h=self.num_heads)
            g = lower_bound + (1 - lower_bound) * (1 - k)
            k, g = 1 - g, g.log()
        k = k.to(v)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(q, k, v, g, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(q, k, v, g, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(q, k, v, g, initial_state=recurrent_state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            conv_state = (conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=0
                # offset=q.shape[2]
            )

        o = rearrange(o, 'b h l d -> b l (h d)').contiguous()

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                      param.new_zeros(batch_size, self.num_key_value_heads * self.head_dim, self.conv_size),
                      param.new_zeros(batch_size, self.num_key_value_heads * self.head_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_dim, self.head_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size