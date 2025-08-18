
from __future__ import annotations

import copy
import math
import warnings
from typing import List, Optional, Tuple, Union, Iterable

import regex as re

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.activations import ACT2FN
# from transformers.modeling_outputs import ( BaseModelOutputWithPast,
#                                            CausalLMOutputWithPast)
# from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import logging

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_pp_group, get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
# from vllm.model_executor.layers.lightning_attn import (
#     lightning_attention, linear_decode_forward_triton
# )
from vllm.model_executor.layers.linear import (
    # ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.models.utils import PPMissingLayer, is_pp_missing_parameter, make_layers, maybe_prefix
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsV0Only, SupportsPP
from vllm.sequence import IntermediateTensors

from fla.modules import ShortConvolution
from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla

from .my_fused_recurrent_gla import my_fused_recurrent_gla
from .hymeta_cache import HymetaCacheManager, HymetaCacheParams
from .configuration_hymeta import HymetaConfig
from .attention import MetaAttention

import os
def log_tensor_to_file(tensor, layer_idx, tag, suffix="", directory="/root/docker_shared/vllm-hymeta/test_vllm/debug_logs", hidden_state_len=0):
    """Log a tensor to a unique file for debugging."""
    if (hidden_state_len != 8192) and (hidden_state_len != 8192 + 128):
        return
    os.makedirs(directory, exist_ok=True)
    device_id = torch.cuda.current_device() if tensor.is_cuda else "cpu"
    filename = f"{directory}/layer{layer_idx:03d}_{tag}_device{device_id}{suffix}.pt"
    torch.save(tensor.detach().cpu(), filename)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    if hidden_states.dim == 4:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape 
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    # [num_tokens, num_heads, head_dim]
    assert hidden_states.dim() == 3, \
        "hidden_states should be 3D tensor, but got {}".format(hidden_states.dim())
    num_tokens, num_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :].expand(num_tokens, num_heads, n_rep, head_dim)
    return hidden_states.reshape(num_tokens, num_heads * n_rep, head_dim)


def replace_weight_name(name: str,
                        key: str = None,
                        to: str = None,
                        count: int = None,
                        prefix: str = None) -> str:
    name = name.replace(key, to) if count is None else \
        name.replace(key, to, count=count)
    return name

# def weight_loader_with_alias(alias: str):

#     def wrapper(func: callable):

#         def inner_func(param: torch.Tensor,
#                        loaded_weight: torch.Tensor,
#                        *args,
#                        prefix: str = None,
#                        **kwargs):
#             value = func(param, loaded_weight, *args, **kwargs)
#             return value
        
#         return inner_func
    
#     return wrapper

class HymetaRMSNormTP(CustomOp):
    # 只有在 tensor parallel 后，以及 reduce 前的 RMSNorm 才需要这个 CustomOp
    name = "HymetaRMSNormTP"

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = nn.Parameter(
            torch.ones(int(hidden_size / self.tp_world), dtype=torch.float32)
        )

        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps
        return

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])
        return

    def _forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        if self.tp_world > 1:
            variance = tensor_model_parallel_all_reduce(
                variance) / self.tp_world
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.weight
        if x.size(-1) != self.weight.size(0):
            if self.weight.size(0) < x.size(-1):
                repeat_count = (x.size(-1) + self.weight.size(0)) // x.size(-1)
                full_weight = self.weight.repeat(repeat_count)
                weight = full_weight[:x.size(-1)]
            else:
                weight = self.weight[:x.size(-1)]

        x = x.to(orig_dtype) * weight
        return x

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert residual is None, "RMSNorm does not support residual connection."
        return self._forward(x)
            

class HymetaRotaryEmbedding(CustomOp):
    name = "HymetaRotaryEmbedding"

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        is_neox_style: bool,
        cache_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position
        self.base = base
        self.is_neox_style = is_neox_style
        self.cache_dtype = cache_dtype
        cache = self._compute_cos_sin_cache().to(cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm import _custom_ops as ops
        self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
        query_cast = query.to(self.cache_dtype)
        key_cast = key.to(self.cache_dtype)
        ops.rotary_embedding(positions, query_cast, key_cast, self.head_size,
                             self.cos_sin_cache, self.is_neox_style)
        query = query_cast.to(query.dtype)
        key = key_cast.to(key.dtype)
        return query, key
    
class HymetaMLP(nn.Module): # GLU

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        prefix: str = "mlp",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate_up, _ = self.gate_up_proj(x) # 第二个返回的是 skip_bias_add=True 时的 bias
        x = self.act_fn(gate_up) # silu(gate) * up
        x, _ = self.down_proj(x) # 没有合并 act_fn 和 down_proj, 因为便于并行
        return x

class HymetaMoE(nn.Module):

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        layer_idx: int = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "moe",
    ) -> None:

        super().__init__()

        self.layer_idx = layer_idx
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        self.quant_config =  quant_config

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f'{prefix}.gate',
            return_bias=False
        )
        self.gate.weight.weight_loader = HymetaMoE.gate_weight_loader

        self.experts = FusedMoE(
            num_experts=self.num_total_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size * self.tp_size,
            params_dtype=self.params_dtype,
            reduce_results=True,
            renormalize=False, # softmax topk 之后对权重归一化
            quant_config=None,#self.quant_config,
            tp_size=self.tp_size,
            activation="silu",
            prefix=f"{prefix}.experts",
        )
        return

    @staticmethod
    def gate_weight_loader(param: nn.Parameter,
                           loaded_weight: torch.Tensor) -> None:
        # assert param.size() == loaded_weight.size(), \
        #     f"Expected {param.size()} but got {loaded_weight.size()}"
        # param.data.copy_(loaded_weight)
        # return # AssertionError: Expected torch.Size([3584, 16]) but got torch.Size([16, 3584])
        # 检查是否为转置关系
        if param.size() == loaded_weight.t().size(): 
            # 转置后复制数据
            param.data.copy_(loaded_weight.t())  
        # 若维度完全一致则直接复制
        elif param.size() == loaded_weight.size():  
            param.data.copy_(loaded_weight)
        # 处理无法匹配的情况
        else:  
            raise ValueError(
                f"Shape mismatch: Expected {param.size()} or its transpose, "
                f"but got {loaded_weight.size()}"
            )
        return 

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        # print(f"HymetaMoE {self.layer_idx} forward, "
        #        f"hidden_states has nan: {torch.isnan(hidden_states).any()}, "
        #        f"router_logits has nan: {torch.isnan(router_logits).any()}, "
        #        f"hidden_states sample: {hidden_states[:5, :5]},"
        #        f"router_logits sample: {router_logits[:5, :5]},"
        #        f"hidden_states max/min:", hidden_states.max(), hidden_states.min(),
        #        f"router_logits max/min:", router_logits.max(), router_logits.min()
        #        )
        final_hidden_states = self.experts(
            hidden_states, router_logits.to(hidden_states.dtype))
        final_hidden = final_hidden_states.reshape(num_tokens, hidden_size)
        return final_hidden
    
class HLinearAttention(nn.Module):
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_meta_tokens: int = 0,
        clamp_max: Optional[float] = 0.95,
        layer_idx: int = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "hlinear_attn"
    ) -> HLinearAttention:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.total_num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // num_key_value_heads
        self.num_meta_tokens = num_meta_tokens

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.key_dim = hidden_size
        self.head_dim = self.hidden_size // num_heads
        self.clamp_max = clamp_max
        self.layer_idx = layer_idx

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        assert self.num_heads % self.tp_size == 0, "num_heads must be divisible by tp_size"
        self.tp_heads = self.num_heads // self.tp_size  # 每个 GPU 负责的头数
        
        # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
        if self.total_num_key_value_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            # 即每个 GPU 都有多个 q heads 和多个 kv heads
            assert self.total_num_key_value_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            # 即每个 GPU 有多个 q heads，每个 GPU 只负责一个 kv head 但是一个 kv head 会被多个 GPU 负责
            assert self.tp_size % self.total_num_key_value_heads == 0
        
        self.tp_kv_heads = max(1, self.total_num_key_value_heads // self.tp_size)
        self.q_size = self.tp_heads * self.head_dim
        self.kv_size = self.tp_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.total_num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.q_feature_map = ACT2FN['relu']

    @staticmethod
    def weight_direct_load(param: torch.Tensor,
                           loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        return

    def _prefill_and_mix_infer(self, q, k, v, g, kv_cache, has_meta_cache, 
                            meta_cache, state_indices_tensor, attn_metadata, is_vllm_testing):
        hidden = []
        # 当前 layer 还没有计算过 meta_cache，则计算一遍，并一劳永逸
        if has_meta_cache == False and not is_vllm_testing:
        # if self.num_meta_tokens > 0 and getattr(attn_metadata, "num_prefills", 0) > 0:
            _start = 0
            _end = self.num_meta_tokens
            # 处理 meta_tokens
            q_slice = q[_start:_end].transpose(0, 1).contiguous() # [num_heads, num_tokens, head_dim]
            k_slice = k[_start:_end].transpose(0, 1).contiguous()
            v_slice = v[_start:_end].transpose(0, 1).contiguous()
            g_slice = g[_start:_end].transpose(0, 1).contiguous()
            # 目前是每次都重新计算一遍
            use_cache = True

            should_pad_dim = q_slice.dim() == 3
            if should_pad_dim:
                q_slice = q_slice.unsqueeze(0) # [1, num_heads, num_tokens, head_dim]
                k_slice = k_slice.unsqueeze(0)
                v_slice = v_slice.unsqueeze(0)
                g_slice = g_slice.unsqueeze(0)
            o, recurrent_state = fused_chunk_gla(
                q_slice, k_slice, v_slice, g_slice, initial_state=None, output_final_state=use_cache)
            
            meta_cache.copy_(recurrent_state.squeeze(0)) # [num_heads, head_dim, head_dim]
            hidden.append(rearrange(o.squeeze(0), 'h n d -> n (h d)').contiguous()) # [num_tokens, h*d]

        # 处理 prefill 请求
        for _prefill_idx in range(getattr(attn_metadata, "num_prefills", 0)):
            if _prefill_idx >= len(attn_metadata.query_start_loc):
                break
            if _prefill_idx >= len(state_indices_tensor):
                break
            # 因为把 meta_tokens 插到了最前面，所以需要加上 num_meta_tokens，所有 prefill 请求共用。 
            meta_tokens_offset = self.num_meta_tokens \
                            if (not has_meta_cache and not is_vllm_testing) else 0
            _start = attn_metadata.query_start_loc[_prefill_idx] + meta_tokens_offset
            _end = attn_metadata.query_start_loc[_prefill_idx + 1] + meta_tokens_offset  # 取出这个 prefill 请求的 tokens 范围
            slot_id = state_indices_tensor[_prefill_idx]

            q_slice = q[_start:_end].transpose(0, 1).contiguous() # [num_heads, num_tokens, head_dim]
            k_slice = k[_start:_end].transpose(0, 1).contiguous()
            v_slice = v[_start:_end].transpose(0, 1).contiguous()
            g_slice = g[_start:_end].transpose(0, 1).contiguous()
            # 为什么 prefill 请求还会有 Cache, 这是因为 backend 一个 prefill 可能被拆分成多个请求
            # https://zhuanlan.zhihu.com/p/1916181593229334390
            initial_state = kv_cache[slot_id, ...].unsqueeze(0) # [1, num_heads, head_dim, head_dim]
            if initial_state.isnan().any(): # 未初始化的 kv_cache 可能有 nan
                initial_state = None
            
            # context_len_tensor[idx] 为 0 说明是第一次处理这个 slot_id 的 prefill 请求
            # https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/flash_attn.py
            # NOTE(sang): Definition of context_len, query_len, and seq_len.
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|
            if attn_metadata.context_lens_tensor[_prefill_idx] == 0 and not is_vllm_testing:
                # assert meta_cache.isnan().any() == False, \
                #     f"fused_chunk_gla layer:{self.layer_idx}, meta_cache has nan: {meta_cache.isnan().any()}, \n" 
                initial_state = meta_cache.unsqueeze(0) # [1, num_heads, head_dim, head_dim]

            should_pad_dim = q_slice.dim() == 3
            if should_pad_dim:
                q_slice = q_slice.unsqueeze(0) # [1, num_heads, num_tokens, head_dim]
                k_slice = k_slice.unsqueeze(0)
                v_slice = v_slice.unsqueeze(0)
                g_slice = g_slice.unsqueeze(0)
            o, recurrent_state = fused_chunk_gla(
                q_slice, k_slice, v_slice, g_slice, initial_state=initial_state, output_final_state=True)
            # num_prefills = getattr(attn_metadata, "num_prefills", 0)
            # warnings.warn(
            #     f"fused_chunk_gla layer:{self.layer_idx}, num_prefills: {num_prefills}, start-end: {_start}-{_end}\n"
            #     f"len(state_indices_tensor): {len(state_indices_tensor)}, len(attn_metadata.query_start_loc): {len(attn_metadata.query_start_loc)} \n"
            #     f"output shape: {o.shape}, recurrent_state shape: {recurrent_state.shape}, state_indices_tensor shape: {state_indices_tensor.shape}\n"
            #     f"has_meta_cache: {has_meta_cache}, initial_state shape: {initial_state.shape}, \n"
            #     f"q_slice shape: {q_slice.shape}, q shape: {q.shape}, \n"
            #     f"v_slice shape: {v_slice.shape}, v shape: {v.shape}, \n"
            #     f"g_slice shape: {g_slice.shape}, g shape: {g.shape}, \n"
            #     f"initial_state sample: {initial_state[:, 0, :3, :3]}, \n"
            #     f"recurrent_state sample: {recurrent_state[:, 0, :3, :3]}, \n"
            # )
            if not is_vllm_testing:
                # 因为我自定义的 cache 是 kv, 但是 gla 的 recurrent 是 vk，但是 gla 读的时候又本来就反过来读的，所以传的时候不用 transpose
                kv_cache[slot_id].copy_(recurrent_state.squeeze(0)) # [1, num_heads, head_dim, head_dim]
            hidden.append(rearrange(o.squeeze(0), 'h n d -> n (h d)').contiguous()) # [num_tokens, h*d]
            # assert o.isnan().any() == False, \
            #     f"fused_chunk_gla layer:{self.layer_idx}, o has nan: {o.isnan().any()}, \n" \
            #     f"o shape: {o.shape}, o sample: {o[0, 0, :5]}, \n" \
            #     f"recurrent_state shape: {recurrent_state.shape}, recurrent_state sample: {recurrent_state[0, 0, :3, :3]}, \n" \
            #     f"recurrent_state has nan: {recurrent_state.isnan().any()}, min/max {recurrent_state.min()}/{recurrent_state.max()} \n" \
            #     f"q_slice shape: {q_slice.shape}, q_slice sample: {q_slice[0, 0, :5]}, \n" \
            #     f"k_slice shape: {k_slice.shape}, k_slice sample: {k_slice[0, 0, :5]}, \n" \
            #     f"v_slice shape: {v_slice.shape}, v_slice sample: {v_slice[0, 0, :5]}, \n" \
            #     f"initial_state shape: {initial_state.shape}, initial_state sample: {initial_state[0, 0, :3, :3]}, \n" 

        if attn_metadata.num_decode_tokens > 0: # 有需要解码的 tokens
            hidden.append(
                self._decode(q, k, v, g, kv_cache, has_meta_cache,
                             state_indices_tensor, attn_metadata, is_vllm_testing)
            )
        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)
        
        hidden = torch.concat(hidden, dim=0).contiguous() # [num_tokens, h*d]
        return hidden
    
    def _decode(self, q, k, v, g, kv_cache, has_meta_cache, state_indices_tensor, attn_metadata, is_vllm_testing):
        # 因为解码的请求的长度都是 1， 这里的 num_tokens 维度等效于 batch 维度(同时处理多个解码请求)
        meta_tokens_offset = self.num_meta_tokens if (not has_meta_cache and not is_vllm_testing) else 0
        _start = attn_metadata.num_prefill_tokens + meta_tokens_offset
        q = q[_start:].unsqueeze(2).contiguous() # [batch=num_tokens, num_heads, seqlen=1, head_dim]
        k = k[_start:].unsqueeze(2).contiguous()
        v = v[_start:].unsqueeze(2).contiguous()
        g = g[_start:].unsqueeze(2).contiguous()
        slot_ids = state_indices_tensor[getattr(attn_metadata, "num_prefills", 0):]
        
        # Just for testing
        # kv_cache[slot_ids[0]] = torch.zeros_like(kv_cache[slot_ids[0]], device=kv_cache.device, dtype=kv_cache.dtype)
        # if not is_vllm_testing:
        #     o_fla, recurrent_state = fused_recurrent_gla(q[:1,...], k[:1,...], v[:1,...], g[:1,...], 
        #                                 initial_state=kv_cache[slot_ids[0]].unsqueeze(0), output_final_state=True)
        #     o_fla = o_fla.reshape(1, -1) # [1, h*d]
        #     # just fo debugging my_gla
        o = my_fused_recurrent_gla(q, k, v, g, kv_caches=kv_cache, slot_idx=slot_ids) # [num_tokens, h*d]
        
        # assert torch.allclose(o[0], o_fla, atol=1e-1), \
        #     f"fla and my_fla output not close, layer: {self.layer_idx}, o shape: {o.shape}, o sample: {o[0, :50]}, \n" \
        #     f"fla output shape: {o_fla.shape}, fla output sample: {o_fla[0, :50]}, \n" \
        #     f"torch.allclose(o, o_fla, atol=1e-1): {torch.allclose(o, o_fla, atol=1e-1)}, \n" \
        #     f"kv_cache[slot_ids[0]] shape: {kv_cache[slot_ids[0]].shape}, \n" \
        #     f"kv_cache[slot_ids[0]] sample: {kv_cache[slot_ids[0]][0, :5, :5]}, \n" \
        #     f"recurrent_state shape: {recurrent_state.shape}, \n" \
        #     f"recurrent_state sample: {recurrent_state[0, 0, :5, :5]}, \n" 
        return o

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_caches: HymetaCacheParams,
        has_meta_cache: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        is_vllm_testing: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        
        # if self.use_short_conv: # config.use_short_conv 是 False， 因为这部分比较麻烦，所以先没实现

        qkv, _ = self.qkv_proj(hidden_states) # [num_tokens, hidden_size]
        # qkv32 = qkv.to(torch.float32)
            # qkvact = torch.nn.functional.silu(qkv32)
        # qkvact = qkv.view(qkv.shape[0], self.tp_heads, -1) 
        # q, k, v = torch.split(qkvact, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(q.shape[0], -1, self.head_dim) # [num_tokens, num_heads, head_dim]
        k = k.reshape(k.shape[0], -1, self.head_dim) # [num_tokens, num_heads, head_dim]
        v = v.reshape(v.shape[0], -1, self.head_dim) # [num_tokens, num_heads, head_dim]

        # log_tensor_to_file(q[:, :, :200], self.layer_idx, "linear_attn_q", hidden_state_len=q.shape[0])
        # log_tensor_to_file(k[:, :, :200], self.layer_idx, "linear_attn_k", hidden_state_len=k.shape[0])
        # log_tensor_to_file(v[:, :, :200], self.layer_idx, "linear_attn_v", hidden_state_len=v.shape[0])

        # 这里之后可以优化不 repeat_kv 
        k = repeat_kv(k, self.tp_heads // self.tp_kv_heads) # [num_tokens, num_heads, head_dim]
        v = repeat_kv(v, self.tp_heads // self.tp_kv_heads)

        q = self.q_feature_map(q)
        # improve precision
        k = k.float()

        # the lower bound for the first layer is zero
        if lower_bound is None or self.layer_idx % 7 == 0:
            k = k.sigmoid()
            k = torch.clamp(k, max=self.clamp_max)
            g = (1 - k).log()
        else:
            k = k.sigmoid()
            lower_bound = rearrange(lower_bound, '(h d) -> 1 h d', h=self.num_heads)
            start_head = self.tp_rank * self.tp_heads
            end_head = start_head + self.tp_heads
            lower_bound = lower_bound[:, start_head:end_head, :]
            # g = lower_bound + (1 - lower_bound) * (1 - k)
            g = 1 - (1 - lower_bound) * k
            k, g = 1 - g, g.log()
        k = k.to(v)

        # log_tensor_to_file(q[:, :, :], self.layer_idx, "linear_attn_q_after", hidden_state_len=q.shape[0])
        # log_tensor_to_file(k[:, :, :], self.layer_idx, "linear_attn_k_after", hidden_state_len=k.shape[0])
        # log_tensor_to_file(v[:, :, :], self.layer_idx, "linear_attn_v_after", hidden_state_len=v.shape[0])
        # log_tensor_to_file(g[:, :, :], self.layer_idx, "linear_attn_g_after", hidden_state_len=g.shape[0])

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        
        kv_cache = kv_caches.hymeta_cache
        meta_cache = kv_caches.meta_linear_cache
        state_indices_tensor = kv_caches.state_indices_tensor

        decode_only = getattr(attn_metadata, "num_prefills", 0) == 0

        # outputs = []
        if not (decode_only) or (not has_meta_cache): 
            # 这里 or not has_meta_cache 是因为 vllm 测试的时候出现 decode only 但是前面没有 prefill 过的请求。
            outputs = self._prefill_and_mix_infer(q, k, v, g, 
                                                kv_cache, 
                                                has_meta_cache,
                                                meta_cache,
                                                state_indices_tensor,
                                                attn_metadata,
                                                is_vllm_testing)
        else:
            outputs = self._decode(q, k, v, g, kv_cache, has_meta_cache,
                                state_indices_tensor, attn_metadata, is_vllm_testing)
        
        return outputs

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size


class FlashAttention(nn.Module):

    def __init__(
        self,
        config: HymetaConfig,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        max_position: int = 8192,#4096 * 32,
        num_meta_tokens: int = 128,
        rope_theta: float = 10000,
        # sliding_window: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "hflash_attn",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads

        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.num_meta_tokens = num_meta_tokens
        self.rope_theta = rope_theta
        
        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.attn = MetaAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            num_meta_tokens=num_meta_tokens,
        )
        return

    def forward(self, hidden_states: torch.Tensor, 
                positions: torch.Tensor,
                kv_caches: HymetaCacheParams,
                has_meta_cache: Optional[bool] = False,
                is_vllm_testing: bool = False,
                **kwargs) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        qkv, _ = self.qkv_proj(hidden_states) # [num_tokens, hidden_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # log_tensor_to_file(q[:, :200], self.layer_idx, "flash_attn_q", hidden_state_len=q.shape[0])
        # log_tensor_to_file(k[:, :200], self.layer_idx, "flash_attn_k", hidden_state_len=k.shape[0])
        # log_tensor_to_file(v[:, :200], self.layer_idx, "flash_attn_v", hidden_state_len=v.shape[0])

        q, k = attn_metadata.rotary_emb(positions, q, k)

        # log_tensor_to_file(q[:, :], self.layer_idx, "flash_attn_q_after_rotary", hidden_state_len=q.shape[0])
        # log_tensor_to_file(k[:, :], self.layer_idx, "flash_attn_k_after_rotary", hidden_state_len=k.shape[0])
        
        # warnings.warn(f"FlashAttention{self.layer_idx} before forward, "
        #       f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}, "
        #       f"q has nan: {torch.isnan(q).any()}, min/max: {q.min()}/{q.max()}, "
        #       f"k has nan: {torch.isnan(k).any()}, min/max: {k.min()}/{k.max()}, "
        #       f"v has nan: {torch.isnan(v).any()}, min/max: {v.min()}/{v.max()}, "
        #       f"hidden_states has nan: {torch.isnan(hidden_states).any()}, min/max: {hidden_states.min()}/{hidden_states.max()}, "
        #        )
        q1, k1, v1 = q, k, v
        q2, k2, v2 = None, None, None
        if (has_meta_cache == False) and (self.num_meta_tokens > 0) and (is_vllm_testing == False):
            q1 = q1[self.num_meta_tokens:] # reshape is done in attn
            k1 = k1[self.num_meta_tokens:]
            v1 = v1[self.num_meta_tokens:]
            # 只有第一次处理 meta_tokens 时才会需要 q2 计算
            q2 = q[:self.num_meta_tokens].reshape(self.num_meta_tokens, self.num_heads, self.head_dim)
            
            kv_caches.meta_fattn_cache[0].copy_(
                k[:self.num_meta_tokens].reshape(self.num_meta_tokens, self.num_kv_heads, self.head_dim)
            )
            kv_caches.meta_fattn_cache[1].copy_(
                v[:self.num_meta_tokens].reshape(self.num_meta_tokens, self.num_kv_heads, self.head_dim)
            )
        
        if self.num_meta_tokens > 0:
            k2 = kv_caches.meta_fattn_cache[0]
            v2 = kv_caches.meta_fattn_cache[1]

        attn_output = self.attn(q1, k1, v1, query2=q2, key2=k2, value2=v2)
        # if is_vllm_testing == False and torch.isnan(attn_output).any():
        #     warnings.warn(f"FlashAttention{self.layer_idx} after forward, "
        #         f"attn_output shape: {attn_output.shape}, "
        #         f"attn_output has nan: {torch.isnan(attn_output).any()}, "
        #         f"attn_output min/max: {attn_output.min()}/{attn_output.max()}, "
        #         f"attn_output all zeros: {torch.all(attn_output == 0)}, "
        #     )
        #     print(f"Warning: FlashAttention{self.layer_idx} has nan in attn_output, "
        #           f"attn_output shape: {attn_output.shape}, "
        #           f"attn_output sample: {attn_output}, \n"
        #           f"q1 shape: {q1.shape}, k1 shape: {k1.shape}, v1 shape: {v1.shape}, \n"
        #           f"q2 shape: {q2.shape if q2 is not None else 'None'}, "
        #           f"k2 shape: {k2.shape if k2 is not None else 'None'}, "
        #           f"v2 shape: {v2.shape if v2 is not None else 'None'}\n"
        #           f"q1 sample: {q1},\n k1 sample: {k1},\n v1 sample: {v1}, \n"
        #           f"q2 sample: {q2 if q2 is not None else 'None'}, \n"
        #           f"k2 sample: {k2 if k2 is not None else 'None'}, \n"
        #           f"v2 sample: {v2 if v2 is not None else 'None'}")
        #     exit(0)
        return attn_output

class IntraHybridAttention(nn.Module):
    def __init__(self, config: HymetaConfig, 
                 layer_idx: int, 
                 quant_config: Optional[QuantizationConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "intra_hybrid_attn"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        self.vanilla_attn = FlashAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=self.head_dim,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            # use_swa=(layer_idx not in config.full_attn_layers),
            # sliding_window=config.sliding_window,
            num_meta_tokens=config.num_meta_tokens,
            rope_theta=config.rope_theta,
            layer_idx=layer_idx,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.vanilla_attn",
        )
        self.linear_attn = HLinearAttention(
            # mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            num_meta_tokens=config.num_meta_tokens,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_attn",
        )

        self.out_proj =  RowParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size, 
            bias=False, prefix=f"{prefix}.out_proj", return_bias=False)
        self.norm1 = HymetaRMSNormTP(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.norm2 = HymetaRMSNormTP(hidden_size=config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        lower_bound: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_caches: HymetaCacheParams = None,
        has_meta_cache: Optional[bool] = False,
        is_vllm_testing: bool = False,
        **kwargs,
    ) -> torch.FloatTensor:
        # log_tensor_to_file(self.norm1.weight, self.layer_idx, "norm1_weight", hidden_state_len=8192)
        # log_tensor_to_file(self.norm2.weight, self.layer_idx, "norm2_weight", hidden_state_len=8192)
        # exit(0)
        out_attn = self.vanilla_attn(
            hidden_states=hidden_states,
            positions=position_ids,
            has_meta_cache=has_meta_cache,
            kv_caches=kv_caches,
            is_vllm_testing=is_vllm_testing,
            **kwargs
        )

        out_linear = self.linear_attn(
            hidden_states=hidden_states,
            lower_bound=lower_bound,
            has_meta_cache=has_meta_cache,
            kv_caches=kv_caches,
            is_vllm_testing=is_vllm_testing,
            **kwargs
        )
        # log_tensor_to_file(out_attn[:, :], self.layer_idx, "out_attn", hidden_state_len=out_attn.shape[0])
        # log_tensor_to_file(out_linear[:, :], self.layer_idx, "out_linear", hidden_state_len=out_linear.shape[0])

        # warnings.warn(f"IntraHybridAttention{self.layer_idx}, out_attn: {out_attn.shape}, out_linear: {out_linear.shape}, hidden_states: {hidden_states.shape}\n"
        #         f"out_attn has nan: {torch.isnan(out_attn).any()}, "
        #         f"out_linear has nan: {torch.isnan(out_linear).any()}, "
        #         f"hidden_states has nan: {torch.isnan(hidden_states).any()}, "
        #         f"out_attn sample: {out_attn[:5, :5]}, "
        #         f"out_linear sample: {out_linear[:5, :5]}, "
        #         f"hidden_states sample: {hidden_states[:5, :5]}, "
        #         f"hidden_states max/min: {hidden_states.max()}, {hidden_states.min()}"
        #        )

        out_attn_norm = self.norm1(out_attn)
        out_linear_norm = self.norm2(out_linear)
        # log_tensor_to_file(out_attn_norm[:, :], self.layer_idx, "out_attn_norm", hidden_state_len=out_attn_norm.shape[0])
        # log_tensor_to_file(out_linear_norm[:, :], self.layer_idx, "out_linear_norm", hidden_state_len=out_linear_norm.shape[0])

        hidden_states = (out_attn_norm + out_linear_norm) / 2


        # log_tensor_to_file(hidden_states[:, :200], self.layer_idx, "out_swa_lin_avg", hidden_state_len=hidden_states.shape[0])

        hidden_states = hidden_states.to(torch.bfloat16)
        hidden_states = self.out_proj(hidden_states)
        
        # assert torch.isnan(hidden_states).any() == False, \
        #     f"IntraHybridAttention {self.layer_idx} forward has nan in hidden_states, " \
        #     f"out_attn: {out_attn.shape}, out_linear: {out_linear.shape}, " \
        #     f"out_attn has nan: {torch.isnan(out_attn).any()}, " \
        #     f"out_linear has nan: {torch.isnan(out_linear).any()}, "
        return hidden_states

class HybridBlock(nn.Module):
    def __init__(self, 
                 config: HymetaConfig, 
                 layer_idx: int,
                 expert_num: int = 1,
                 quant_config: Optional[QuantizationConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "decoder",):
        super().__init__()

        self.expert_num = expert_num
        self.layer_idx = layer_idx
        self.cache_config = cache_config

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.attn = IntraHybridAttention(
            config=config, 
            layer_idx=layer_idx,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        
        if expert_num == 1:
            self.mlp = HymetaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.block_sparse_moe = HymetaMoE(
                num_experts=expert_num,
                top_k=config.num_experts_per_topk,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                params_dtype=torch.bfloat16,
                layer_idx=layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        
            self.shared_moe = False
            
            shared_intermediate = getattr(config, "shared_intermediate_size", 0)
            if isinstance(shared_intermediate, list):
                if len(shared_intermediate) > layer_idx:
                    shared_intermediate = shared_intermediate[layer_idx]
                else:
                    shared_intermediate = 0
            if shared_intermediate > 0:
                self.shared_moe = True
                self.shared_mlp = HymetaMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=shared_intermediate,
                    quant_config=quant_config,
                    layer_idx=layer_idx,
                    prefix=f"{prefix}.shared_mlp",
                )
                # self.coefficient = ReplicatedLinear(
                #     self.hidden_size,
                #     1,
                #     bias=False,
                #     params_dtype=torch.float32,
                #     quant_config=quant_config,
                #     # prefix=f"{prefix}.coefficient",
                # )
                # self.coefficient.weight.weight_loader = (
                #     self.shared_moe_coefficient_loader
                # )
                # self.shared_moe_mode = getattr(config, "shared_moe_mode", "softmax")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        lower_bound: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_caches: HymetaCacheParams = None,
        has_meta_cache: Optional[bool] = False,
        is_vllm_testing: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        residual = hidden_states
        # log_tensor_to_file(hidden_states[:, :], self.layer_idx, "input", hidden_state_len=hidden_states.shape[0])

        hidden_states = self.attn_norm(hidden_states)
        # log_tensor_to_file(hidden_states[:, :], self.layer_idx, "after_attn_norm", hidden_state_len=hidden_states.shape[0])

        hidden_states = self.attn(
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            lower_bound=lower_bound,
            position_ids=position_ids,
            has_meta_cache=has_meta_cache,
            kv_caches=kv_caches,
            is_vllm_testing=is_vllm_testing,
            **kwargs
        )
        # log_tensor_to_file(hidden_states[:, :], self.layer_idx, "after_attn", hidden_state_len=hidden_states.shape[0])

        norm_hidden_states, residual = self.mlp_norm(hidden_states, residual)
        # log_tensor_to_file(norm_hidden_states[:, :], self.layer_idx, "after_mlp_norm", hidden_state_len=hidden_states.shape[0])

        # assert torch.isnan(norm_hidden_states).any() == False, \
        #     f"HybridBlock {self.layer_idx} forward after attn has nan in norm_hidden_states, "
        if self.expert_num == 1:
            hidden_states = self.mlp(norm_hidden_states)
            # log_tensor_to_file(hidden_states[:, :], self.layer_idx, "after_mlp", hidden_state_len=hidden_states.shape[0])
        else:
            moe_hidden_states = self.block_sparse_moe(copy.deepcopy(norm_hidden_states))
            # log_tensor_to_file(moe_hidden_states[:, :], self.layer_idx, "after_block_sparse_moe", hidden_state_len=hidden_states.shape[0])
            if self.shared_moe:
                before_moe_dtype = norm_hidden_states.dtype
                # moe_hidden_fp32 = moe_hidden_states.to(torch.float32)
                shared_mlp_out = self.shared_mlp(norm_hidden_states).to(before_moe_dtype)
                # log_tensor_to_file(shared_mlp_out[:, :], self.layer_idx, "after_shared_mlp", hidden_state_len=hidden_states.shape[0])

                # no coef
                # hidden_states = shared_mlp_out + moe_hidden_fp32
                hidden_states = shared_mlp_out + moe_hidden_states
                # coef, _ = self.coefficient(hidden_states.to(torch.float32))
                # if self.shared_moe_mode == "softmax":
                #     coef = torch.nn.functional.softmax(coef, dim=-1)
                #     hidden_states = coef * shared_mlp_out + (1 - coef) * moe_hidden_fp32
                # elif self.shared_moe_mode == "sigmoid":
                #     coef = torch.nn.functional.sigmoid(coef)
                #     hidden_states = coef * shared_mlp_out + (1 - coef) * moe_hidden_fp32
                # hidden_states = hidden_states.to(before_moe_dtype)
            else:
                hidden_states = moe_hidden_states
           
        hidden_states = residual + hidden_states
        # log_tensor_to_file(hidden_states[:, :], self.layer_idx, "final_output", hidden_state_len=hidden_states.shape[0])

        # assert torch.isnan(hidden_states).any() == False, \
        #     f"HybridBlock {self.layer_idx} forward after mlp or moe has nan in hidden_states, " 
        return hidden_states, None
    
    # @staticmethod
    # def shared_moe_coefficient_loader(param: torch.Tensor,
    #                                   loaded_weight: torch.Tensor) -> None:
    #     assert param.size() == loaded_weight.size()
    #     param.data.copy_(loaded_weight.to(torch.float32))
    #     return

class HymetaModel(nn.Module):

    def __init__(
        self,
        config: HymetaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        scheduler_config=None,
        enforce_eager=False,
        prefix: str = "",
    ):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.use_lower_bound = config.use_lower_bound
        self.num_meta_tokens = config.num_meta_tokens

        if get_pp_group().is_first_rank:
            self.embeddings = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=self.vocab_size,
            )
        else:
            self.embeddings = PPMissingLayer()

        def layer_fn(prefix):
            layer_idx = int(prefix.split('.')[-1])
            layer_config = config
            layer_config.layer_idx = layer_idx
            expert_num = 1
            if hasattr(config, "num_layer_experts") and isinstance(
                config.num_layer_experts, list):
                expert_num = config.num_layer_experts[layer_idx]
            if hasattr(config, "num_layer_experts") and isinstance(
                config.num_layer_experts, int):
                expert_num = config.num_layer_experts
            return HybridBlock(
                config=config, 
                layer_idx=layer_idx,
                expert_num=expert_num,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=prefix,
            )
        
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, layer_fn, prefix=f"{prefix}.layers"
        )

        # self.cache = pass
        linear_layer_nums = config.num_hidden_layers
        max_slots_number = scheduler_config.max_num_seqs # Maximum number of sequences to be processed in a single iteration.
        self.cache_shape = (linear_layer_nums, max_slots_number,
                            config.num_attention_heads //
                            get_tensor_model_parallel_world_size(),
                            config.head_dim, config.head_dim)
        # 表示对 meta_tokens 进行线性注意力后的隐状态
        # 因为所有 requests都共享同一个 meta linear cache，所以不需要 slots 维度
        self.meta_linear_cache_shape =  (linear_layer_nums,
                            config.num_attention_heads //
                            get_tensor_model_parallel_world_size(),
                            config.head_dim, config.head_dim) 
        # [num_layers, 2, num_meta_tokens, num_heads, head_dim] , 0 for key, 1 for value
        self.meta_fattn_cache_shape = (
            linear_layer_nums, 2, self.num_meta_tokens,
            max(1, config.num_key_value_heads //
                get_tensor_model_parallel_world_size()),  # 注意这里是 num_key_value_heads
            config.head_dim)

        _dummy = torch.zeros(1)
        self._dtype = _dummy.dtype
        del _dummy

        self.has_meta_cache = 0 # Become True after the first forward pass through all layers.
        self.meta_cache_threshold = 2 if enforce_eager else 37 #, 第一层也是 2 是因为需要给后面的层传
        # but when the vllm is launching, it will run several times ignoring the order of pipelines
        # so we must set it True until it has run beyond a threshold, here we let it be 2 if --enforce-eager
        # else we let it be 1(128k) + 256/8(256,248,...,8) + 3(4, 2, 1) + 1(True request) = 1 + 32 + 3 + 1 = 37
        self.hymeta_cache = HymetaCacheManager(
            dtype=torch.bfloat16,
            cache_shape=self.cache_shape,
            meta_linear_cache_shape=self.meta_linear_cache_shape,
            meta_fattn_cache_shape=self.meta_fattn_cache_shape,
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        if hasattr(config, "max_model_len") and isinstance(
                config.max_model_len, int):
            max_position_embeddings = min(config.max_position_embeddings,
                                          config.max_model_len)
            
        self.rotary_emb = HymetaRotaryEmbedding(
            head_size=head_dim,
            rotary_dim=config.rotary_dim if hasattr(config, "rotary_dim") else head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=True,
            cache_dtype=torch.bfloat16,
            # prefix=f"{prefix}.rotary_emb",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        else:
            self.norm = PPMissingLayer()

        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))
        if config.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(torch.randn(config.num_meta_tokens, config.hidden_size))

        # self.gradient_checkpointing = False
    
    def _clear_prefill_cache(self, attn_metadata,
                            hymeta_cache_tensors: torch.Tensor, **kwargs):
        seq_to_slot_maps = {}
        seq_id_map = sum(list(kwargs["request_ids_to_seq_ids"].values()), [])
        for _, seq_to_slot_map in (
                self.hymeta_cache.cache_indices_mapping.items()):
            seq_to_slot_maps.update(seq_to_slot_map)

        slots_to_clear = []
        for _prefill_id in range(getattr(attn_metadata, "num_prefills", 0)):
            if _prefill_id >= len(seq_id_map):
                break
            seq_id = seq_id_map[_prefill_id]
            # context_len = 0 represents the prefill request is newly created
            # and has not been processed yet, so we can clear the cache for this seq_id
            if attn_metadata.context_lens_tensor[_prefill_id] == 0 \
                    and seq_id in seq_to_slot_maps:
                slots_to_clear.append(seq_to_slot_maps[seq_id])
        
        if slots_to_clear:
            slots_tensor = torch.tensor(slots_to_clear,
                                        device=hymeta_cache_tensors.device,
                                        dtype=torch.long)
            # meta_cache dont need to be cleared because it's alwyas the same
            # for all requests, so we can just reuse it.
            hymeta_cache_tensors[:, slots_tensor, ...] = 0


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return None
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if "request_ids_to_seq_ids" not in kwargs:
            kwargs["request_ids_to_seq_ids"] = {}
        if "finished_requests_ids" not in kwargs:
            kwargs["finished_requests_ids"] = []

        (
            hymeta_cache_tensors,
            meta_linear_cache_tensors,
            meta_fattn_cache_tensors,
            state_indices_tensor,
        ) = self.hymeta_cache.current_run_tensors(**kwargs)

        if getattr(attn_metadata, "num_prefills", 0) > 0:
            self._clear_prefill_cache(attn_metadata, hymeta_cache_tensors, **kwargs)
        
        hymeta_cache_params = HymetaCacheParams(
            hymeta_cache_tensors, meta_linear_cache_tensors,
            meta_fattn_cache_tensors, state_indices_tensor
        )

        # input_meta_tokens = (self.num_meta_tokens > 0) and (past_key_values is None or past_key_values.get_seq_length() == 0)
        # if input_meta_tokens:
        #     meta_tokens = repeat(self.meta_tokens, 'n d -> b n d', b = batch_size)
        #     inputs_embeds = torch.cat((meta_tokens, inputs_embeds), dim=1)
        input_meta_tokens = (self.num_meta_tokens > 0) and (self.has_meta_cache < self.meta_cache_threshold)
        # (getattr(attn_metadata, "num_prefills", 0) > 0) # 不管有没有 prefill 都要

        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.embeddings(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None

            # meta_token 的插入需要在第一个 pipeline stage 处理
            # attn_metadata.num_prefill_tokens > 0 是为了避免 cuda graph 的时候因为全 decode
            # 然而 metatoken attn 的 decode 传入的 q2 为 None，导致 meta cache 存储错误。
            if input_meta_tokens and attn_metadata.num_prefill_tokens > 0:
                hidden_states = torch.cat(
                    (self.meta_tokens, hidden_states), dim=0
                ) # [num_tokens, hidden_size]
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors['hidden_states']
            residual = intermediate_tensors['residual']
        
        is_vllm_testing = False
        if attn_metadata.num_prefill_tokens + \
            attn_metadata.num_decode_tokens == hidden_states.shape[0] and \
                self.has_meta_cache < self.meta_cache_threshold:
            # 这是 vllm 一开始的测试运行, 由于 pipeline parallel 它是并行一起测试的
            # 所以 hidden_states 并不是来源于上一层 pipeline，因此没有上一层的 meta tokens
            # assert get_pp_group().is_first_rank is False
            is_vllm_testing = True

        attn_metadata.rotary_emb = self.rotary_emb

        if self.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        # 不管实际前面插没插入 meta tokens，这里都要加上偏移值
        new_positions = positions + self.num_meta_tokens
        if new_positions.shape[0] < hidden_states.shape[0]:
            # 因为 meta token 的插入导致 positions 的长度变了
            # 不一定是在这个 pipeline stage 添加的，所以这样判断
            meta_positions = torch.arange(
                self.num_meta_tokens, dtype=positions.dtype, device=positions.device
            )
            new_positions = torch.cat(
                (meta_positions, new_positions), dim=0
            )

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            _caches = hymeta_cache_params.at_layer_idx(i)

            lower_bound = lower_bounds[i % 7] if self.use_lower_bound else None
            
            # JUST FOR DEBUGGING
            # warnings.warn(
            #     f"hidden_states at layer {i} shape {hidden_states.shape}, "
            #     f"hidden_states sample: {hidden_states[:5, :5]}, "
            #     f"has_meta_cache: {self.has_meta_cache}, "
            #     f"input_meta_tokens: {input_meta_tokens}, "
            #     f"get_pp_group().is_last_rank: {get_pp_group().is_last_rank}, "
            # )

            hidden_states, residual = layer(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                lower_bound=lower_bound,
                position_ids=new_positions,
                kv_caches=_caches,
                has_meta_cache=(self.has_meta_cache >= self.meta_cache_threshold),
                is_vllm_testing=is_vllm_testing,
                # residual=residual, # 没用
            )
        
        # After the first forward pass, we can set has_meta_cache to True
        self.has_meta_cache = min(254, self.has_meta_cache + 1)
        # if not is_vllm_testing:
        #     self.has_meta_cache = True

        if not get_pp_group().is_last_rank:
            # 不能去掉前面的 meta token 因为下一个 pipeline stage可能需要
            # if input_meta_tokens:
            #     hidden_states = hidden_states[self.num_meta_tokens:, :]
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual":
                    torch.zeros(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device),
                # "residual": residual # None 就别传
                # File "/opt/conda/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 2042, in forward
                # self.input_buffers[key].copy_(intermediate_tensors[key],
                # TypeError: copy_(): argument 'other' (position 1) must be Tensor, not NoneType
            })

        hidden_states = self.norm(hidden_states)
        # 最后一个 pipeline stage 需要去掉前面的 meta token
        if input_meta_tokens and not is_vllm_testing:
            hidden_states = hidden_states[self.num_meta_tokens:, :]

        return hidden_states


class HymetaForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsV0Only, SupportsPP):

    def __init__(self, *, 
                 vllm_config: VllmConfig,
                 prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self._lora_config = lora_config
        self.CONCAT_FFN = True

        if not hasattr(config, "sliding_window"):
            config.sliding_window = None
        
        self.unpadded_vocab_size = config.vocab_size
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len

        self.model = HymetaModel(
            self.config,
            quant_config=quant_config,
            cache_config=vllm_config.cache_config,
            scheduler_config=vllm_config.scheduler_config,
            enforce_eager=vllm_config.model_config.enforce_eager,
            prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                self.config.vocab_size)
        
        else:
            self.lm_head = PPMissingLayer()
        
        self.lm_head.float()
        self.kv_cache = [torch.tensor([]) for _ in range(self.config.num_hidden_layers)]
        return

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.model.hymeta_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )
    
    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.model.hymeta_cache.get_seqlen_agnostic_capture_inputs(
            batch_size
        )

    def get_input_embeddings(
        self, 
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids) 
    
    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, 
                                    inputs_embeds, **kwargs)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                        sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # print(f"compute_logits: hidden_states shape {hidden_states.shape}, "
        #       f"hidden_states sample: {hidden_states[:5, :5]}, ")
        logits = self.logits_processor(self.lm_head, hidden_states.float(),
                                        sampling_metadata)
        # if logits is not None:
        #     print(f"logits shape {logits.shape}, has nan: {torch.isnan(logits).any()}, "
        #             f"logits min/max: {logits.min()}/{logits.max()}, "
        #             f"logits sample: {logits[:5, :5]}")
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype,
        device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })
    
    # https://zhuanlan.zhihu.com/p/1908151478557839879
    def load_weights(self, weights: 
        Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def which_layer(name: str) -> int:
            if "layers" in name:
                after_layer = name.split("layers")[-1]
                return int(after_layer.split(".")[1])
            return None

        def is_linear_attn_layer(name: str) -> bool:
            return "linear_attn" in name

        def is_moe_weight(name: str) -> bool:
            # no bias for moe's experts
            return "block_sparse_moe" in name and not name.endswith(".bias")
        
        def get_expert_id(param_name: str):
            pattern = r'model\.layers\.(\d+)\.block_sparse_moe\.experts.(\d+)\.'
            match = re.search(pattern, param_name)
            if match:
                layer_idx = int(match.group(1))
                expert_id = int(match.group(2))
                return layer_idx, expert_id
            return None, None
        
        def load_sparse_moe_weight(name: str, loaded_weight: torch.Tensor,
                                   self) -> None:
            # from :
            #   model.layers.0.block_sparse_moe.experts.0.gate_proj.weight
            #   model.layers.0.block_sparse_moe.experts.0.up_proj.weight
            #   model.layers.0.block_sparse_moe.experts.0.down_proj.weight
            # to:
            #   model.layers.0.block_sparse_moe.experts.w13.weight
            #   model.layers.0.block_sparse_moe.experts.w2.weight
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_local_experts
            )   
            for (param_name, weight_name, expert_id,
                 shard_id) in expert_params_mapping:
                name_layer_id, name_expert_id = get_expert_id(name)
                if name_expert_id is not None and int(name_expert_id) != int(
                        expert_id):
                    continue
                if weight_name not in name:
                    continue
                if is_pp_missing_parameter(name, self):
                    return
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                # weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight, name, shard_id, expert_id)
                # weight_loader(param,
                #               loaded_weight,
                #               weight_name,
                #               expert_id=expert_id,
                #               shard_id=shard_id)
                loaded_params.add(name)
                # print(f"load_sparse_moe_weight: {name}, weight_name: {weight_name}, param_name: {param_name}, "
                #       f"loaded_weight shape: {loaded_weight.shape}, "
                #       f"expert_id: {expert_id}, shard_id: {shard_id}")
                break
            else: # 如果没有找到对应的 expert 参数名，那么是 moe.gate 直接加载
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                # weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                # print(f"load_sparse_moe_weight: {name}, loaded_weight shape: {loaded_weight.shape}")
            return
        
        def is_shared_mlp_weight(name: str) -> bool:
            return "shared_mlp" in name and not name.endswith(".bias")
        
        def is_densed_mlp_weight(name: str) -> bool:
            return ".mlp." in name
        
        def load_mlp_weight(name: str, loaded_weight: torch.Tensor,
                                   self) -> None:
            if not self.CONCAT_FFN: # CONCAT_FFN is True by default
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "w1", 1)
                elif "up_proj" in name:
                    name = name.replace("up_proj", "w3", 1)
                elif "down_proj" in name:
                    name = name.replace("down_proj", "w2", 1)
            else:
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "gate_up_proj", 1)
                    loaded_shard_id = 0
                elif "up_proj" in name:
                    name = name.replace("up_proj", "gate_up_proj", 1)
                    loaded_shard_id = 1
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            # weight_loader = weight_loader_with_alias(name)(weight_loader)

            if not self.CONCAT_FFN:
                weight_loader(param, loaded_weight)
            else:
                if "gate_up_proj" in name:
                    weight_loader(param, loaded_weight, loaded_shard_id)
                elif "down_proj" in name:
                    weight_loader(param, loaded_weight)
                else:
                    raise AssertionError(
                        "MLP weight not in [gate_up_proj, down_proj]")
            # print(f"load_mlp_weight: {name}, loaded_weight shape: {loaded_weight.shape}")
            loaded_params.add(name)
            return
        
        def is_flash_attn_weight(name: str) -> bool:
            return "vanilla_attn" in name

        def load_attn_weight(name: str, loaded_weight: torch.Tensor,
                                  self) -> None:
            flash_mha_params_mapping = [
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
            ]
            for (param_name, weight_name, shard_id) in flash_mha_params_mapping:
                if weight_name not in name:
                    continue
                if is_pp_missing_parameter(name, self):
                    return
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                # print(f"load_attn_weight: {name}, shard_id: {shard_id}, weight_name: {weight_name}, param_name: {param_name}, loaded_weight shape: {loaded_weight.shape}")
                break
            else:
                if is_pp_missing_parameter(name, self):
                    return
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # weight_loader = weight_loader_with_alias(name)(weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                # print(f"load_attn_weight: {name}")
            return
        
        def is_layer_norm_weight(name: str) -> bool:
            return "norm" in name and not name.endswith(
                ".bias") and name in params_dict

        
        def load_basic_weight(name: str, loaded_weight: torch.Tensor,
                              self) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            # weight_loader = weight_loader_with_alias(name)(weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            # print(f"load_basic_weight: {name}, loaded_weight shape: {loaded_weight.shape}")
            return

        # warnings.warn(
        #     f"The params of the model is: {list(params_dict.keys())}"
        # )
        for name, loaded_weight in weights:
            weight_at_layer = which_layer(name)
            if weight_at_layer is not None and weight_at_layer >= self.config.num_hidden_layers:
                continue
            
            if "mode." in name: # the checkpoint store model.norm wrongly as "mode.norm"
                name = name.replace("mode.", "model.")
            if is_layer_norm_weight(name):
                load_basic_weight(name, loaded_weight, self)
                continue
            if is_flash_attn_weight(name) or is_linear_attn_layer(name):
                load_attn_weight(name, loaded_weight, self)
                continue

            if is_moe_weight(name):
                load_sparse_moe_weight(name, loaded_weight, self)
                continue
            if is_shared_mlp_weight(name) or is_densed_mlp_weight(name):
                load_mlp_weight(name, loaded_weight, self)
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            load_basic_weight(name, loaded_weight, self)
        # exit(0) # for debug
        return loaded_params

