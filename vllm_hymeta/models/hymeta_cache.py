# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mamba_cache.py
from dataclasses import dataclass

import torch

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.model_executor.models.constant_size_cache import ConstantSizeCache

@dataclass
class HymetaCacheParams:
    hymeta_cache: torch.Tensor = torch.Tensor()
    meta_linear_cache: torch.Tensor = torch.Tensor()
    meta_fattn_cache: torch.Tensor = torch.Tensor()
    # state_indices_tensor is used to track the state indices of the current run.
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return HymetaCacheParams(self.hymeta_cache[layer_idx, ...],
                                 self.meta_linear_cache[layer_idx, ...],
                                 self.meta_fattn_cache[layer_idx, ...],
                                 self.state_indices_tensor)


class HymetaCacheManager(ConstantSizeCache):

    def __init__(self, dtype, cache_shape, 
                 meta_linear_cache_shape,
                 meta_fattn_cache_shape,):
        super().__init__(cache_shape[1]) # max_batch_size is cache_shape[1]
        
        hymeta_cache = torch.empty(size=cache_shape,
                                         dtype=dtype,
                                         device="cuda")
        meta_linear_cache = torch.zeros(size=meta_linear_cache_shape,
                                       dtype=dtype,
                                       device="cuda")
        meta_fattn_cache = torch.zeros(size=meta_fattn_cache_shape,
                                       dtype=dtype,
                                       device="cuda")
        self._hymeta_cache = (hymeta_cache, meta_linear_cache, meta_fattn_cache)

    @property
    def cache(self):
        return self._hymeta_cache

    def _copy_cache(self, from_index: int, to_index: int):
        assert len(self.cache) > 0
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                        non_blocking=True)
            
    def current_run_tensors(self, **kwargs):
        cache_tensors, state_indices_tensor = super().current_run_tensors(**kwargs)
        return (cache_tensors[0], cache_tensors[1], cache_tensors[2], state_indices_tensor)

    def get_seqlen_agnostic_capture_inputs(self, batch_size):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the Mamba Cache during the CUDA graph
        replay runs.
        """
        return self._hymeta_cache, torch.as_tensor(
            [PAD_SLOT_ID] * batch_size, dtype=torch.int32, device="cuda")