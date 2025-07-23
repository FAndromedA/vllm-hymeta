from typing import Tuple

import torch

from .my_fused_recurrent import my_fused_recurrent


def my_fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor = None,
    kv_caches: torch.Tensor = None,
    slot_idx: torch.Tensor = None,
    scale: int = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    reverse: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o = my_fused_recurrent(q, k, v, g, kv_caches, slot_idx, scale, initial_state, output_final_state, reverse)
    return o
