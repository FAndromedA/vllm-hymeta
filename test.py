# from vllm.model_executor.layers.fused_moe import FusedMoE

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional

# from vllm.model_executor.layers.linear import (
#     ColumnParallelLinear,
#     MergedColumnParallelLinear,
#     QKVParallelLinear,
#     ReplicatedLinear,
#     RowParallelLinear
# )

# from vllm.distributed.parallel_state import (
#     get_pp_group, get_tensor_model_parallel_rank,
#     get_tensor_model_parallel_world_size,
#     init_distributed_environment)

# from vllm.model_executor.layers.quantization.base_config import (
#     QuantizationConfig
# )

# from vllm.distributed.parallel_state import (
#     init_distributed_environment,
#     initialize_model_parallel
# )
# class Hmoe(nn.Module):

#     def __init__(
#         self,
#         num_experts: int,
#         top_k: int,
#         hidden_size: int,
#         intermediate_size: int,
#         params_dtype: Optional[torch.dtype] = None,
#         layer_idx: int = None,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "moe",
#     ) -> None:

#         super().__init__()

#         self.layer_idx = layer_idx
#         self.tp_size = get_tensor_model_parallel_world_size()
#         self.num_total_experts = num_experts
#         self.top_k = top_k
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size // self.tp_size
#         self.quant_config =  quant_config

#         if params_dtype is None:
#             params_dtype = torch.get_default_dtype()
#         self.params_dtype = params_dtype

#         self.gate = ReplicatedLinear(
#             self.hidden_size,
#             self.num_total_experts,
#             bias=False,
#             params_dtype=torch.float32,
#             quant_config=None,
#             prefix=f'{prefix}.gate',
#         )
#         self.gate.weight.weight_loader = Hmoe.gate_weight_loader
#         self.qkv_proj = QKVParallelLinear(
#             hidden_size,
#             head_size=5,
#             total_num_heads=7,
#             total_num_kv_heads=11,
#             bias=True,
#             quant_config=quant_config,
#             prefix=f"{prefix}.qkv_proj",
#         )
#         self.experts = FusedMoE(
#             num_experts=self.num_total_experts,
#             top_k=self.top_k,
#             hidden_size=self.hidden_size,
#             intermediate_size=self.intermediate_size * self.tp_size,
#             params_dtype=self.params_dtype,
#             reduce_results=True,
#             renormalize=True, # softmax topk 之后对权重归一化
#             quant_config=self.quant_config,
#             tp_size=self.tp_size,
#             activation="silu",
#             prefix=f"{prefix}.experts",
#         )

#         return

#     @staticmethod
#     def gate_weight_loader(param: nn.Parameter,
#                            loaded_weight: torch.Tensor) -> None:
#         assert param.size() == loaded_weight.size()
#         param.data.copy_(loaded_weight.to(torch.float32))
#         return

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         num_tokens, hidden_size = hidden_states.shape
#         hidden_states = hidden_states.view(-1, self.hidden_size)
#         router_logits_fp32, _ = self.gate(hidden_states.to(torch.float32))
#         final_hidden_states = self.experts(
#             hidden_states, router_logits_fp32.to(hidden_states.dtype))
#         final_hidden = final_hidden_states.view(num_tokens, hidden_size)
#         return final_hidden
    
# init_distributed_environment(
#     world_size=1,
#     rank=0,
#     local_rank=0,
#     distributed_init_method="file:///tmp/tmp_init",
#     backend="gloo"
# )
# initialize_model_parallel(
#     tensor_model_parallel_size=1,
#     pipeline_model_parallel_size=1,
# )

# a = Hmoe(
#     num_experts=15,
#     top_k=1,
#     hidden_size=81,
#     intermediate_size=81 * 2,
#     params_dtype=torch.float16,
#     layer_idx=0,
#     quant_config=None,
#     prefix="moe"
# )

# # print(list(a.named_parameters()))
# for name, param in a.named_parameters():
#     print(name, param.shape, param.dtype)

# expert_params_mapping = FusedMoE.make_expert_params_mapping(
#                     ckpt_gate_proj_name="gate_proj",
#                     ckpt_down_proj_name="down_proj",
#                     ckpt_up_proj_name="up_proj",
#                     num_experts=4)   
# print(expert_params_mapping)

import torch

# # 只读取权重名，不加载 tensor
# def get_weight_keys(bin_path):
#     with open(bin_path, "rb") as f:
#         # 只读取元信息（注意这个方式也会花几秒时间，但不会真正分配大内存）
#         checkpoint = torch.load(f, map_location="meta")
#         if "state_dict" in checkpoint:  # 某些模型保存时嵌套了一层
#             checkpoint = checkpoint["state_dict"]
#         for k in checkpoint.keys():
#             print(k)

# # 示例用法
# get_weight_keys("/home/zhongky/XLargeM/yupeng/checkpoint/hybrid/Hybrid-MoE-TP1-PP4-EP8-NUM_DENSE7-64k-HF/pytorch_model.bin")


# state_dict = load_state_dict("/home/zhongky/XLargeM/yupeng/checkpoint/hybrid/Hybrid-MoE-TP1-PP4-EP8-NUM_DENSE7-64k-HF/pytorch_model.bin", is_safetensors=False)  # HuggingFace 有特殊 loader
state_dict = torch.load('/home/zhongky/XLargeM/yupeng/checkpoint/hybrid/Hybrid-MoE-TP1-PP4-EP8-NUM_DENSE7-64k-HF/pytorch_model.bin', map_location='cpu')
for k in state_dict.keys():
    print(k)