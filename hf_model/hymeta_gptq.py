
#### This code should be placed at /usr/local/lib/python3.12/dist-packages/gptqmodel/models/definitions/hymeta.py
#### refer to https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/models/definitions/deepseek_v2.py

from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel

class HymetaGPTQ(BaseGPTQModel):
    # Strict=True -> all layer_modules must exists in model
    # Some models (deepseek2-lite) dynamically create lora modules based on config.rank
    layer_modules_strict = False # Cause some of our layers are dense, some are sparse

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_local_experts contains the actual expert count used for index
    dynamic_expert_index = "num_local_experts"

    base_modules = ["model.embeddings", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = "HybridBlock"
    layer_modules = [
        ["attn.linear_attn.q_proj", "attn.linear_attn.k_proj", "attn.linear_attn.v_proj"],
        ["attn.vanilla_attn.q_proj", "attn.vanilla_attn.k_proj", "attn.vanilla_attn.v_proj"],
        ["attn.out_proj"],

        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],

        # uses dynamic_expert_index
        [f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj", f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj"],
        [f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],

        ["shared_mlp.down_proj"], ["shared_mlp.gate_proj"], ["shared_mlp.up_proj"]
    ]