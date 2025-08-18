
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from typing import Optional, Union, List

logger = logging.get_logger(__name__)


class HymetaConfig(PretrainedConfig):

    model_type = "hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        num_hidden_layers=28,
        attn_mode="chunk",
        num_attention_heads=28,
        num_key_value_heads=4,
        use_short_conv=False,
        conv_size=4,
        use_lower_bound=True,
        num_meta_tokens=128,
        intermediate_size=18944,
        hidden_act="swish",
        max_position_embeddings=4096 * 32,
        sliding_window=4096,
        elementwise_affine=True,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        attention_dropout=0.0,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151643,
        tie_word_embeddings=False,
        initializer_range=0.02,
        fuse_cross_entropy=True,
        num_local_experts=16,
        num_experts_per_topk=1,
        router_jitter_noise = 0.0,
        shared_intermediate_size =18944,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_lower_bound = use_lower_bound
        self.num_meta_tokens = num_meta_tokens
        self.intermediate_size = intermediate_size
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.fuse_cross_entropy = fuse_cross_entropy
        self.full_attn_layers = [3, 10, 17, 24]
        # dense MLP 层的索引
        self.dense_mlp_layers = [0, 1, 2, 4, 6, 8, 10]

        self.interleaved_sliding_window = [
            0 if idx in self.full_attn_layers else sliding_window
            for idx in range(num_hidden_layers)
        ] # 在 meta_attention.py 中 0-1=-1 表示全局注意力

        # File "/opt/conda/lib/python3.10/site-packages/vllm/config.py", line 1162, in get_num_layers_by_block_type
        # [rank0]:     raise ValueError(
        # [rank0]: ValueError: The model is an hybrid without alayers_block_type or an attn_type_list in the hf_config,cannot determine the num of attention layers
        # 1 表示这一层有 softmax attention, 方便 vllm 管理 kv cache(因为我们是层内混合,所以都是1)
        self.attn_type_list = [1 for _ in range(num_hidden_layers)]
        
        self.num_local_experts = num_local_experts
        self.num_layer_experts = [
            num_local_experts if idx not in self.dense_mlp_layers else 1
            for idx in range(num_hidden_layers)
        ]
        self.num_experts_per_topk = num_experts_per_topk
        self.router_jitter_noise = router_jitter_noise
        self.shared_intermediate_size=shared_intermediate_size
        
        for key, value in kwargs.items():
            # print(key, value)
            # quantization_config
            if key == "quantization_config":
                self.quantization_config = dict(value)
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )