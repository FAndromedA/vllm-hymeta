from vllm import ModelRegistry
from transformers import AutoConfig, AutoModel

def register_model():
    from .modeling_hymeta import HymetaForCausalLM
    from .configuration_hymeta import HymetaConfig

    AutoConfig.register("hybrid", HymetaConfig)

    ModelRegistry.register_model(
        "HymetaForCausalLM",
        "vllm_hymeta.models.modeling_hymeta:HymetaForCausalLM",
    )