import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from . import HymetaForCausalLM, HymetaConfig

MODEL_ID = "/root/zhuangjh/hymeta-70B-8K"

AutoConfig.register("hybrid", HymetaConfig)
AutoModelForCausalLM.register(HymetaConfig, HymetaForCausalLM)

# config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
# with init_empty_weights():
#     model = SparseAutoModelForCausalLM.from_config(config)

# max_memory = {i: "60GiB" for i in range(8)}
offload_folder = "/nvme/offload"  # 尽量用 NVMe 路径

# model = load_checkpoint_and_dispatch(
#     model, checkpoint=MODEL_ID, 
#     max_memory=max_memory,
#     device_map="auto", dtype="bfloat16",
#     offload_folder=offload_folder,
#     offload_state_dict=True,
# )

SAVE_NAME = MODEL_ID.split("/")[-1] + "-W8A8-Dynamic-Per-Token"
SAVE_DIR = "/root/zhuangjh/" + SAVE_NAME

############# gptqmodel

# SAVE_DIR += '-GPTQ'

# from gptqmodel import GPTQModel, QuantizeConfig
# from datasets import load_dataset, Dataset

# calibration_dataset = load_dataset(
#     "HuggingFaceH4/ultrachat_200k", split="train_sft"
# ).select(range(2048))["prompt"]

# # calibration_dataset = load_dataset(
# #     "m-a-p/Matrix", split="train", streaming=True
# # )

# # calibration_samples = calibration_dataset.take(1024)
# # calibration_list = list(calibration_samples)
# # calibration_text = Dataset.from_list(calibration_list)["text"]

# quant_config = QuantizeConfig(bits=8, group_size=64)

# model = GPTQModel.load(MODEL_ID, quant_config)
# model.quantize(calibration_dataset, batch_size=32)
# model.save(SAVE_DIR)
############# torchao

# from transformers import TorchAoConfig
# from torchao.quantization import Int8WeightOnlyConfig

# quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
# quantized_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config)
# quantized_model.save_pretrained(SAVE_DIR, max_shard_size="5GB", safe_serialization=False)

############# modelopt
# import modelopt.torch.quantization as mtq
# config = mtq.FP8_DEFAULT_CFG
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, 
#     device_map='cpu', 
#     torch_dtype='auto', 
#     local_files_only=True,
# )

# # Define a forward loop function for calibration
# def forward_loop(model):
#     pass

# # PTQ with in-place replacement of quantized modules
# model = mtq.quantize(model, config, forward_loop)

# from modelopt.torch.export import export_hf_checkpoint

# with torch.inference_mode():
#     export_hf_checkpoint(
#         model,  # The quantized model.
#         SAVE_DIR,  # The directory where the exported files will be stored.
#     )

############# llmcompressor
from llmcompressor.transformers import SparseAutoModelForCausalLM
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map='auto', 
    torch_dtype=torch.bfloat16, 
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH,
                     truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

recipe = [
    # SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
# post_process | WARNING - Optimized model is not saved. To save, please provide`output_dir` as input arg.Ex. `oneshot(..., output_dir=...)`
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

SAVE_NAME = MODEL_ID.split("/")[-1] + "-W8A8-Dynamic-Per-Token-llmcompressor"
SAVE_DIR = "/root/zhuangjh/" + SAVE_NAME
# print(SAVE_DIR)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# accelerate launch --gpu_ids=4,5,6,7 -m hf_model.quant_w8a8