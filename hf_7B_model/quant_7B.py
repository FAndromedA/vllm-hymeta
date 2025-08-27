import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from modeling.modeling_gla_swa import GLAswaForCausalLM
from modeling.configuration_gla_swa import GLAswaConfig

AutoConfig.register("gla_swa", GLAswaConfig)
AutoModelForCausalLM.register(GLAswaConfig, GLAswaForCausalLM)

MODEL_ID = '/root/zhuangjh/hymeta-7B/modeling'
# QUANT_PATH = '/root/zhuangjh/hymeta-7B-8bit'

############# gptqmodel

SAVE_DIR = '/root/zhuangjh/hymeta-7B-gptq-exclude-gk'

from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset, Dataset

calibration_dataset = load_dataset(
    "HuggingFaceH4/ultrachat_200k", split="train_sft"
).select(range(1024))["prompt"]

# calibration_dataset = load_dataset(
#     "m-a-p/Matrix", split="train", streaming=True
# )

# calibration_samples = calibration_dataset.take(1024)
# calibration_list = list(calibration_samples)
# calibration_text = Dataset.from_list(calibration_list)["text"]

quant_config = QuantizeConfig(bits=8, group_size=64)

model = GPTQModel.load(MODEL_ID, quant_config)
model.quantize(calibration_dataset, batch_size=32)
model.save(SAVE_DIR)

#### llmcompressor

# from llmcompressor.transformers import SparseAutoModelForCausalLM
# model = SparseAutoModelForCausalLM.from_pretrained(
#     MODEL_ID, 
#     device_map='auto', 
#     torch_dtype=torch.bfloat16, 
#     local_files_only=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

# from datasets import load_dataset

# NUM_CALIBRATION_SAMPLES = 512
# MAX_SEQUENCE_LENGTH = 2048

# ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
# ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# def preprocess(example):
#     return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
# ds = ds.map(preprocess)

# def tokenize(sample):
#     return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH,
#                      truncation=True, add_special_tokens=False)
# ds = ds.map(tokenize, remove_columns=ds.column_names)

# from llmcompressor.transformers import oneshot
# from llmcompressor.modifiers.quantization import GPTQModifier
# from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# recipe = [
#     # SmoothQuantModifier(smoothing_strength=0.8),
#     GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
# ]
# # post_process | WARNING - Optimized model is not saved. To save, please provide`output_dir` as input arg.Ex. `oneshot(..., output_dir=...)`
# oneshot(
#     model=model,
#     dataset=ds,
#     recipe=recipe,
#     max_seq_length=MAX_SEQUENCE_LENGTH,
#     num_calibration_samples=NUM_CALIBRATION_SAMPLES,
# )

# # SAVE_NAME = MODEL_ID.split("/")[-1] + "-W8A8-Dynamic-Per-Token-llmcompressor"
# # SAVE_DIR = "/root/zhuangjh/" + SAVE_NAME
# # print(SAVE_DIR)
# model.save_pretrained(QUANT_PATH, save_compressed=True)
# tokenizer.save_pretrained(QUANT_PATH)