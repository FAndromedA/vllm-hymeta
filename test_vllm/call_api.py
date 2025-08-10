import torch
# from vllm import LLM, SamplingParams

from transformers import AutoTokenizer

# 加载模型（本地或远程）
# llm = LLM(model="Hymeta-70B")

# read input_ids from ./input_ids.pt

TOKENIZER_PATH = "/root/zhuangjh/hymeta-70B-8K"
FILE_PATH = "./input_ids.pt"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
input_ids_dict = torch.load(FILE_PATH)
input_ids = input_ids_dict['input_ids']

prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)

encode_text = tokenizer.encode(prompt, return_tensors='pt')
while encode_text.shape[1] < 8192:
    prompt += "<|image_pad|>"
    encode_text = tokenizer.encode(prompt, return_tensors='pt')

print(f"shape: {input_ids.shape}, encode shape: {encode_text.shape}, prompt length: {len(prompt)}")
# exit(0)
# # 设置 Sampling 参数
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=32)

# # 直接调用 generate
# outputs = llm.generate(prompt, sampling_params=sampling_params)

import requests

url = "http://localhost:8765/v1/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "Hymeta-70B",
    "prompt": prompt,
    "max_tokens": 128,
    "temperature": 0.8
}

# 输出结果
response = requests.post(url, headers=headers, json=data)
print(response.json())