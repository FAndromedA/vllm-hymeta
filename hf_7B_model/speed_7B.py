from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                        AutoTokenizer, AutoConfig)

from modeling.modeling_gla_swa import GLAswaForCausalLM
from modeling.configuration_gla_swa import GLAswaConfig

AutoConfig.register("gla_swa", GLAswaConfig)
AutoModelForCausalLM.register(GLAswaConfig, GLAswaForCausalLM)

import torch
# https://huggingface.co/docs/transformers/quantization/bitsandbytes?bnb=8-bit
# https://huggingface.co/docs/transformers/zh/main_classes/quantization#%E9%80%9A%E7%94%A8%E7%94%A8%E6%B3%95

# # This is for w4a16
# quantization_config = BitsAndBytesConfig(
#     load_in_4it=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model_8bit = AutoModelForCausalLM.from_pretrained(
#     "/root/zhuangjh/hymeta-7B/modeling", 
#     device_map="auto",
#     torch_dtype="auto",
#     quantization_config=quantization_config
# )


import argparse
import time
import numpy as np  

import torch
from datasets import load_dataset


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    parser.add_argument("--tk_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--no-cache", action='store_true')
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--topp", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark runs")  # 添加运行次数参数
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    print(f"Loading {args.tk_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tk_path,
        use_fast=False,
        add_eos_token=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    print(f"{tokenizer}")

    if args.model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={"": device},
            torch_dtype=dtype,
            use_cache=not args.no_cache,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            # quantization_config=quantization_config
        )
    else:
        print("args.model_path must be provided")
        exit(0)
    
    if args.compile:
        print("Compiling the model")
        model = torch.compile(model)
    
    model.eval()
    print(f"{model.config}\n{model}\nNumber of parameters: {model.num_parameters()} ({sizeof_fmt(model.num_parameters())})\n")

    if args.data is not None:
        print(f"Loading {args.data}")
        dataset = load_dataset(args.data, split='train', trust_remote_code=True)
        print(f"{dataset}")

        prompt = dataset[0]['text']
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)[:, :args.length].contiguous()
        max_length = input_ids.shape[1] + args.maxlen
    else:
        print(f"Creating benchmark data")
        tokens = tokenizer(["我"*args.length], return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)[:, :args.length].contiguous()
        max_length = input_ids.shape[1] + args.maxlen

    # 预热运行
    print("Running warm-up...")
    with torch.inference_mode():
        text = model.generate(
            input_ids=input_ids,
            use_cache=not args.no_cache,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.bos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Warm-up completed.")

    # 正式基准测试 - 运行多次并记录时间
    print(f"Starting benchmark with {args.num_runs} runs...")
    latencies = []
    
    for run_idx in range(args.num_runs):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start = time.time()
        with torch.inference_mode():
            text = model.generate(
                input_ids=input_ids,
                use_cache=not args.no_cache,
                max_new_tokens=args.maxlen,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.bos_token_id,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.topp,
                repetition_penalty=args.repetition_penalty
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        latencies.append(elapsed * 1000)  # 转换为毫秒
        
        # 打印单次运行结果
        generated_tokens = len(text[0]) - len(input_ids[0])
        print(f"Run {run_idx+1}/{args.num_runs}: "
              f"Time = {elapsed * 1000:.0f}ms, "
              f"Tokens/s = {generated_tokens / elapsed:.2f}")
    
    # 计算统计信息
    max_memory = sizeof_fmt(torch.cuda.max_memory_allocated())
    latencies_ms = np.array(latencies)
    avg_latency = np.mean(latencies_ms)
    min_latency = np.min(latencies_ms)
    max_latency = np.max(latencies_ms)
    std_latency = np.std(latencies_ms)
    
    # 计算吞吐量（基于最后一次运行的token数量）
    generated_tokens = len(text[0]) - len(input_ids[0])
    tokens_per_sec = generated_tokens / (avg_latency / 1000)
    
    # 打印最终结果
    print("\n===== Benchmark Results =====")
    print(f"Model path: {args.model_path}")
    print(f"Prompt length: {len(input_ids[0])}, Generation length: {generated_tokens}")
    print(f"Runs: {args.num_runs}, Temperature: {args.temperature}, Top-p: {args.topp}")
    print(f"Average latency: {avg_latency:.0f}ms ± {std_latency:.1f}ms")
    print(f"Min latency: {min_latency:.0f}ms, Max latency: {max_latency:.0f}ms")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/second")
    print(f"Max memory used: {max_memory}")

'''
python3 -m speed_7B \
    --model_path /root/zhuangjh/hymeta-7B/modeling \
    --tk_path /root/zhuangjh/hymeta-7B/modeling \
    --length 130000 \
    --maxlen 128 \
    --num_runs 10

8-bit
nohup python3 -m speed_7B \
    --model_path /root/zhuangjh/hymeta-7B-gptq \
    --tk_path /root/zhuangjh/hymeta-7B-gptq \
    --length 130000 \
    --maxlen 128 \
    --num_runs 10 > _vllm_int8_speed_32k_bench.log 2>&1 &
'''

"""
32768 66536 130000

bitsandbytes w4a16
quant:
32k  2349ms ± 17.2ms    Max memory used: 15.1GiB
64k  4691ms ± 10.5m     Max memory used: 24.6GiB
128k 9506ms ± 81.4ms    Max memory used: 43.3GiB

non-quant:
32k  2332ms ± 19.2ms    Max memory used: 23.8GiB
64k  4680ms ± 7.0ms     Max memory used: 33.6GiB
128k 10512ms ± 287.2ms  Max memory used: 52.0GiB

int8 gptqmodel
32k  2410ms ± 9.4ms     Max memory used: 18.1GiB
64k  4773ms ± 10.8ms    Max memory used: 27.9GiB
128k 10015ms ± 434.3ms  Max memory used: 46.3GiB


int8 llmcompressor
32k  5983ms ± 88.2ms
64k  11380ms ± 113.2ms
128k 20104ms ± 64.7ms

"""
