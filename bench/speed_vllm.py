# -*- coding: utf-8 -*-
# 改编用于 vLLM 基准测试

import argparse
import time
import numpy as np
import torch

# 导入必要的 vLLM 类
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM 生成（Generation）基准测试")
    # vLLM 会自动处理分词器，所以我们只需要模型路径。
    parser.add_argument("--model_path", type=str, default=None, help="模型文件夹的路径。")
    parser.add_argument("--length", type=int, default=128, help="输入提示（prompt）的长度。")
    parser.add_argument("--maxlen", type=int, default=128, help="要生成的最大新词元（token）数量。")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--topp", type=float, default=0.2, help="即 top_p。")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚系数。")
    parser.add_argument("--num_runs", type=int, default=10, help="基准测试的运行次数。")
    # 添加 vLLM 特有的参数以进行更精细的控制
    #parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="用于张量并行的 GPU 数量。")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="用于模型执行器的 GPU 显存使用比例。")
    args = parser.parse_args()

    # --- 1. 创建提示（Prompt） ---
    # vLLM 接受原始字符串作为输入，无需预先手动进行分词。
    print(f"正在创建基准测试数据，提示长度为 {args.length}")
    prompt = "我" * args.length
    # 我们将一个提示列表传递给 generate 方法。
    prompts = [prompt]

    # --- 2. 加载模型并定义采样参数 ---
    #print(f"正在加载 {args.model_path}，张量并行（TP）大小 = {args.tensor_parallel_size}")

    from vllm_hymeta.models import register_model
    
    register_model()

    llm = LLM(
        model="/root/zhuangjh/hymeta-70B-8K",
        tensor_parallel_size=4,
        # pipeline_parallel_size=1,
        # enforce_eager=True,
        enable_expert_parallel=True,
        trust_remote_code=True,
        block_size=256,
        dtype='bfloat16',
        max_model_len=args.length+200,
        max_num_seqs=3,
        gpu_memory_utilization=0.65,
    )
    # llm = LLM(
    #     model="/root/zhuangjh/hymeta-70B-8K-W8A8-Dynamic-Per-Token-llmcompressor",
    #     tensor_parallel_size=4,
    #     # pipeline_parallel_size=1,
    #     # enforce_eager=True,
    #     enable_expert_parallel=True,
    #     trust_remote_code=True,
    #     block_size=256,
    #     dtype='auto',
    #     max_model_len=args.length+200,
    #     max_num_seqs=3,
    #     gpu_memory_utilization=0.65,
    # #    quantization="awq",
    # )

    # llm = LLM(
    #     model="/root/zhuangjh/hymeta-7B-8bit",
    #     # pipeline_parallel_size=1,
    #     # enforce_eager=True,
    #     trust_remote_code=True,
    #     block_size=64,
    #     dtype='auto',
    #     max_model_len=args.length+200,
    #     max_num_seqs=3,
    #     gpu_memory_utilization=0.65,
    # )

    # 采样参数在一个专门的对象中进行配置。
    sampling_params = SamplingParams(
        n=1, # 每个提示返回的输出序列数量。
        temperature=args.temperature,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.maxlen, # 对应于 huggingface 的 max_new_tokens
        ignore_eos=True # 设置重复的“我”，会触发vllm的自动结束标识
    )

    #max_tokens=args.maxlen

    print(f"模型加载完成。采样参数: {sampling_params}")

    # --- 3. 预热运行 ---
    # vLLM 的首次运行会编译 CUDA 内核，因此预热非常重要。
    print("正在进行预热...")
    for _ in range(10):
        llm.generate(prompts, sampling_params, use_tqdm=False)
    print("预热完成。")

    # --- 4. 正式基准测试 ---
    print(f"开始基准测试，共 {args.num_runs} 轮...")
    latencies = []
    
    for run_idx in range(args.num_runs):
        start_time = time.time()

        # generate 方法是 vLLM 功能的核心。
        # 它在内部处理批处理（batching）和生成。
        outputs = llm.generate(prompts, sampling_params)

        torch.cuda.synchronize()

        end_time = time.time()
        elapsed = end_time - start_time
        latencies.append(elapsed * 1000) # 转换为毫秒

        # 从输出对象中提取统计信息
        prompt_len = len(outputs[0].prompt_token_ids)
        generated_tokens = len(outputs[0].outputs[0].token_ids)

        print(f"第 {run_idx+1}/{args.num_runs} 轮: "
              f"耗时 = {elapsed * 1000:.0f}ms, "
              f"速度 = {generated_tokens / elapsed:.2f} tokens/s")

    # --- 5. 计算并打印最终结果 ---
    latencies_ms = np.array(latencies)
    avg_latency = np.mean(latencies_ms)
    min_latency = np.min(latencies_ms)
    max_latency = np.max(latencies_ms)
    std_latency = np.std(latencies_ms)

    # 使用最后一轮的词元数量进行报告
    prompt_len = len(outputs[0].prompt_token_ids)
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    
    # 基于平均延迟计算吞吐量
    tokens_per_sec = generated_tokens / (avg_latency / 1000)

    print("\n===== vLLM 基准测试结果 =====")
    print(f"模型路径: {args.model_path}")
    #print(f"张量并行大小 (TP): {args.tensor_parallel_size}")
    print(f"提示长度: {prompt_len}, 生成长度: {generated_tokens}")
    print(f"运行次数: {args.num_runs}, 温度: {args.temperature}, Top-p: {args.topp}")
    print(f"平均延迟: {avg_latency:.0f}ms ± {std_latency:.1f}ms")
    print(f"最小延迟: {min_latency:.0f}ms, 最大延迟: {max_latency:.0f}ms")
    print(f"吞吐量: {tokens_per_sec:.2f} tokens/second")

"""
python3 speed_vllm.py \
    --length 32768 \
    --maxlen 128 \
    --num_runs 10 

nohup python3 speed_vllm.py     --length 65536     --maxlen 128     --num_runs 10 > _speed_test_64k.log 2>&1 &

VLLM_ATTENTION_BACKEND="XFORMERS" nohup python3 speed_vllm.py     --length 130000     --maxlen 128     --num_runs 10 > _speed_7B_test_128k.log 2>&1 &

"""
# 32768 65536 130000 130871 (131072-200-1)
# non-quant
# 32k  6949ms ± 31.7ms
# 64k  10228ms ± 9.9ms
# 130k 19726ms ± 16.7ms

# qaunt 8bit
# 32k  6262ms ± 46.3ms
# 64k  9909ms ± 9.9ms
# 130k 17225ms ± 9.1ms