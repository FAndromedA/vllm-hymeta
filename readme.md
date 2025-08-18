# VLLM-HYMETA

## 1. 介绍
***vllm-hymeta*** 是模型 hymeta 对 [vllm 推理框架](https://github.com/vllm-project/vllm/tree/main) 的适配 plugins，目前实现了 hymeta 在 nvidia 和 metaX GPU 平台上的 vllm 推理适配。

vllm plugins 机制提供了一种灵活、模块化的方法来集成后端，使用 plugins 的好处有：
- 解耦代码库：硬件后端插件代码保持独立，使 vLLM 核心代码更清晰。
- 减少维护负担：vLLM 开发人员可以专注于通用功能，而不会被后端特定实现所造成的差异所困扰。
- 更快的集成和更独立：新的后端可以快速集成，减少工作量并独立发展。

---
## 2. 安装

### 2.1 容器部署

对于 Nvidia 平台：
```
sudo docker run -itd \
	--entrypoint /bin/bash \
	--network host \
	--name hymeta-bench \
	--shm-size 160g \
	--gpus all \
	--privileged \
	-v /host_path:/container_path \
	--env "HF_ENDPOINT=https://hf-mirror.com" \
	docker.1ms.run/vllm/vllm-openai:v0.10.0
```

对于 MetaX 平台：
```
docker run \
	-itd --device=/dev/dri \
	--device=/dev/mxcd \
	--device=/dev/infiniband \
	--group-add video \
    --entrypoint /bin/bash \
    --network host \
    --name "${CONTAINER_NAME}" \
    --security-opt seccomp=unconfined \
	--security-opt apparmor=unconfined \
    --shm-size 512g \
    --ulimit memlock=-1 \
    --ulimit nofile=65536:65536 \
    --privileged \
	-v /host_path:/container_path \
    -e "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" \
	-e "GLOO_SOCKET_IFNAME=inbond1" \
	-e "MCCL_IB_HCA=bnxt_re1,bnxt_re2" \
	-e "TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1" \
	-e "TRITON_DISABLE_MACA_OPT_MMA_PREFETCH=1" \
	-e "TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1" \
	-e "TRITON_ENABLE_MACA_COMPILER_INT8_OPT=True" \
    "${ADDITIONAL_ARGS[@]}" \
    cr.metax-tech.com/public-ai-release/maca/vllm:maca.ai2.33.0.13-torch2.6-py310-ubuntu22.04-amd64 \
    -c "export PATH=$PATH:/opt/conda/bin:/opt/conda/condabin && ${RAY_START_CMD}"
```

### 2.2 安装 plugins
要使用 **vllm-hymeta** 使用以下命令进行安装：
```
git clone https://github.com/FAndromedA/vllm-hymeta.git
cd vllm-hymeta
pip install .
```

对于 Nvidia 平台，在 `nvidia` 分支下，推荐在以下环境下安装 **vllm-hymeta**：
```
decorator
pyyaml
scipy
setuptools
setuptools-scm
flash_attn==2.7.3
flash-linear-attention==0.1
vllm==0.10.0
torch==2.7.1
```

对于 MetaX 平台，在 `main` 分支下，推荐在以下环境下安装 **vllm-hymeta**：
```
decorator
pyyaml
scipy
setuptools
setuptools-scm
flash_attn==2.6.3
flash-linear-attention==0.1
vllm==0.8.5
torch==2.6.0
```

### 2.3 可能遇到的问题

- 如果需要使用 opencompass 进行评测，pyext0.7 安装时，由于AttributeError: module ‘inspect‘ has no attribute ‘getargspec‘. Did you mean: ‘getargs‘ 导致 opencompass 无法安装。

    解决方法，直接修改 /usr/lib/python3.12/inspect.py 创建一个 getargspec 函数并且在其内部调用 getfullargspec 函数。

- flash_attn 需要 block_size 为 256 的倍数，但是 vllm 只支持 1,8,16,32,64,128。

    解决方法：修改 `vllm/config.py` 的 BlockSize = Literal[1,8,16,32,64,128,256]

---
## 3. 使用

### 3.1 利用 vllm cli 部署 hymeta 模型：

在 Nvidia 平台上：
```
nohup vllm serve $YOUR_MODEL_PATH \
	--tensor-parallel-size 4 \
	--pipeline-parallel-size 2 \
	--enable-expert-parallel \
	--max-model-len 32k \
	--served-model-name Hymeta-70B \
	--gpu-memory-utilization 0.95 \
	--block-size 256 \
	--dtype bfloat16 \
	--port 8765 \
	--trust-remote-code \
	> _vllm_serve_hymeta_nohup.log 2>&1 &
```

在 metaX 平台上：
```
nohup vllm serve $YOUR_MODEL_PATH \
	--tensor-parallel-size 4 \
	--pipeline-parallel-size 2 \
	--enable-expert-parallel \
	--max-model-len 32k \
	--served-model-name Hymeta-70B \
	--gpu-memory-utilization 0.95 \
	--block-size 64 \
	--dtype bfloat16 \
	--port 8765 \
	--trust-remote-code \
	> _vllm_serve_hymeta_nohup.log 2>&1 &
```

请求部署的 hymeta：
```
curl http://localhost:8765/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Hymeta-70B",
    "prompt": "YOUR PROMPT HERE",
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

### 3.2 vllm benchmark

详细使用请参考：[vllm benchmarks readme](https://github.com/vllm-project/vllm/tree/933f45334a79dcb69aa93178b3bbf3d9e0d46f09/benchmarks)

例如吞吐量测试如下：
```
vllm bench throughput \
	--model /root/zhuangjh/hymeta-70B-8K \
    --dataset-name sonnet \
    --dataset-path /root/zhuangjh/vllm-hymeta/bench/sonnet.txt \
    --num-prompts 10000 \
    --gpu-memory-utilization 0.97 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 9k \
    --max-num-seqs 1000 \
    --block-size 256 \
    --dtype bfloat16 \
    --trust-remote-code \
    > _vllm_bf16_throughput_bench.log 2>&1 &
```

### 3.3 使用 opencompass 进行评测

在安装了 opencompass 之后，您也可以在 opencompass 文件夹下使用以下命令来对 hymeta 进行评测：

```
nohup python3 run.py /your_path_to_vllm-hymeta/bench/eval.py > logs/base_eval.log 2>&1 &
```