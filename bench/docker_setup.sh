sudo docker run -itd \
 --entrypoint /bin/bash \
 --network host \
 --name hymeta-bench \
 --shm-size 160g \
 --gpus all \
 --privileged \
 -v /9950backfile/zhuangjh:/root/zhuangjh \
 --env "HF_ENDPOINT=https://hf-mirror.com" \
 docker.1ms.run/vllm/vllm-openai:v0.8.5