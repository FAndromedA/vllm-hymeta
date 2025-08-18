from mmengine.config import read_base

from opencompass.models import OpenAISDK, VLLM
from opencompass.models import VLLMwithChatTemplate
# from opencompass.tasks import OpenICLTask
from opencompass.models import HuggingFaceBaseModel

# meta_template = dict(
#     round=[
#         dict(role='HUMAN', api_role='HUMAN', begin='<im_start>user\n', end='<im_end>\n'),
#         dict(role='BOT', api_role='BOT', begin='<im_start>assistant\n', end='<im_end>\n', generate=True),
#     ],
#     begin='<im_start>system\nYou are a helpful assistant.<im_end>\n',
# )
#vllm
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='Llama3.1-8B-Chat',
        path='/root/zhuangjh/modelbase/Llama-3.1-8B-Instruct',
        model_kwargs=dict(tensor_parallel_size=1, 
                        #   pipeline_parallel_size=4,  # opencompass not support pp
                          gpu_memory_utilization=0.64,
                        #   enable_expert_parallel=True,
                          max_model_len=8192,
                          block_size=256,
                          dtype='bfloat16',
                          enforce_eager=True,
                          trust_remote_code=True,
                          ),
        max_seq_len=8192,
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        generation_kwargs=dict(temperature=0),
    ),
    
] 
#http://0.0.0.0:8765

# models = [
#     dict(
#         type=VLLM,
#         abbr='Hymeta-70B',
#         path='/root/docker_shared/Hybrid-MoE-TP1-PP4-EP8-NUM_DENSE7-HF',
#         engine_url='http://0.0.0.0:8765',
#         tokenizer_path='/root/docker_shared/Hybrid-MoE-TP1-PP4-EP8-NUM_DENSE7-HF',
#         max_seq_len=8192,
#         batch_size=8,
#         generation_kwargs=dict(temperature=0),
#     )
# ]

# models = [
#     dict(
#         abbr='Hymeta-70B',
#         type=OpenAISDK,
#         key='EMPTY',
#         openai_api_base='http://localhost:8765/v1/completions',
#         path='Hymeta-70B',
#         tokenizer_path='/root/docker_shared/Hybrid-MoE-TP1-PP4-EP8-NUM_DENSE7-HF',
#         rpm_verbose=True, # 是否打印请求速率
#         # meta_template=api_meta_template, # 服务请求模板
#         query_per_second=1, # 服务请求速率
#         max_out_len=1024, # 最大输出长度
#         max_seq_len=8192, # 最大输入长度
#         temperature=0.0, # 生成温度
#         batch_size=8, # 批处理大小
#         retry=3, # 重试次数
#     )
# ]

### 第二步：定义tasks
with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    # from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_ppl import cmmlu_datasets
    # from opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
    # from opencompass.configs.datasets.ARC_c.ARC_c_ppl import ARC_c_datasets
    # from opencompass.configs.datasets.ARC_c.ARC_c_clean_ppl import ARC_c_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets
    # from opencompass.configs.datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets

    from opencompass.configs.summarizers.example import summarizer
    # from opencompass.configs.datasets.ceval.ceval_ppl import ceval_datasets
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from opencompass.configs.datasets.nq.nq_gen import nq_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen import triviaqa_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.QuALITY.QuALITY_gen import QuALITY_datasets

datasets = [*ifeval_datasets,]

# 评测任务配置
# tasks = [
#     dict(
#         type='OpenICLTask',
#         models=models,
#         datasets=datasets,
#         output_dir='./outputs',
#     )
# ]