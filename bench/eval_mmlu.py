from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

from opencompass.models import OpenAISDK, VLLM
from opencompass.models import VLLMwithChatTemplate
# from opencompass.tasks import OpenICLTask
from opencompass.models import HuggingFaceBaseModel
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN', begin='<im_start>user\n', end='<im_end>\n'),
        dict(role='BOT', api_role='BOT', begin='<im_start>assistant\n', end='<im_end>\n', generate=True),
    ],
    begin='<im_start>system\nYou are a helpful assistant.<im_end>\n',
)
#vllm
models = [
    dict(
        type=VLLM,
        abbr='Hymeta-70B-SFT-stage3',
        path='/root/zhuangjh/hymeta-70B-8K-SFT-reasoning',
        model_kwargs=dict(tensor_parallel_size=4, 
                        #   pipeline_parallel_size=4,  # opencompass not support pp
                          gpu_memory_utilization=0.57,
                          enable_expert_parallel=True,
                          max_model_len=8192,
                          block_size=256,
                          dtype='bfloat16',
                          enforce_eager=True,
                          trust_remote_code=True,
                          ),
        meta_template=meta_template,
        max_out_len=2048,
        max_seq_len=8192,
        batch_size=2,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
    )
]

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    # from opencompass.configs.datasets.cmmlu.cmmlu_ppl import cmmlu_datasets
datasets = [*mmlu_datasets]