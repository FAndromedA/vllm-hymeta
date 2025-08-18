from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TriviaQADataset, TriviaQAEvaluator
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

from mmengine.config import read_base

from opencompass.models import OpenAISDK, VLLM
from opencompass.models import VLLMwithChatTemplate
# from opencompass.tasks import OpenICLTask
from opencompass.models import HuggingFaceBaseModel

#vllm
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='Hymeta-70B-SFT-stage3',
        path='/root/zhuangjh/hymeta-70B-8K-SFT-reasoning',
        model_kwargs=dict(tensor_parallel_size=4, 
                        #   pipeline_parallel_size=4,  # opencompass not support pp
                          gpu_memory_utilization=0.67,
                          enable_expert_parallel=True,
                          max_model_len=8192,
                          block_size=256,
                          dtype='bfloat16',
                          enforce_eager=True,
                          trust_remote_code=True,
                          ),
        max_out_len=2048,
        max_seq_len=8192,
        batch_size=64,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
    )
]


triviaqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev',
    test_split='dev')

triviaqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Answer these questions, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                dict(role='BOT', prompt='A:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer ))

triviaqa_eval_cfg = dict(
    evaluator=dict(type=TriviaQAEvaluator), pred_role='BOT',
     pred_postprocessor=dict(
        type=extract_non_reasoning_content,
    ))

triviaqa_datasets = [
    dict(
        type=TriviaQADataset,
        abbr='triviaqa',
        path='opencompass/trivia_qa',
        reader_cfg=triviaqa_reader_cfg,
        infer_cfg=triviaqa_infer_cfg,
        eval_cfg=triviaqa_eval_cfg)
]

datasets = [*triviaqa_datasets,]
