import argparse
import json
import os.path
import types
import tqdm
import torch
import copy
from copy import deepcopy
import dataclasses
from xopen import xopen
import math
import matplotlib.pyplot as plt 

from rouge import Rouge
import logging
import numpy as np
import pdb
from eval import (
    scorer
)
# from lost_in_the_middle.prompting import (
#     Document,
#     get_closedbook_qa_prompt,
#     get_qa_prompt,
#     get_qa_prompt_index
# )

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from utils_real_drop.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def plot_sample_num(input_h):
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(input_h, bins=20, color='blue', alpha=0.7, edgecolor='black')  # bins 控制直方图的分箱数量
    plt.title("Distribution of Sample Lengths")
    plt.xlabel("Sample Length")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    plt.savefig("/home/ubuntu/data/results/sample_length_distribution.png")


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": None,
    "llama_h2o": H2OLlamaForCausalLM
}

TAGET_MODULE = {
    "llama": None,
    "llama_h2o": H2OLlamaAttention
}

# 画图
def plot_all_scores(all_rouge_scores_list):
    x = range(1, len(all_rouge_scores_list[0]) + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(x, all_rouge_scores_list[0], label='ROUGE-1')
    plt.plot(x, all_rouge_scores_list[1], label='ROUGE-2')
    plt.plot(x, all_rouge_scores_list[2], label='ROUGE-L')

    plt.xlabel('Sample Index')
    plt.ylabel('ROUGE Score')
    plt.title('ROUGE Scores for Generated Texts')
    plt.legend()
    plt.savefig(f'./summary_results/xsum_0_h2o_rouge_scores_tmp.png')
    plt.show()

############### 
# copy from https://github.com/Zefan-Cai/KVCache-Factory/blob/main/run_longbench.py

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "xsum": 64, # 添加的，与h2o的对齐
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "xsum": "Please summarize the given text into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

# model2maxlen = {
#     "Llama-2-7b-chat-hf": 3950,
#     "Llama-3-8B-Instruct": 7950,
#     "Meta-Llama-3-70B-Instruct": 7950,
#     "Meta-Llama-3-8B-Instruct-32k": 31500,
#     "Llama-2-7B-32K-Instruct": 31500,
#     "Mistral-7B-Instruct-v0.2": 31500,
#     "Mistral-7B-Instruct-v0.1": 31500,

# }

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500
}

def build_chat(prompt):
    prompt = f"[INST] {prompt} [/INST]"
    return prompt

# long bench path gproject/h2o_hf/data/LongBench
# 记得引用
"""
@article{bai2023longbench,
title={LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding},
author={Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
journal={arXiv preprint arXiv:2308.14508},
year={2023}
}
"""

########################

# 拆分eval逻辑
@torch.no_grad()
def eval_task(model, tokenizer,args, task_type="xsum"):
    requests = []
    # 定义模型最大长度
    model_max_len = 2048 # default
    input_max_len = 0
    for key in model2maxlen:
        if key in args.model_name.lower():
            model_max_len = model2maxlen[key]
    output_max_len = dataset2maxlen[task_type]

    # 读取测试数据集
    # 数据的格式是json
    # 属性包括
    # input,context,answers,length,dataset,language,all_classes,_id
    # 或者是 request: {article,summary_gt,logprobs,max_tokens,model,n,prompt,request_type,stop,temperature,top_p,scenario,}
    prompts =[]
    len_prompt = []

    with open(args.data_file, 'r') as f:
        for line in f:
            if line.strip() != '':
                example = json.loads(line)
                if "article" in example:
                    prompt_list = example["article"].split("###")
                    example["input"] = prompt_list[-1]
                    example["context"] = " ".join(prompt_list[:-1])
                    # answers代表可能有多种答案，这里只1个
                    example["answers"] = [example["summary_gt"]]
                template = model2prompt[task_type]
                prompt = template.format(**example)
                len_prompt.append(len(prompt))
                # length = example["length"]
                # if length > input_max_len: input_max_len = length
                if "llama2" in args.model_name.lower():
                    prompt = build_chat(prompt)
                example["prompt"] = prompt
                requests.append(example)
                prompts.append(example["prompt"])
    plot_sample_num(len_prompt)

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples from {} samples'.format(args.sample_num, len(requests)))
    requests = requests[:args.sample_num]

    predictions, answers, lengths = [], [], []
    all_classes = None
    #  batch 处理
    for i in tqdm.tqdm(range(0, len(requests[:50]), args.batch_size)):
        batch_prompts = prompts[i:i+args.batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest",
                                    return_tensors="pt",
                                    add_special_tokens=True).to(model.device)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)

            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask
        context_length = batch_input_ids.shape[-1]
        output = model.generate(
                **tokenized_prompts,
                # output_attentions = True , # default False
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=True,
                top_k = args.k,
                top_p = 1,
                temperature=0.3,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id],
                return_dict_in_generate=True, output_scores=True,
            )
        if args.enable_h2o_cache:
            for name, m in model.named_modules():
                if isinstance(m, TAGET_MODULE['llama_h2o']):
                    m._clean_cache()
        # import pdb;pdb.set_trace()
        batch_outputs =tokenizer.batch_decode(output[0][:,context_length:].tolist(),
                                            skip_special_tokens=True)
        batch_generations = batch_outputs
        print(batch_generations)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for j in range(len(batch_generations)):
            id = i*args.batch_size+j
            answers.append(requests[id]['answers'])
            predictions.append(batch_generations[j])
            all_classes = requests[id].get('all_classes', None)
            if "length" in requests[id]:
                lengths.append(len(batch_generations[j]))
    # import pdb;pdb.set_trace()
    score = scorer(task_type, predictions, answers, all_classes)
    print(f"dataset {task_type} method {args.method} scores {score}")
    # 将结果写入文件
    result_path = "/home/ubuntu/data/results/rouge_scores/keep_first.txt"
    with open(result_path, "a") as f:  # 使用 "a" 模式以追加内容
        # 格式化结果字符串
        result_str = 'keep_first: {}, dataset: {}, method: {}, scores: {}\n'.format(
            args.keep_first,
            task_type,
            args.method,
            score
        )
        # 写入文件
        f.write(result_str)

    
# bash scripts/summarization/eval.sh xsum 3 h2o 5 0.1 0.1 0
# 使用方法参考
# bash scripts/summarization/eval.sh xsum 3 h2o 5 0.1 0.1 0
# bash scripts/summarization/eval.sh xsum 3 narrativeqa 5 0.1 0.1 0
 
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--method", type=str, default="h2o")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--hh_size", type=int, default=1024)
    parser.add_argument("--recent_size", type=int, default=1024)

    # 默认是full
    parser.add_argument("--hh_ratio", type=float, default=0)
    parser.add_argument("--recent_ratio", type=float, default=1)
    parser.add_argument("--window_ratio", type=float, default=0)
    parser.add_argument("--token_block_ratio", type=float, default=0)
    parser.add_argument("--keep_first", type=int, default=0)

    parser.add_argument('--enable_h2o_cache', action='store_true')

    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)

    # if args.batch_size>1:
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.enable_h2o_cache:
        print('Enabling H2O KV cache')
        # 指定具体使用什么方法
        config.method = args.method
        config.hh_size = args.hh_size
        config.recent_size = args.recent_size
        config.hh_ratio = args.hh_ratio
        config.recent_ratio = args.recent_ratio
        config.window_ratio = args.window_ratio
        config.token_block_ratio = args.token_block_ratio
        config.keep_first = args.keep_first
        model = ENABLE_Heavy_Hitter_FUNCTIONS['llama_h2o'].from_pretrained(model_name, config=config,
                                                                            cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
    if config.method == "lx":
        from patch_llama_model_forward import _patched_forward
        # 将 _patched_forward 绑定到 model.model 实例
        model.model.forward = types.MethodType(_patched_forward, model.model)


    model.half().eval().cuda()
    # 获取数据集
    if args.dataset=="xsum":
        args.data_file = args.input_path
    else:
        args.data_file = f"data/LongBench/{args.dataset}.jsonl"
    # args.method = "h2o"

    # 评估
    eval_task(model, tokenizer, args, task_type=args.dataset)

# 使用方法参考
# bash scripts/summarization/eval.sh xsum 3 h2o 5 0.1 0.1 0
# bash scripts/summarization/eval.sh xsum 3 h2o 5 0.1 0.1 0
# bash scripts/summarization/eval.sh triviaqa 3 h2o 5 0.1 0.1 0