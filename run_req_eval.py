import argparse
import numpy as np
import os
import random
import torch

from data.dataset_utils import (
    generation_for_infobench,
    generation_for_ifeval,
    generation_for_followbench,
    generation_for_wildchat,
    generation_for_rewardbench,
    generation_for_alpacaeval
)
from data.dataset_utils import evaluate_infobench, evaluate_ifeval, evaluate_followbench
from models.openai_models import OpenAIModel
from models.huggingface_models import HFModel
from models.vllm import vLLM_Model
from models.self_refine import SelfRefineModel

supported_models = ["Qwen/Qwen2.5-7B-Instruct",
                    "Qwen/Qwen2.5-7B",
                    "Qwen/Qwen2.5-32B-Instruct",
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen2.5-0.5B",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/Qwen2.5-1.5B",
                    "Qwen/Qwen2.5-3B-Instruct",
                    "Qwen/Qwen2.5-3B",
                    "neulab/gpt-4o-mini-2024-07-18",
                    "openai-community/gpt2-xl",
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "TIGER-Lab/MAmmoTH2-8B",
                    "TIGER-Lab/MAmmoTH2-8B-Plus"]

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=str, choices=["InFoBench", "IFEval", "FollowBench", "RewardBench", "WildChat", "AlpacaEval"], required=True, help="Specify which requirement-following benchmark to use")
parser.add_argument("--method", type=str, choices=["direct", "SelfRefine"], default="direct")
parser.add_argument("--llm", type=str, default="neulab/gpt-4o-mini-2024-07-18")
parser.add_argument("--truncate-response", action="store_true", help="Whether to truncate responses that include implicit reasoning")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top-p", type=float, default=0.9)
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--special-cache-infix", type=str, default=None)
parser.add_argument("--num-batches", type=int, default=1)
parser.add_argument("--seed", type=int, default=1041995)
parser.add_argument("--dataset-start-idx", type=int, default=None)
parser.add_argument("--dataset-end-idx", type=int, default=None)
parser.add_argument("--think-step-by-step", action="store_true", help="Whether or not to explicitly reason before producting the answer")
parser.add_argument("--use-skywork-reranker", action="store_true", help="Whether or not to use a skywork model as the reranker")
parser.add_argument("--ignore-chat-template", action="store_true")

def set_seed(seed):
    # set seed for all possible avenues of stochasticity
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = parser.parse_args()

    set_seed(args.seed)

    if args.method == "direct":
        if args.llm == "neulab/gpt-4o-mini-2024-07-18":
            model = OpenAIModel(name=args.llm)
        elif args.llm in supported_models or (args.llm.startswith("/") and os.path.exists(args.llm)):
                model = vLLM_Model(name=args.llm,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                special_cache_infix=args.special_cache_infix,
                                truncate_response=args.truncate_response,
                                n=args.num_samples,
                                chat=not args.ignore_chat_template)
        else:
            raise ValueError(f"LLM {args.llm} not supported for direct generation")
    elif args.method == "SelfRefine":
        if args.llm == "gpt-3.5-turbo" or args.llm == "gpt-4":
            llm = OpenAIModel(name=args.llm)
            model = SelfRefineModel(llm, args.llm)
        elif args.llm == "llama3_8b":
            llm = HFModel(name="meta-llama/Meta-Llama-3-8B")
            model = SelfRefineModel(llm, args.llm)
        elif args.llm == "llama3_8b_instruct":
            llm = HFModel(name="meta-llama/Meta-Llama-3-8B-Instruct")
            model = SelfRefineModel(llm, args.llm)

    if args.benchmark == "InFoBench":
        responses_by_category = generation_for_infobench(model=model)
        print("No evaluation!")
        metrics = evaluate_infobench(model, responses_by_category)
    elif args.benchmark == "IFEval":
        responses_by_category = generation_for_ifeval(model=model, think_step_by_step=args.think_step_by_step)
        metrics = evaluate_ifeval(model, responses_by_category)
    elif args.benchmark == "FollowBench":
        responses_by_category = generation_for_followbench(model=model)
        metrics = evaluate_followbench(model, responses_by_category)
    elif args.benchmark == "WildChat":
        responses_by_category = generation_for_wildchat(model=model,
                                                        special_cache_infix=args.special_cache_infix,
                                                        num_batches=args.num_batches,
                                                        dataset_start_idx=args.dataset_start_idx,
                                                        dataset_end_idx=args.dataset_end_idx,
                                                        think_step_by_step=args.think_step_by_step)
        print("No evaluation!")
    elif args.benchmark == "RewardBench":
        responses_by_category = generation_for_rewardbench(model=model)
        print("No evaluation!")
    elif args.benchmark == "AlpacaEval":
        responses_by_category = generation_for_alpacaeval(model=model)
        print("No evaluation!")
    else:
        raise ValueError(f"Benchmark {args.benchmark} not supported")
