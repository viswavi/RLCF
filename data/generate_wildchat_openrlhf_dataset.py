import argparse
import copy
import csv
import json
import jsonlines
import numpy as np
import random
import os
import pickle
import signal
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import datasets

parser = argparse.ArgumentParser()

parser.add_argument("--wildchat-rewards", type=str, default="combined_wildchat_scores.json")
parser.add_argument("--skyworks-rewards", type=str, default=None)
parser.add_argument("--code-requirement-path", type=str, default=None)
parser.add_argument("--dataset-type", type=str, choices=["rl", "cot_sft", "sft"])
parser.add_argument("--output-file", type=str, default=None, required=True,)
parser.add_argument("--relabel-rewards", action="store_true")
parser.add_argument("--reward-threshold", type=float, default=None)
parser.add_argument("--constraint-threshold", type=float, default=None)
parser.add_argument("--pct-to-keep", type=float, default=None)
parser.add_argument("--filter-by", type=str, choices=["reward", "constraint"], default=None)
parser.add_argument("--use-ai-judge", action="store_true")
parser.add_argument("--dataset-name-hf", type=str, default=None)
parser.add_argument("--texture", action="store_true")
parser.add_argument("--code-only", action="store_true")
parser.add_argument("--using-armorm", action="store_true")
parser.add_argument("--do-not-scale-score", action="store_true")
parser.add_argument("--dont_average", action="store_true")


""""
Start new block
"""

import importlib
import inspect
import pkgutil
import re
import sys
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr

def restricted_import(name, *args, **kwargs):
    """A restricted __import__ function that only allows standard library imports"""
    # Checking if it's a standard library module
    if name == "subprocess":
        raise ImportError(f"Subprocess is considered dangerous and is not allowed")

    if name in sys.builtin_module_names or name in sys.stdlib_module_names:
        return importlib.import_module(name)
    # For Python versions that don't have sys.stdlib_module_names
    try:
        module = importlib.import_module(name)
        # Check if the module's file path is within the standard library directories
        if hasattr(module, '__file__') and module.__file__:
            for path in sys.path:
                if 'site-packages' not in path and module.__file__.startswith(path):
                    return module
    except (ImportError, AttributeError):
        pass
    raise ImportError(f"Restricted import: {name} is not a standard library module")

def _getiter_(obj):
    """Return an iterator for the given object."""
    return iter(obj)


def _getiter_(obj):
    """Return an iterator for the given object."""
    return iter(obj)

def _getitem_(obj, index):
    """Get an item from the object."""
    return obj[index]

def _inplacevar_(op, val1, val2):
    """Handle in-place operations like +=, -=, etc. for simple variables"""
    if op == '+=':
        return val1 + val2
    elif op == '-=':
        return val1 - val2
    elif op == '*=':
        return val1 * val2
    elif op == '/=':
        return val1 / val2
    elif op == '%=':
        return val1 % val2
    elif op == '**=':
        return val1 ** val2
    elif op == '//=':
        return val1 // val2
    elif op == '<<=':
        return val1 << val2
    elif op == '>>=':
        return val1 >> val2
    elif op == '&=':
        return val1 & val2
    elif op == '^=':
        return val1 ^ val2
    elif op == '|=':
        return val1 | val2
    else:
        raise NotImplementedError(f"Unsupported inplace operation: {op}")

safe_builtins_to_add = { 'all': all, 'any': any, 'ascii': ascii, 'bin': bin, 'bool': bool, 'bytearray': bytearray, 'bytes': bytes, 'callable': callable, 'chr': chr, 'complex': complex, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate, 'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset, 'hex': hex, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct, 'ord': ord, 'pow': pow, 'range': range, 'repr': repr, 'reversed': reversed, 'round': round, 'set': set, 'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip, 'format': format, 'dict': dict, 'list': list, 'set': set, 'tuple': tuple, 'abs': abs, 'dir': dir, 'getattr': getattr, 'hasattr': hasattr, 'id': id, 'vars': vars, }

restricted_globals = dict(safe_globals)
restricted_globals['__builtins__'] = {
    **safe_globals['__builtins__'],
    '__import__': restricted_import,
    **safe_builtins_to_add}
restricted_globals['_getitem_'] = _getitem_
restricted_globals['_getiter_'] = _getiter_
restricted_globals['_inplacevar_'] = _inplacevar_
restricted_globals['_iter_unpack_sequence_'] = guarded_iter_unpack_sequence
restricted_globals['getattr'] = safer_getattr
restricted_globals['__name__'] = 'restricted_module'
restricted_globals['json'] = json
restricted_globals['re'] = re

def timeout_handler(signum, frame):
    """Handler called when timeout occurs"""
    raise TimeoutError("Code execution timed out")

def execute_with_timeout(function_to_execute, response, timeout=30):
    """
    Execute the given function with a timeout.
    Returns (result, timed_out) where result is the function's return value
    and timed_out is a boolean indicating if execution timed out.
    """
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set alarm for `timeout` seconds
    
    result = None
    timed_out = False
    
    try:
        # Execute the function
        result = function_to_execute(response)
        signal.alarm(0)  # Cancel the alarm if execution completes
    except TimeoutError:
        timed_out = True
    except Exception as e:
        result = {'_error': str(e)}
    finally:
        signal.alarm(0)  # Ensure the alarm is canceled
        
    return result, timed_out

def try_to_reannotate(code_str, response):
    if code_str.strip() != "NONE" and code_str.strip().startswith("python"):
        code = code_str.split("python")[1].strip()
        loc = {}
        try:
            byte_code = compile_restricted(code, '<inline>', 'exec')
            try:
                exec(byte_code, restricted_globals, loc)
                function_to_execute = loc[list(loc.keys())[-1]]
                if inspect.isfunction(function_to_execute):
                    sig = inspect.signature(function_to_execute)
                    required_params = sum(1 for param in sig.parameters.values() if param.default is param.empty and param.kind == param.POSITIONAL_OR_KEYWORD)
                    if required_params == 1:
                        try:
                            pass_or_fail, timeout = execute_with_timeout(function_to_execute, response, timeout=10)
                            if timeout:
                                return None
                            try:
                                if rescale_score:
                                    reannotated_score = 100 if pass_or_fail else 0
                                    breakpoint()
                                else:
                                    reannotated_score = 1.0 if pass_or_fail else 0
                                return reannotated_score
                            except Exception as e:
                                reannotated_score = None
                        except Exception as e:
                            reannotated_score = None
                    else:
                        reannotated_score = None
                else:
                    reannotated_score = None
            except Exception as e:
                reannotated_score = None
        except Exception as e:
            reannotated_score = None
    else:
        reannotated_score = None 
    return None

""""
End new block
"""


MIDJOURNEY_PROMPT = """As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and you will provide a detailed prompt for Midjourney AI to generate an image."""

def filter_out_row(row: dict) -> bool:
    conversation = row.get("conversation", [])
    if len(conversation) >= 1:
        content = conversation[0]["content"]
        if content.strip().startswith(MIDJOURNEY_PROMPT):
            return True
    return False

def load_wildchat():
    ds = datasets.load_dataset("allenai/WildChat-1M", split='train')
    two_turn_rows_english = []
    for r in ds:
        if r["language"].lower() != "english":
            continue
        elif len(r["conversation"]) == 2:
            two_turn_rows_english.append(r)
    del ds
    raw_rows = two_turn_rows_english

    instruction_to_ground_truth_map = {}
    for row in raw_rows:

        if filter_out_row(row):
            continue

        row["timestamp"] = row["timestamp"].isoformat()
        turns = []
        for turn in row["conversation"]:
            if turn.get("timestamp") is not None:
                turn["timestamp"] = turn["timestamp"].isoformat()
            turns.append(turn)
        row["conversation"] = turns

        assert len(row["conversation"]) == 2
        prompt = row["conversations"][0]["content"]
        output = row["conversations"][1]["content"]
        instruction_to_ground_truth_map[prompt] = output
    return instruction_to_ground_truth_map

def load_raw_dataset(path, use_skyworks_scores = False, skyworks_rewards_path=None, normalize_skyworks_scores=True, skyworks_score_floor=-40, skyworks_score_ceiling=23, difference_threshold=1.65, use_AI_judge=False):
    if use_skyworks_scores:
        skyworks_rewards = json.load(open(skyworks_rewards_path))
        normalized_skywork_scores = {}
        for prompt, row in skyworks_rewards.items():
            scorelist = {}
            for response, raw_scores in row["scores"].items():
                normalized_scorelist = []
                for score in raw_scores:
                    normalized_score = (score - skyworks_score_floor) / (skyworks_score_ceiling - skyworks_score_floor)
                    normalized_score = min(1, max(0, normalized_score))
                    normalized_scorelist.append(normalized_score)
                scorelist[response] = normalized_scorelist
            if len(scorelist) != 2:
                continue
            normalized_skywork_scores[prompt] = scorelist
    else:
        normalized_skywork_scores = None

    filtered_data = []
    for i, (prompt, row) in enumerate(json.load(open(path)).items()):
        if use_skyworks_scores:
            if prompt not in normalized_skywork_scores or len(normalized_skywork_scores[prompt]) != 2:
                continue
            try:
                score_difference = abs(normalized_skywork_scores[prompt][0] - normalized_skywork_scores[prompt][1])
            except:
                breakpoint()
            if score_difference < difference_threshold:
                continue
            row["scores"] = normalized_skywork_scores[prompt]
        else:
            try:
                if "scores" not in row:
                    continue
                scores_pair = row["scores"]
            except:
                breakpoint()
            trasformed_scores_pair = {}
            if len(scores_pair) < 2:
                continue
            score_list_a = list(scores_pair.values())[0]
            score_list_b = list(scores_pair.values())[1]

            if not use_AI_judge:
                requirement_weights = [x[1] for x in row["requirements"]]
                if len(requirement_weights) == 0 or min(requirement_weights) == 0:
                    continue
                if len(score_list_a) != len(requirement_weights) or len(score_list_b) != len(requirement_weights):
                    continue

                for response, scores in scores_pair.items():
                    transformed_scores = [score/weight for score, weight in zip(scores, requirement_weights)]
                    trasformed_scores_pair[response] = transformed_scores
                row["scores"] = trasformed_scores_pair
        row["prompt"] = prompt
        filtered_data.append(row)
    return filtered_data

def format_conversation(prompt, response):
    return [
        {
            "content": prompt,
            "role": "user"
        },
        {
            "content": response,
            "role": "assistant"
        }
    ]

def relabel_scores_with_verifiers(prompt, response, requirements, score_list, verifiers, code_only=False, code_cache={}, rescale_score=True, dont_average=False):
    num_corrections = 0
    if len(requirements) != len(score_list):
        return score_list
    for i, (req, score) in enumerate(zip(requirements, score_list)):
        if (prompt, req) in verifiers:
            code_str = verifiers[(prompt, req)]
            if code_str.strip() != "NONE" and code_str.strip().startswith("python"):
                code = code_str.split("python")[1].strip()
                # print(f"For prompt\n{prompt}\n, I'm trying to execute\n--\n{code}\n--for response --\n\n{response}\n-- and requirement --\n\n{req}\n--\n\n\n\n\n\n\n\n\n")

            if code_cache and (code_str, response) in code_cache:
                reannotated_score = code_cache[(code_str, response)]
            else:
                reannotated_score = try_to_reannotate(code_str, response)
                if code_cache is not None:
                    code_cache[(code_str, response)] = reannotated_score
            if rescale_score is not True and reannotated_score is not None:
                # In this case, we will not rescale the score later
                reannotated_score /= 100
            if reannotated_score is not None:
                if code_only:
                    score_list[i] = reannotated_score
                    num_corrections += 1
                else:
                    if dont_average:
                        score_list[i] = reannotated_score
                    else:
                        score_list[i] = (score_list[i] + reannotated_score)/2
                    num_corrections += 1
            else:
                if code_only:
                    score_list[i] = -1
        else:
            if code_only:
                score_list[i] = -1
    return score_list, num_corrections

def convert_to_openrlhf_format(raw_dataset, verifiers=None, use_skyworks_scores=False, min_score_diff=0.001, texture=False, constraint_threshold=None, reward_threshold=None, use_AI_judge=False, code_only=False, pct_to_keep=None, filter_by=None, code_cache={}, scale_score=True, dont_average=False):
    formatted_dataset = []
    total_number_of_corrections = 0
    total_number_of_requirements = 0
    max_differences = []
    reward_gaps = []
    for i, row in tqdm(enumerate(raw_dataset)):
        if row["prompt"].strip().startswith(MIDJOURNEY_PROMPT):
            continue
        if use_skyworks_scores:
            scorelist = []
            for resp, score_list in row["scores"].items():
                score_unclipped = sum(score_list)
                score = min(1, max(0, score_unclipped))
                scorelist.append((resp, score))
            requirements = None
        elif use_AI_judge:
            scorelist = []
            for resp, score_list in row["scores"].items():
                sum_score = score_list
                normalized_score_unclipped = sum(sum_score) / 5
                normalized_score = min(1, max(0, normalized_score_unclipped)) * 100
                scorelist.append((resp, normalized_score))
        else:
            req_scores = [tup[1] for tup in row["requirements"]]
            req_strings = [tup[0] for tup in row["requirements"]]
            scorelist = []
            for resp, score_list in row["scores"].items():
                orig_score_list = copy.deepcopy(score_list)
                if verifiers is not None:
                    score_list, num_corrections = relabel_scores_with_verifiers(row["prompt"], resp, req_strings, score_list, verifiers, code_only=code_only, code_cache=code_cache, rescale_score=scale_score, dont_average=dont_average)
                    total_number_of_corrections += num_corrections
                else:
                    assert not code_only, "Code-only mode is not supported if a verifier is provided."
                if code_only:
                    sum_score = [s * req_scores[i] for i, s in enumerate(score_list) if s != -1]
                    corresponding_req_scores = [req_scores[i] for i, s in enumerate(score_list) if s != -1]
                    if len(sum_score) == 0:
                        normalized_score_unclipped = 0
                    else:
                        normalized_score_unclipped = sum(sum_score) / sum(corresponding_req_scores)
                else:
                    sum_score = [s * req_scores[i] for i, s in enumerate(score_list)]
                    normalized_score_unclipped = sum(sum_score) / sum(req_scores)
                if scale_score:
                    normalized_score_unclipped /= 100

                normalized_score = float(min(1.0, max(0.0, normalized_score_unclipped)))
                scorelist.append((resp, normalized_score))

            requirements = []
            sorted_requirements = sorted(row["requirements"], key=lambda tup: tup[1], reverse=True)
            for i, (req, weight) in enumerate(sorted_requirements):
                formatted_req = f"{str(i+1)}) {req.strip()} (importance: {int(weight)}/100)"
                requirements.append(formatted_req)
                total_number_of_requirements += 1

        random.shuffle(scorelist)
        prompt = row["prompt"]
        sorted_responses = sorted(scorelist, key=lambda tup: tup[1], reverse=True)
        if texture:
            chosen_response = sorted_responses[0][0]
        else:
            chosen_response = format_conversation(prompt, sorted_responses[0][0])
        chosen_response_score = sorted_responses[0][1]
        if texture:
            rejected_response = sorted_responses[1][0]
        else:
            rejected_response = format_conversation(prompt, sorted_responses[1][0])
        rejected_response_score = sorted_responses[1][1]

        reward_gap = chosen_response_score - rejected_response_score
        reward_gaps.append(reward_gap)

        if reward_threshold is not None and abs(reward_gap) < reward_threshold:
            continue

        if "per_category_scores" in row:
            try:
                chosen_scorelist = row["per_category_scores"][sorted_responses[0][0]]
                rejected_scorelist = row["per_category_scores"][sorted_responses[1][0]]
            except:
                continue
        else:
            chosen_scorelist = row["scores"][sorted_responses[0][0]]
            rejected_scorelist = row["scores"][sorted_responses[1][0]]
        max_difference = [a - b for a, b in zip(chosen_scorelist, rejected_scorelist)]
        max_differences.append(max(max_difference))

        if constraint_threshold is not None and max(max_difference) < constraint_threshold:
            continue


        formatted_row = {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "chosen_score": np.float64(chosen_response_score),
            "rejected_score": np.float64(rejected_response_score)
        }
        if not use_skyworks_scores and not use_AI_judge:
            if texture:
                reqs_without_text = [r.split("(importance:")[0].strip() for r in requirements]
                reqs_without_text = [r.split(")")[1].strip() for r in reqs_without_text]
                importances_with_numbers = [int(r.split("(importance: ")[1].split("/100")[0].strip()) for r in requirements]
                formatted_row["requirements"] = reqs_without_text
                formatted_row["importances"] = importances_with_numbers
            else:
                formatted_row["requirements"] = "\n".join(requirements)
        formatted_dataset.append(formatted_row)

    

    if pct_to_keep is not None:
        print(f"len(max_differences): {len(max_differences)}")
        if filter_by == "reward":
            sorted_indices = sorted(range(len(reward_gaps)), key=lambda i: reward_gaps[i], reverse=True)
        elif filter_by == "constraint":
            sorted_indices = sorted(range(len(max_differences)), key=lambda i: max_differences[i], reverse=True)
        else:
            raise ValueError("Invalid filter_by value. Use 'reward' or 'constraint'.")
        top_to_keep = int(len(formatted_dataset) * pct_to_keep / 100)
        top_indices = sorted_indices[:top_to_keep]
        formatted_dataset = [formatted_dataset[i] for i in top_indices]

    return formatted_dataset

def convert_dataset_to_llamafactory_format(raw_dataset: list[dict], cot: bool) -> list[dict]:

    wildchat_responses_map = load_wildchat()

    processed_rows = []

    for row in tqdm(raw_dataset):

        if row["prompt"].strip().startswith(MIDJOURNEY_PROMPT):
            continue

        requirements_sorted = sorted(row["requirements"], key=lambda tup: tup[1], reverse=True)
        requirements_formatted = ""
        for i, (req, weight) in enumerate(requirements_sorted):
            weight_str = "importance: " + str(int(weight)) + "/100"
            requirement = f"{str(i+1)}) {req.strip()} ({weight_str})"
            requirements_formatted = requirements_formatted + requirement
            if i != len(requirements_sorted) - 1:
                requirements_formatted = requirements_formatted + "\n"

        ground_truth = wildchat_responses_map.get(row["prompt"], None)

        instruction = row["prompt"]
        if cot:
            output = f"""Let's think about the requirements that an ideal response must satisfy:
{requirements_formatted}

Now, let's produce a response that satisfies the requirements above.

Response:
{ground_truth}"""
        else:
            output = ground_truth

        instance = {
            "instruction": instruction,
            "input": "",
            "output": output
        }
        processed_rows.append(instance)
    return processed_rows


if __name__ == "__main__":
    args = parser.parse_args()
    
    raw_dataset = load_raw_dataset(args.wildchat_rewards, use_skyworks_scores=args.relabel_rewards, skyworks_rewards_path=args.skyworks_rewards, use_AI_judge=args.use_ai_judge)

    if args.code_requirement_path is not None:
        verifiers = {}
        for row in jsonlines.open(args.code_requirement_path):
            prompt, req, code = row
            if code.strip() != "NONE":
                verifiers[(prompt, req)] = code
        code_cache_file = "code_verification_cache.pkl"
        if os.path.exists(code_cache_file):
            code_cache = pickle.load(open(code_cache_file, 'rb'))
        else:
            code_cache = {}
    else:
        verifiers = None
        assert args.code_only == False, "Code-only mode is not supported if a verifier is provided."
        print("Not using code-based verification!")
        code_cache = None

    if args.dataset_type == "rl":
        dataset = convert_to_openrlhf_format(raw_dataset,
                                             verifiers,
                                             use_skyworks_scores=args.relabel_rewards,
                                             texture=args.texture,
                                             reward_threshold=args.reward_threshold,
                                             constraint_threshold=args.constraint_threshold,
                                             use_AI_judge=args.use_ai_judge,
                                             code_only=args.code_only,
                                             pct_to_keep=args.pct_to_keep,
                                             filter_by=args.filter_by,
                                             scale_score=not args.do_not_scale_score,
                                             code_cache=code_cache,
                                             dont_average=args.dont_average)
    elif args.dataset_type.endswith("sft"):
        cot = args.dataset_type.startswith("cot")
        dataset = convert_dataset_to_llamafactory_format(raw_dataset, cot)

    if args.output_file.endswith(".jsonl"):
        writer = jsonlines.open(args.output_file, "w")
        writer.write_all(dataset)
        writer.close()
    elif args.output_file.endswith(".json"):
        json.dump(dataset, open(args.output_file, 'w'), indent=4)
    elif args.output_file.endswith(".tsv"):
        header = ["Instruction", "Requirements", "Chosen Response", "Rejected Response", "Chosen Score", "Rejected Score"]
        rows = []
        for row in dataset:
            tsv_row = [
                row["prompt"],
                row["requirements"],
                row["chosen"][1]["content"],
                row["rejected"][1]["content"],
                round(row["chosen_score"], 3),
                round(row["rejected_score"], 3)
            ]
            rows.append(tsv_row)

        short_rows = [r for r in rows if len(r[0].split()) < 60 and len(r[2].split()) < 180 and len(r[3].split()) < 180]
        rows = [header] + random.sample(short_rows, 50)
        with open(args.output_file, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            for row in rows:
                writer.writerow(row)
    else:
        raise ValueError(f"Invalid output file format .{args.output_file.split('.')[-1]}")

    if args.code_requirement_path is not None and code_cache_file is not None:
        with open(code_cache_file, "wb") as f:
            pickle.dump(code_cache, f)

    if args.dataset_name_hf is not None:
        dataset_name = args.dataset_name_hf
        dataset.push_to_hub(dataset_name)
