import argparse
import json
import litellm
import numpy as np
import os
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default="~/checklist_finetuning/data/requirement_generation/qwen_72b_with_expert/")
parser.add_argument("--output-file", type=str, default="~/checklist_finetuning/data/requirement_generation/qwen_72b_with_expert/")
parser.add_argument("--infobench-file", type=str, default="~/checklist_finetuning/data/requirement_generation/infobench_requirements.json")

home_dir = os.path.expanduser("~")
cache_file=f"{home_dir}/Downloads/gpt4o_eval_cache.json"
if os.path.exists(cache_file):
    cache = json.load(open(cache_file))
else:
    cache = {}
cache = json.load(open(cache_file))

def remove_low_weight(requirements, thresh = 0.0):
    return {k: [r[0] for r in v if r[0].strip().lower() != "none" and r[1] >= thresh] for k, v in requirements.items()}


def query_llm(prompt, max_tokens=10, cache_file=f"{home_dir}/Downloads/gpt4o_eval_cache.json"):
    if prompt in cache:
        return cache[prompt]
    completion = litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url="https://cmu.litellm.ai",
            model="openai/neulab/gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=max_tokens,
    )
    completion_str = completion.choices[0].message.content
    cache[prompt] = completion_str
    json.dump(cache, open(cache_file, 'w'))
    return completion_str

class RequirementEvaluation:
    def __init__(self, entailment_labels=[], objectivity_labels=[], disentanglement="", atomic="", comprehensive=""):
        self.entailment_labels = entailment_labels
        self.objectivity_labels = objectivity_labels
        self.disentanglement = disentanglement
        self.atomic = atomic
        self.comprehensive = comprehensive

def judge_basic_info(instruction, requirements, max_retries=3, use_in_context_examples=False):
    checklist_str = json.dumps(requirements, indent=4)
    if use_in_context_examples:
        in_context_examples = """

Example:
Instruction: "Write a touching Iambic poem about a cat and a dog who become friends."

Checklist:
[
    "Is the response a poem?",
    "Is the response a poem about a cat and a dog who become friends?",
    "Does the poem rhyme?",
    "Is the response touching?",
    "Is the response written in Iambic meter?",
    "Does the poem alternating unstressed and stressed syllables five times per line?",
    "Is the poem at least 10 lines long?",
]

For each requirement, write whether it entailed by the instruction. Respond with "yes" or "no":

Is the response a poem?:
yes
Is the response a poem about a cat and a dog who become friends?
yes
Does the poem rhyme?
no
Is the response touching?
yes
Is the response written in Iambic meter?
yes
Does the poem alternating unstressed and stressed syllables five times per line?
yes
Is the poem at least 10 lines long?
no

For each requirement, write whether it is objective and readily measurable by an LLM. Respond with "yes" or "no":

Is the response a poem?:
yes
Is the response a poem about a cat and a dog who become friends?
yes
Does the poem rhyme?
yes
Is the response touching?
no
Is the response written in Iambic meter?
yes
Does the poem alternating unstressed and stressed syllables five times per line?
yes
Is the poem at least 10 lines long?
yes


For each requirement, write whether it is objective and readily measurable by an LLM. Respond with "yes" or "no"
no

Does this checklist contain pairwise-disentangled and non-redundant requirements? Respond with "yes" or "no"
yes

Are the items within this checklist all atomic (conditioned on the other items in the checklist)? Respond with yes or "no"
yes

Is the checklist comprehensive? Respond with "yes" or "no"
yes"""
    else:
        in_context_examples = ""

    entailment_prompt = f"""You are an AI judge tasked with assessing the quality of requirements produced in response to a given instruction. Given an instruction, we wish to write our a set of criteria (posed as yes/no questions) in order to more objectively assess the quality of an instruction. These checklists will later be given to an AI language model to judge the quality of various responses; accordingly, a good checklist will be as objectively measurable as possible, detailed, and reasonable (in agreement with the original instruction). It will also contain criteria that are atomic (each introducing exactly one new aspect to measure relative to the others), disentangled (not redundant from each other), and comprehensive (collectively covering all key aspects of response quality). You will be given either a set of criteria or individual criteria, and tasked with assessing each of these qualities separately.{in_context_examples}

Instruction: "{instruction}"

Checklist:
{checklist_str}

For each requirement, write whether it entailed by the instruction. Respond with "yes" or "no":\n\n"""
    
    entailment_labels = []
    for req in requirements:
        entailment_prompt += req + "\n"
        
        valid_output = False
        retries = 0
        while not valid_output and retries < max_retries:
            answer = query_llm(entailment_prompt, max_tokens=3).strip().lower()
            if len(answer.split()) > 1:
                answer = answer.split()[0]
            if answer.startswith("\"") and answer.endswith("\"") and len(answer) > 1:
                answer = answer[1:-1]
            valid_output = answer in ["yes", "no"]
            if not valid_output:
                retries += 1
        if not valid_output:
            answer = "n/a"
        entailment_labels.append(answer)
        entailment_prompt += answer + "\n"

    entailment_prompt += """\nFor each requirement, write whether it is objective and readily measurable by an LLM. Respond with "yes" or "no":"""

    objectivity_labels = []
    for req in requirements:
        entailment_prompt += req + "\n"
        
        valid_output = False
        retries = 0
        while not valid_output and retries < max_retries:
            answer = query_llm(entailment_prompt, max_tokens=3).strip().lower()
            if len(answer.split()) > 1:
                answer = answer.split()[0]
            if answer.startswith("\"") and answer.endswith("\"") and len(answer) > 1:
                answer = answer[1:-1]
            valid_output = answer in ["yes", "no"]
            if not valid_output:
                retries += 1
        if not valid_output:
            answer = "n/a"
        objectivity_labels.append(answer)
        entailment_prompt += answer + "\n"

    entailment_prompt += """\nDoes this checklist contain pairwise-disentangled and non-redundant requirements? Respond with "yes" or "no":"""

    valid_output = False
    retries = 0
    while not valid_output and retries < max_retries:
        disentanglement = query_llm(entailment_prompt, max_tokens=3).strip().lower()
        if len(disentanglement.split()) > 1:
            disentanglement = disentanglement.split()[0]
        if disentanglement.startswith("\"") and disentanglement.endswith("\"") and len(disentanglement) > 1:
            disentanglement = disentanglement[1:-1]
        valid_output = disentanglement in ["yes", "no"]
        if not valid_output:
            retries += 1
    if not valid_output:
        disentanglement = "n/a"
    entailment_prompt += disentanglement + "\n"

    entailment_prompt += """\nAre the items within this checklist all atomic (conditioned on the other items in the checklist)? Respond with yes or "no":"""

    valid_output = False
    retries = 0
    while not valid_output and retries < max_retries:
        atomic = query_llm(entailment_prompt, max_tokens=3).strip().lower()
        if len(atomic.split()) > 1:
            atomic = atomic.split()[0]
        if atomic.startswith("\"") and atomic.endswith("\"") and len(atomic) > 1:
            atomic = atomic[1:-1]
        valid_output = atomic in ["yes", "no"]
        if not valid_output:
            retries += 1
    if not valid_output:
        atomic = "n/a"
    entailment_prompt += atomic + "\n"


    entailment_prompt += """\nIs the checklist comprehensive? Respond with "yes" or "no":"""
    retries = 0
    valid_output = False
    while not valid_output and retries < max_retries:
        comprehensive = query_llm(entailment_prompt, max_tokens=3).strip().lower()
        if len(comprehensive.split()) > 1:
            comprehensive = comprehensive.split()[0]
        if comprehensive.startswith("\"") and comprehensive.endswith("\"") and len(comprehensive) > 1:
            comprehensive = comprehensive[1:-1]
        valid_output = comprehensive in ["yes", "no"]
        if not valid_output:
            retries += 1
    if not valid_output:
        comprehensive = "n/a"
    
    # return entailment_labels, objectivity_labels, disentanglement, atomic, comprehensive
    return RequirementEvaluation(
        entailment_labels=entailment_labels,
        objectivity_labels=objectivity_labels,
        disentanglement=disentanglement,
        atomic=atomic,
        comprehensive=comprehensive)

def format_requirement_list(checklist, labels):
    return "\n".join([f"{checklist[i]}\n{labels[i]}" for i in range(len(checklist))])

def construct_comparison_prompt(instruction,
                     checklist_1,
                     checklist_1_eval,
                     checklist_2,
                     checklist_2_eval): 
    checklist_1_str = json.dumps(checklist_1, indent=4)
    checklist_2_str = json.dumps(checklist_2, indent=4)
    return  f"""You are an AI judge tasked with assessing the quality of requirements produced in response to a given instruction. Given an instruction, we wish to write our a set of criteria (posed as yes/no questions) in order to more objectively assess the quality of an instruction. These checklists will later be given to an AI language model to judge the quality of various responses; accordingly, a good checklist will be as objectively measurable as possible, detailed, and reasonable (in agreement with the original instruction). It will also contain criteria that are atomic (each introducing exactly one new aspect to measure relative to the others), disentangled (not redundant from each other), and comprehensive (collectively covering all key aspects of response quality). You will be given either a set of criteria or individual criteria, and tasked with assessing each of these qualities separately.

Instruction: "{instruction}"

Checklist 1:
{checklist_1_str}

For each requirement, write whether it entailed by the instruction. Respond with "yes" or "no":
{format_requirement_list(checklist_1, checklist_1_eval.entailment_labels)}

For each requirement, write whether it is objective and readily measurable by an LLM. Respond with "yes" or "no"
{format_requirement_list(checklist_1, checklist_1_eval.objectivity_labels)}

Does this checklist contain pairwise-disentangled and non-redundant requirements?
{checklist_1_eval.disentanglement}

Are the items within this checklist all atomic (conditioned on the other items in the checklist)?
{checklist_1_eval.atomic}

Is the checklist comprehensive?
{checklist_1_eval.comprehensive}

Checklist 2:
{checklist_2_str}

For each requirement, write whether it entailed by the instruction. Respond with "yes" or "no":
{format_requirement_list(checklist_2, checklist_2_eval.entailment_labels)}

For each requirement, write whether it is objective and readily measurable by an LLM. Respond with "yes" or "no"
{format_requirement_list(checklist_2, checklist_2_eval.objectivity_labels)}

Does this checklist contain pairwise-disentangled and non-redundant requirements?
{checklist_2_eval.disentanglement}

Are the items within this checklist all atomic (conditioned on the other items in the checklist)?
{checklist_2_eval.atomic}

Is the checklist comprehensive?
{checklist_2_eval.comprehensive}

Overall, which checklist is better, considering all the criteria above as well as your overall assessment of the critical aspects of this instruction? Which checklist would be more useful for guiding an AI language model judge in repeatably, systematically, and faithfully evaluating responses to this instruction?

Respond with "1" for the first checklist, "2" for the second checklist, or "tie" if they are roughly equal. First, write a few sentences of reasoning about which checklist is more useful, and then write "Final Answer: X", where X is either 1, 2, or tie.

Reasoning:"""


def parse_answer_from_reasoning(raw_answer):
    return raw_answer.lower().split("final answer:")[1].strip()


def judge_comparison(instruction: str,
                     joint_reqs: list[str],
                     joint_eval: RequirementEvaluation,
                     agnostic_reqs: list[str],
                     agnostic_eval: RequirementEvaluation,
                     max_retries: int = 3):

    original_order_reqs = [joint_reqs, agnostic_reqs]
    original_order_evals = [joint_eval, agnostic_eval]

    coin_flip = random.choice([0, 1])
    if coin_flip == 0:
        checklist_1, checklist_2 = joint_reqs, agnostic_reqs
        checklist_1_eval, checklist_2_eval = joint_eval, agnostic_eval
    else:
        checklist_1, checklist_2 = agnostic_reqs, joint_reqs
        checklist_1_eval, checklist_2_eval = agnostic_eval, joint_eval

    checklist_1_str = json.dumps(checklist_1, indent=4)
    checklist_2_str = json.dumps(checklist_2, indent=4)

    entailment_prompt = construct_comparison_prompt(instruction, checklist_1, checklist_1_eval, checklist_2, checklist_2_eval)
    flipped_prompt = construct_comparison_prompt(instruction, checklist_2, checklist_2_eval, checklist_1, checklist_1_eval)

    if flipped_prompt in cache:
        entailment_prompt = flipped_prompt
        checklist_1, checklist_2 = checklist_2, checklist_1
        checklist_1_eval, checklist_2_eval = checklist_2_eval, checklist_1_eval
        coin_flip = not coin_flip

    valid_output = False
    retries = 0
    while not valid_output and retries < max_retries:
        if entailment_prompt in cache:
            raw_answer = cache[entailment_prompt]
        else:
            raw_answer = query_llm(entailment_prompt, max_tokens=300).strip().lower()
        answer = parse_answer_from_reasoning(raw_answer)
        if answer.startswith("\"") and answer.endswith("\"") and len(answer) > 1:
            answer = answer[1:-1]
        valid_output = answer in ["1", "2", "tie"]
        if not valid_output:
            retries += 1
    if not valid_output:
        answer = "N/A"
    if answer == "1":
        if not coin_flip:
            return 0
        else:
            return 1
    elif answer == "2":
        if not coin_flip:
            return 1
        else:
            return 0
    else:
        return -1    

def count_aggregate_info(binary_label_array):
    success = len([label for label in binary_label_array if label == "yes"]) 
    failure = len([label for label in binary_label_array if label == "no"])
    nones = len([label for label in binary_label_array if label == "N/A"])
    return success, failure, nones

def compute_aggregate_info(single_model_evals):
    entailment_successes = []
    entailment_failures = []

    objectivity_successes = []
    objectivity_failures = []

    disentanglements = []
    atomics = []
    comprehensivenesses = []

    for prompt_eval in single_model_evals:
        entailment_succ, entailment_fail, _ = count_aggregate_info(prompt_eval.entailment_labels)
        entailment_successes.append(entailment_succ)
        entailment_failures.append(entailment_fail)

        objectivity_succ, objectivity_fail, _ = count_aggregate_info(prompt_eval.objectivity_labels)
        objectivity_successes.append(objectivity_succ)
        objectivity_failures.append(objectivity_fail)

        if prompt_eval.disentanglement == "yes":
            disentanglements.append(1)
        elif prompt_eval.disentanglement == "no":
            disentanglements.append(0)
        
        if prompt_eval.atomic == "yes":
            atomics.append(1)
        elif prompt_eval.atomic == "no":
            atomics.append(0)

        if prompt_eval.comprehensive == "yes":
            comprehensivenesses.append(1)
        elif prompt_eval.comprehensive == "no":
            comprehensivenesses.append(0)

    entailment_average = sum(entailment_successes) / (sum(entailment_successes) + sum(entailment_failures))
    objectivity_average = sum(objectivity_successes) / (sum(objectivity_successes) + sum(objectivity_failures))
    disentanglement_average = sum(disentanglements) / len(disentanglements)
    atomic_average = sum(atomics) / len(atomics)
    comprehensive_average = sum(comprehensivenesses) / len(comprehensivenesses)

    return entailment_average, objectivity_average, disentanglement_average, atomic_average, comprehensive_average


if __name__ == "__main__":
    args = parser.parse_args()
    if os.path.exists(cache_file):
        cache = json.load(open(cache_file))
    else:
        cache = {}

    initial_cache_size = len(cache)
    api_key=os.environ.get("LITELLM_API_KEY")

    if args.input_folder.startswith("~"):
        args.input_folder = os.path.expanduser(args.input_folder)
    if args.infobench_file.startswith("~"):
        args.infobench_file = os.path.expanduser(args.infobench_file)

    infobench_requirements = json.load(open(args.infobench_file, "r"))
    jointly_judgments = remove_low_weight(json.load(open(f"{args.input_folder}/joint_requirements.json")))
    agnostic_judgments = remove_low_weight(json.load(open(f"{args.input_folder}/agnostic_requirements.json")))

    joint_betters = []
    all_reasonability_segments = []
    all_objectivity_segments = []
    all_disentanglements = []
    all_atomics = []
    all_comprehensivenesses = []

    joint_evals = []
    agnostic_evals = []
    gt_evals = []

    joint_vs_gt = []
    agnostic_vs_gt = []
    joint_vs_agnostic = []

    for i, (instruction, gt_reqs) in tqdm(enumerate(list(infobench_requirements.items())[:49])):
        joint_reqs = jointly_judgments[instruction]
        agnostic_reqs = agnostic_judgments[instruction]
        
        joint_eval = judge_basic_info(instruction, joint_reqs)
        agnostic_eval = judge_basic_info(instruction, agnostic_reqs)
        gt_eval = judge_basic_info(instruction, gt_reqs)

        joint_evals.append(joint_eval)
        agnostic_evals.append(agnostic_eval)
        gt_evals.append(gt_eval)

        compare_joint_gt = judge_comparison(instruction, joint_reqs, joint_eval, gt_reqs, gt_eval)
        compare_agnostic_gt = judge_comparison(instruction, agnostic_reqs, agnostic_eval, gt_reqs, gt_eval)
        compare_joint_agnostic = judge_comparison(instruction, joint_reqs, joint_eval, agnostic_reqs, agnostic_eval)

        joint_vs_gt.append(compare_joint_gt)
        agnostic_vs_gt.append(compare_agnostic_gt)
        joint_vs_agnostic.append(compare_joint_agnostic)

    breakpoint()

    joint_vs_gt_no_ties = [label for label in joint_vs_gt if label != -1]
    agnostic_vs_gt_no_ties = [label for label in agnostic_vs_gt if label != -1]
    joint_vs_agnostic_no_ties = [label for label in joint_vs_agnostic if label != -1]

    joint_entailment_average, joint_objectivity_average, joint_disentanglement_average, joint_atomic_average, joint_comprehensive_average = compute_aggregate_info(joint_evals)
    print("\n\nResponse-Based Requirement Generation:")
    print(f"Entailment: {round(joint_entailment_average * 100, 2)}%")
    print(f"Objectivity: {round(joint_objectivity_average * 100, 2)}%")
    print(f"Disentanglement: {round(joint_disentanglement_average * 100, 2)}%")
    print(f"Atomicity: {round(joint_atomic_average * 100, 2)}%")
    print(f"Comprehensiveness: {round(joint_comprehensive_average * 100, 2)}%")
    print(f"% better than GT: {round(sum(joint_vs_gt_no_ties) / len(joint_vs_gt) * 100, 2)}%")
    print(f"% better than Agnostic: {round(sum(joint_vs_agnostic_no_ties) / len(joint_vs_agnostic) * 100, 2)}")

    agnostic_entailment_average, agnostic_objectivity_average, agnostic_disentanglement_average, agnostic_atomic_average, agnostic_comprehensive_average = compute_aggregate_info(agnostic_evals)
    print("\n\nResponse-Agnostic Requirement Generation:")
    print(f"Entailment: {round(agnostic_entailment_average * 100, 2)}%")
    print(f"Objectivity: {round(agnostic_objectivity_average * 100, 2)}%")
    print(f"Disentanglement: {round(agnostic_disentanglement_average * 100, 2)}%")
    print(f"Atomicity: {round(agnostic_atomic_average * 100, 2)}%")
    print(f"Comprehensiveness: {round(agnostic_comprehensive_average * 100, 2)}%")
    print(f"% better than GT: {round(sum(agnostic_vs_gt_no_ties) / len(agnostic_vs_gt) * 100, 2)}%")
    print(f"% better than Joint: {round((len(joint_vs_agnostic_no_ties) - sum(joint_vs_agnostic_no_ties)) / len(joint_vs_agnostic) * 100, 2)}%")

    gt_entailment_average, gt_objectivity_average, gt_disentanglement_average, gt_atomic_average, gt_comprehensive_average = compute_aggregate_info(gt_evals)
    print("\n\nGround Truth Requirement Generation:")
    print(f"Entailment: {round(gt_entailment_average * 100, 2)}%")
    print(f"Objectivity: {round(gt_objectivity_average * 100, 2)}%")
    print(f"Disentanglement: {round(gt_disentanglement_average * 100, 2)}%")
    print(f"Atomicity: {round(gt_atomic_average * 100, 2)}%")
    print(f"Comprehensiveness: {round(gt_comprehensive_average * 100, 2)}%")
    print(f"% better than Joint: {round((len(joint_vs_gt_no_ties) - sum(joint_vs_gt_no_ties)) / len(joint_vs_gt) * 100, 2)}%")
    print(f"% better than Agnostic: {round((len(agnostic_vs_gt_no_ties) - sum(agnostic_vs_gt_no_ties)) / len(agnostic_vs_gt) * 100, 2)}%")
    
    breakpoint()

    if len(cache) > initial_cache_size:
        json.dump(cache, open(cache_file, 'w'))
