import argparse
import glob
import json
import jsonlines
import os
import math
import pickle
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch

from prompt import LLAMA_SYSTEM_PROMPT, LLAMA_SYSTEM_PROMPT_NO_REFERENCE, LLAMA_SYSTEM_PROMPT_NO_CANDIDATES


supported_models = ["Qwen/Qwen2.5-0.5B",
                    "Qwen/Qwen2.5-1.5B",
                    "Qwen/Qwen2.5-3B",
                    "Qwen/Qwen2.5-7B"]

parser = argparse.ArgumentParser()
parser.add_argument("--job-idx", type=int, required=True)
parser.add_argument("--num-jobs", type=int, default=8)
parser.add_argument("--combined-response-data", type=str, default="combined_wildchat_responses")
parser.add_argument("--out-dir", type=str, default="combined_wildchat_requirements")
parser.add_argument("--batch-size", type=int, default=5000)
parser.add_argument("--batch-offset", type=int, default=0)
parser.add_argument("--max-batch-idx", type=int, default=-1)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--direct-generation", action="store_true", help="Whether to use direct generation mode")
parser.add_argument("--skip-reference", action="store_true", help="Whether to skip the reference response in the prompt")


def split_list(lst, n):
    # Check if n is valid
    if n <= 0 or n > len(lst):
        raise ValueError("n must be a positive integer not greater than the length of the list")
    # Create n empty lists to hold the chunks
    chunks = [[] for _ in range(n)]
    # Distribute elements to chunks
    for i, item in enumerate(lst):
        chunks[i % n].append(item)
    return chunks

def split_list_contiguous(lst, split_size):
    num_splits = math.ceil(len(lst) / split_size)
    result = []
    start = 0
    for _ in range(num_splits):
        # Determine the end index for this split
        end = start + split_size
        if end > len(lst):
            end = len(lst)
        # Add this group of elements to the result
        result.append(lst[start:end])
        # Update the start index for the next split
        start = end
    return result

def format_instruction(conversation, value_key):
    if len(conversation) == 2:
        instruction = conversation[0][value_key]
    else:
        instruction = conversation[0][value_key] + "\n" + conversation[1][value_key]
    return instruction

def format_multiple_responses(responses_by_model):
    formatted_str = None
    for i, model in enumerate(supported_models):
        addendum = f"Response {i+1}:\n{responses_by_model[model]}"
        if formatted_str is None:
            formatted_str = addendum
        else:
            formatted_str = f"{formatted_str}\n--\n{addendum}"
    return formatted_str

def parse_response(response):
    if "<END>" in response:
        response_up_to_end = response.split("<END>")[0].strip()
    else:
        response_up_to_end = response
    if len(response_up_to_end.split("Key Criteria Questions:")) == 2:
        response_up_to_end = response_up_to_end.split("Key Criteria Questions:")[1].strip()
    requirements = []
    if response_up_to_end.strip().lower() == "none":
        return []
    if "* " in response_up_to_end:
        lines = response_up_to_end.split("* ")
    else:
        lines = response_up_to_end.split("- ")
    for line in lines:
        line = line.strip()
        if line.startswith("- "):
            line = line[2:]
        if line.strip().lower() == "none":
            return []
        if line.endswith(")") and line.split(")")[-2].split("(")[-1].isnumeric():
            requirement = "(".join(line.split("(")[:-1]).strip()
            weight = float(line.split(")")[-2].split("(")[-1])
            if len(requirement.split()) > 0 and not requirement.startswith("Here are"):
                requirements.append([requirement, weight])
    return requirements


def get_value_key(conversation):
    if "value" in conversation[0].keys():
        value_key = "value"
    elif "response" in conversation[0].keys():
        value_key = "response"
    elif "content" in conversation[0].keys():
        value_key = "content"
    else:
        raise ValueError(f"Could not find value key in conversation: {conversation}")
    return value_key


if __name__ == "__main__":

    args = parser.parse_args()

    all_data_files = glob.glob(os.path.join(args.combined_response_data, "*.json"))
    job_data_files = split_list(all_data_files, args.num_jobs)[args.job_idx]

    rows = []
    orig_file = []
    for data_file in job_data_files:
        with open(data_file, 'r') as f:
            shard_rows = json.load(f)
            rows.extend(shard_rows)
            orig_file.extend([data_file] * len(shard_rows))


    stop_tokens = ["USER:", "ASSISTANT:",  "### Instruction:", "Response:", "<|eot_id|>", "####", "<END>"]
    sampling_params = SamplingParams(temperature=args.temperature, repetition_penalty=1.0, top_p=0.9, max_tokens=1024, stop=stop_tokens)
    model = LLM(model="Qwen/Qwen2.5-72B-Instruct", tensor_parallel_size=torch.cuda.device_count(),
                dtype="bfloat16", trust_remote_code=True)

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = model.get_tokenizer()

    prompts = []
    model_responses = []
    for row in tqdm(rows):
        conversation = row["conversations"]
        VALUE_KEY = get_value_key(conversation)

        instruction = format_instruction(conversation, VALUE_KEY)
        gpt_4_response = conversation[-1][VALUE_KEY]

        if args.direct_generation:
            formatted_parsing_prompt = LLAMA_SYSTEM_PROMPT_NO_CANDIDATES.replace("{instruction}", instruction).replace("{gpt_4_response}", gpt_4_response)
        else:
            formatted_response = format_multiple_responses(row["all_responses"])
            if args.skip_reference:
                formatted_parsing_prompt = LLAMA_SYSTEM_PROMPT_NO_REFERENCE.replace("{instruction}", instruction).replace("{formatted_worker_responses}", formatted_response)
            else:
                formatted_parsing_prompt = LLAMA_SYSTEM_PROMPT.replace("{instruction}", instruction).replace("{gpt_4_response}", gpt_4_response).replace("{formatted_worker_responses}", formatted_response)
        prompts.append(formatted_parsing_prompt)

    if args.batch_size == -1:
        batches = [list(range(len(prompts)))]
    else:
        batches = split_list_contiguous(list(range(len(prompts))), args.batch_size)

    if args.max_batch_idx != -1:
        batches = batches[:args.max_batch_idx]

    if args.batch_offset == 0:
        matching_job_files = glob.glob(os.path.join(args.out_dir, f"{args.job_idx}_batch_*.jsonl"))
        matching_batch_numbers = [int(f.split("_")[-1].split(".")[0]) for f in matching_job_files]
        # Find the maximum batch number for which the associated file has nonzero size
        largest_existing_batch_number = -1
        for batch_number in sorted(matching_batch_numbers, reverse=True):
            file_path = os.path.join(args.out_dir, f"{args.job_idx}_batch_{batch_number}.jsonl")
            if os.path.getsize(file_path) > 0:
                largest_existing_batch_number = batch_number
                break
        args.batch_offset = largest_existing_batch_number + 1

    batches = batches[args.batch_offset:]

    for batch_number, batch_idxs in enumerate(batches):
        revised_batch_number = batch_number + args.batch_offset
        print(f"Starting batch {revised_batch_number} of {len(batches) + args.batch_offset}")
        with jsonlines.open(os.path.join(args.out_dir, f"{args.job_idx}_batch_{revised_batch_number}.jsonl"), mode="w") as out_file:

            prompt_batch = []
            for i in batch_idxs:
                prompt = prompts[i]
                if len(tokenizer.tokenize(prompt)) > min(min(model.max_length, tokenizer.model_max_length), 32000):
                    continue
                prompt_batch.append(prompt)
            responses = model.generate(prompt_batch, sampling_params, use_tqdm=True)

            out_objects = []
            for row_idx, prompt, response in zip(batch_idxs, prompt_batch, responses):
                orig_row = rows[row_idx]
                data_file = orig_file[row_idx]
                instruction = format_instruction(orig_row["conversations"], get_value_key(orig_row["conversations"]))

                response_text = response.outputs[0].text

                try:
                    parsed_response = parse_response(response_text)
                except:
                    parsed_response = []

                out_object = {
                    "instruction": instruction,
                    "raw_response": response_text,
                    "requirements": parsed_response,
                    "orig_row": orig_row,
                    "orig_row_idx": row_idx,
                    "source_file": data_file,
                }
                out_objects.append(out_object)
            out_file.write_all(out_objects)