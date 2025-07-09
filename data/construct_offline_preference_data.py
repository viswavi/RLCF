import argparse
import glob
import json
import jsonlines
import os
import litellm
import math
import numpy as np
import pickle
import requests
import string

from datasets import load_dataset
import sglang
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
from transformers import AutoTokenizer


from prompt import LLAMA_SYSTEM_PROMPT

supported_models = ["Qwen/Qwen2.5-0.5B",
                    "Qwen/Qwen2.5-1.5B",
                    "Qwen/Qwen2.5-3B",
                    "Qwen/Qwen2.5-7B"]

parser = argparse.ArgumentParser()
parser.add_argument("--requirements-dir", type=str, default="combined_wildchat_requirements")
parser.add_argument("--out-dir", type=str, default="combined_wildchat_requirements/preference_data")
parser.add_argument("--candidates-source", type=str, choices=["rewardbench", "wildchat"], default="wildchat")
parser.add_argument("--use-reward-model", action="store_true")
parser.add_argument("--sglang-url", type=str, default=None)
parser.add_argument("--wildchat-candidates-glob", type=str, default="wildchat_responses/Qwen_Qwen2_5_7B_run_[0-9]_wildchat_response.jsonl")
parser.add_argument("--generate-requirements-online", action="store_true")
parser.add_argument("--produce-numerical-answers", action="store_true")
parser.add_argument("--probability-estimation", action="store_true")
parser.add_argument("--inference-type", type=str, default="vllm")
parser.add_argument("--evaluate-rewardbench", action="store_true")
parser.add_argument("--batch-start-idx", type=int, default=None)
parser.add_argument("--batch-end-idx", type=int, default=None)
parser.add_argument("--parse-step-by-step-responses", action="store_true", help="Whether or not to explicitly exclude reasoning chain before parsing the answer")
parser.add_argument("--add-universal-requirements", action="store_true", help="Whether or not to add universal requirements to the checklist")


def batch_examples(examples, batch_size):
    return [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]

def get_num_requirement_batches(requirements_dir):
    return len(glob.glob(os.path.join(requirements_dir, "*.jsonl")))

def get_universal_requirements():
    return [
        ("""Does the response satisfy the following two criteria:
1) The response directly address the request without excessive or off-topic information not necessary for addressing the user's instruction?
2) The response should match the context and the instruction, whether it requires professionalism, friendliness, formality, or neutrality.""", 100),
    ]

def load_requirements(requirements_dir, add_universal_requirements=False):
    for file in glob.glob(os.path.join(requirements_dir, "*.jsonl")):
        instruction_to_requirements = {}
        with jsonlines.open(file, 'r') as f:
            for line in f:
                if add_universal_requirements:
                    requirements = line["requirements"] + get_universal_requirements()
                else:
                    requirements = line["requirements"]
                instruction_to_requirements[line["instruction"]] = requirements
            yield instruction_to_requirements

def load_rewardbench_responses():
    rows = load_dataset("allenai/reward-bench", split='filtered')
    all_responses = {}
    metadata = {}
    for row in rows:
        all_responses[row["prompt"]] = [row["chosen"], row["rejected"]]
        if row["prompt"] not in metadata:
            metadata[row["prompt"]] = []
        metadata[row["prompt"]].append(
            {"subset": row["subset"], "chosen_model": row["chosen_model"], "rejected_model": row["rejected_model"]}
        )
    return all_responses, metadata

def create_judgment_prompt(response, instruction, requirement, binary_prediction=True):
    numerical_prompt = f""""Based on the provided input instruction and response from a worker, assess the response based on the following single-criterion question. Score the response with a rating (a number between 0 and 100) assessing how well the response answers that question. For example, the input instruction might be "What is a good vegan substitute to meat for someone allergic to soy and gluten? Provide an answer followed by a factually detailed and humorous one-sentence explanation" and the criterion question might be "Is the explanation factually detailed?". Your selection should be based primarily on the response and the question alone, with the instruction shown for context when needed:
- 100: Select 100 if the generated text represents an optimal solution that expertly balances all relevant factors mentioned in the question. For objective criteria (like "Does each sentence in the generated text use a second person?"), even minor deviations exclude a 100 rating (and probably lead to a 0 rating). For subjective criteria, the response should basically be perfect. For the example above (about the vegan substitute), and the criterion above (about factual detail), an example 100-point response is "Mushrooms, because they can be easily caramelized and browned, they are rich in the glutamates which lead to incredible umami flavors, they can be cooked into crispy OR chewy meatlike textures.". This response is richly detailed and factual, and though it fails to be humorous, it is still a 100-point response on the factual detail criterion.
- 75: Return ~75 if the generated text very effectively addresses the main requirements but has room for minor improvements. The response should be unconditionally acceptable (at a professional level) but may not be absolutely perfect. There are no mistakes that critically undermine the question. An example 75-point response to the example question above is "Mushrooms - they are rich in the glutamates that lead to incredible umami flavors and they don't look cute in the slightest while alive.". This response has one interesting fact but could be more detailed.
- 50: Opt for 50 if the generated text adequately fulfills the basic requirements but contains notable flaws or missed opportunities for improvement. The response should still be functionally acceptable. The response contains at most one minor inadequacy or inaccuracy related to the question but there are no mistakes that critically undermine the question. An example 50-point response to the example question above is "Mushrooms, because they can be easily caramelized and browned, they're universally beloved by sophisticated palates, and they don't look cute in the slightest while alive." The statement that they're universally beloved by people with sophisticated palates, while potentially true, is vague and not objective.
- 25: Return ~25 if the generated text fulfills the key condition specified by the question and demonstrates awareness of the key requirements but fails to execute them effectively. The text may contain non-critical inaccuracies or irrelevant information. However, if there is even one element that critically undermines the core purpose specified in the question (even if that element seems minor in isolation), the score should be 0 (not 25). An example 25-point response to the example question above is "Mushrooms, because they can be easily caramelized and browned, universally beloved by kids, and they don't look cute in the slightest while alive." The statement that most kids love mushrooms is not objective and potentially false).
- 0: Opt for 0 if the generated text fails to meet the question’s requirements or provides no information that could be utilized to answer the question. If the response contains a critical error relevant to the question, return a 0. For the question about the vegan substitute, an example 0-point response is "Mushrooms, because they make you question why you ever thought a dead animal could compare to this vegan delight." While funny and engaging, this response contains zero factual detail about mushrooms, critically violating the question.

Your score can be any number between 0 and 100 (not just the ones listed above). If you are totally confused, return -1 as a default. You should use your judgment to determine the most appropriate score. Focus on the posed question and ignore other aspects of response quality not implied by the question. Return only a number - do not include any other text in your response.

Input:
{instruction}

Generated Text:
{response}

Question:
{requirement}

Score: """
    binary_prompt = f""""Based on the provided input instruction and response from a worker, assess the response based on the following single-criterion question with either A (YES) or B (NO) based on how well the response satisfies that criterion. Your selection should be based primarily on the response and the question alone, with the instruction shown for context when needed. Respond with "A" (Yes) if the generated text effectively and optimally addresses the criterion specified by the question, and "B" (No) if it does not. Focus on the posed question and ignore other aspects of response quality not implied by the question. A successful response must completely satisfy the given criterion. If the response fails to satisfy any part of the question, respond with "B" (No).

Instruction:
{instruction}

Response:
{response}

Question:
{requirement}

(A) Yes
(B) No

The best answer is:"""
    if binary_prediction:
        return binary_prompt
    else:    
        return numerical_prompt

def query_model(prompts, paired_instruction_response_and_requirement, model, tokenizer, binary_prediction=False, use_ensemble_sampling=True, batch_size = 10000):
    scores = []
    cache = {}
    for prompt, prompt_response_and_requirement in zip(prompts, paired_instruction_response_and_requirement):
        (_, _, parsed_response, _) = prompt_response_and_requirement
        # If the response here is None, that indicates that the formatting of this
        # response is corrupted. Immediately produce a score of "0".
        if parsed_response is None and prompt not in cache:
            cache[prompt] = 0.0
        tokens = tokenizer(prompt, truncation=False)
        if len(tokens["input_ids"]) > 30000:
            cache[prompt] = -1

    if isinstance(model, LLM):
        stop_tokens = ["USER:", "ASSISTANT:",  "### Instruction:", "Response:", "<|eot_id|>", "####", "<END>"]
        sampling_params = SamplingParams(temperature=0.0, repetition_penalty=1.0, top_p=0.9, max_tokens=10, stop=stop_tokens)
        if binary_prediction:
            if use_ensemble_sampling:
                sampling_params.n = 25
            else:
                sampling_params.logprobs = 5
        else:
            if use_ensemble_sampling:
                sampling_params.n = 25
                sampling_params.temperature = 1.3
                sampling_params.top_p = 0.9

        new_prompts = [p for p in prompts if p not in cache]

        batches = batch_examples(new_prompts, batch_size=batch_size)

        for batch_idx, batch_prompts in enumerate(batches):
            print(f"Starting batch {batch_idx + 1} of {len(batches)}")

            batch_responses = model.generate(batch_prompts, sampling_params)

            for i, (prompt, response) in enumerate(zip(batch_prompts, batch_responses)):

                if binary_prediction:
                    if use_ensemble_sampling:
                        tokens = [output.text.strip().lower() for output in response.outputs]
                        number_of_yesses = len([t for t in tokens if t == "yes"])
                        if len(tokens) == 0:
                            score = -1
                        else:
                            score = float(len(number_of_yesses)) / len(tokens)
                    else:
                        first_matching_token = None
                        for token_object in response.outputs[0].logprobs:
                            if len(token_object) == 0:
                                continue
                            first_tokens = []
                            first_probabilities = []
                            top_idx = None
                            top_prob = -1000000000
                            for k, v in token_object.items():
                                if v.logprob > top_prob:
                                    top_prob = v.logprob
                                    top_idx = k
                            if token_object[top_idx].decoded_token.strip().lower() in ["a", "b"]:
                                first_matching_token = token_object
                                break
                        if first_matching_token is None:
                            score = -1
                        else:
                            matching_tokens = [t for t in first_matching_token if not isinstance(t, int) and t.decoded_token.strip().lower() in ["a", "b"]]
                            if len(matching_tokens) == 0:
                                score = -1
                            else:
                                score = 0.0
                                total_probability = 0.0
                                for token in matching_tokens:
                                    probability = math.exp(token.logprob)
                                    if token.decoded_token.lower().strip() == "a":
                                        score += probability
                                    total_probability += probability

                                if total_probability > 0.0:
                                    score /= total_probability
                                else:
                                    score = -1
                else:
                    completions = [output.text for output in response.outputs]
                    completion_values = []
                    for completion in completions:
                        if len(completion.strip()) == 0:
                            continue
                        stripped_completion = completion.strip().split()[0]
                        if stripped_completion.isnumeric():
                            score = float(stripped_completion)
                            score = min(max(score, 0), 100)
                            completion_values.append(score)
                    if len(completion_values) == 0:
                        score = -1
                    else:
                        score = sum(completion_values) / len(completion_values)
                cache[prompt] = score
    else:
        new_prompts = [p for p in prompts if p not in cache]

        for prompt in new_prompts:
            if binary_prediction:
                client = litellm.completion(
                    api_key=os.environ.get("LITELLM_API_KEY"),
                    base_url="https://cmu.litellm.ai",
                    model="openai/neulab/meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    logprobs=3,
                )
                assert hasattr(client.choices[0], "logprobs")
                top_logprobs = client.choices[0].logprobs["top_logprobs"]
                first_token_logprobs = top_logprobs[0]
                keys = sorted(list(first_token_logprobs.keys()))
                values = [math.exp(first_token_logprobs[key]) for key in keys]
                score = 0.0
                total_probability = 0.0
                for response, probability in zip(keys, values):
                    if response.lower().strip() == "yes":
                        score += probability
                    total_probability += probability
                if total_probability > 0.0:
                    score /= total_probability
                cache[prompt] = score
            else:
                client = litellm.completion(
                    api_key=os.environ.get("LITELLM_API_KEY"),
                    base_url="https://cmu.litellm.ai",
                    model="openai/neulab/meta-llama/Meta-Llama-3.1-70B-Instruct",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
                score = float(client.choices[0].message.content)
            cache[prompt] = score

    for prompt in prompts:
        if prompt not in cache:
            print(f"Prompt missing")
            score = -1
        else:
            score = cache[prompt]
        scores.append(score)

    return scores

def query_model_skyworks(model, convs, tokenizer, sglang_url, cache_file = "skyworks_cache.pkl", batch_size = 50, cache_interval = 3000):
    if not os.path.exists(cache_file):
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        cache = {}
    else:
        cache = pickle.load(open(cache_file, "rb"))

    prompts = tokenizer.apply_chat_template(convs, tokenize=False)

    new_prompts = [p for p in prompts if p not in cache]
    batches = batch_examples(new_prompts, batch_size=batch_size)

    num_prompts_since_last_cache = 0
    for batch_prompts in tqdm(batches):
        if sglang_url is not None:
            data = {"model": "Skywork/Skywork-Reward-Gemma-2-27B-v0.2", "text": batch_prompts}
            NUM_RETRIES = 3
            success = False
            for i in range(NUM_RETRIES):
                try:
                    responses = requests.post(sglang_url, json=data).json()
                    for prompt, response in zip(batch_prompts, responses):
                        cache[prompt] = response["embedding"][0]
                    success = True
                    break
                except Exception as e:
                    print(f"Error querying model: {e}")
                    continue
            if not success:
                continue
        else:
            try:
                responses = model.encode(batch_prompts)
                for prompt, response in zip(batch_prompts, responses):
                    cache[prompt] = response["embedding"][0]
            except:
                print(f"Skipping batch of {len(batch_prompts)} prompts")

        num_prompts_since_last_cache += len(batch_prompts)
        if num_prompts_since_last_cache >= cache_interval or batch_idx == len(batches) - 1:
            print(f"Caching responses - {len(cache)} responses cached")
            pickle.dump(cache, open(cache_file, "wb"))
            num_prompts_since_last_cache = 0

    scores = []
    for prompt in prompts:
        if prompt not in cache:
            score = np.nan
        else:
            score = cache[prompt]
        scores.append(score)
    return scores


def deduplicate_responses(responses):
    seen_prompts = set()
    deduplicated_responses = []
    for response in responses:
        prompt = response["prompt"]
        if prompt not in seen_prompts:
            deduplicated_responses.append(response)
            seen_prompts.add(prompt)
    return deduplicated_responses

def parse_response_without_reasoning(response: str) -> str:
    answer_delimiter = "<ANSWER>"
    if answer_delimiter not in response:
        return None
    answer_segment = response.split(answer_delimiter)[1]
    if len(answer_segment.strip()) == 0:
        return None
    if answer_segment[0] == ":" or answer_segment[0] == "-":
        answer_segment = answer_segment[1:]
    parsed_response = answer_segment.strip()
    return parsed_response

def construct_prompt_with_reasoning_template(prompt: str) -> str:
    prompt_template = f"""You are an AI assistant that is responsible for giving helpful, accurate, and harmless responses to user queries. For each user query, first write \"<THINKING>\", and think out loud in a step-by-step manner about what the instruction is asking you to do. Then, on a new line, write \"<ANSWER>\" and then write your final response.

Example user query:
Write a 3-line haiku about pink frogs

<THINKING>
The user has made a request to write a haiku about pink frogs. Let me analyze what I need to do:
1. A haiku is a short poem that follows a 5-7-5 syllable pattern across three lines.
2. The subject matter of the haiku should be pink frogs.
3. Haikus typically exhibit strong imagery and often focus on nature. Therefore, I should create an aesthetically pleasing haiku that captures something interesting about pink frogs.

<ANSWER>
Pink frogs in the sun
Beautifully glistening
Nature's rare treasure

New user query:
{prompt}

<THINKING>"""
    return prompt_template


def load_wildchat_response_pairs(responses_a_file, responses_b_file):
    responses_a = deduplicate_responses(jsonlines.open(responses_a_file))
    responses_b = deduplicate_responses(jsonlines.open(responses_b_file))
    response_pairs_by_instruction = {}
    for response_list in [responses_a, responses_b]:
        for row in response_list:
            prompt = row["prompt"]
            response = row["response"]
            if prompt not in response_pairs_by_instruction:
                response_pairs_by_instruction[prompt] = []
            response_pairs_by_instruction[prompt].append(response)
    response_pairs_by_instruction = {p: rs for p, rs in response_pairs_by_instruction.items() if len(set(rs)) == 2}
    candidate_response_counts = []
    for prompt in response_pairs_by_instruction:
        candidate_response_counts.append(len(response_pairs_by_instruction[prompt]))
    assert set(candidate_response_counts) == {2}
    return response_pairs_by_instruction


MIDJOURNEY_PROMPT = """As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and you will provide a detailed prompt for Midjourney AI to generate an image."""


if __name__ == "__main__":
    args = parser.parse_args()
    if args.inference_type == "vllm":
        model = LLM(
                    model="Qwen/Qwen2.5-72B-Instruct", tensor_parallel_size=torch.cuda.device_count(),
                    dtype="bfloat16"
                )
    elif args.use_reward_model and args.inference_type == "sglang":
        model = sglang.Engine(model_path="Skywork/Skywork-Reward-Gemma-2-27B-v0.2", is_embedding=True)
    else:
        assert args.inference_type == "litellm" or args.inference_type == "sglang"
        model = None

    requirement_batches = load_requirements(args.requirements_dir, args.add_universal_requirements)
    if not args.generate_requirements_online:
        if args.candidates_source == "rewardbench":
            responses, metadata = load_rewardbench_responses()
        elif args.candidates_source == "wildchat":
            candidates_files = glob.glob(args.wildchat_candidates_glob)
            assert len(candidates_files) == 2
            responses_a_file, responses_b_file = candidates_files
            responses = load_wildchat_response_pairs(responses_a_file, responses_b_file)
            metadata = None
            

    if args.batch_start_idx is None:
        batch_start_idx = 0
    else:
        batch_start_idx = args.batch_start_idx

    if args.batch_end_idx is None:
        batch_end_idx = get_num_requirement_batches(args.requirements_dir) - 1
    else:
        batch_end_idx = args.batch_end_idx

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.use_reward_model:
        if args.inference_type == "sglang":
            tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Gemma-2-27B-v0.2")
        else:
            tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")

    for batch_idx, batch_requirements in tqdm(enumerate(requirement_batches)):
        if batch_idx < batch_start_idx or batch_idx > batch_end_idx:
            continue

        print(f"Starting batch {batch_idx + 1} (in range {batch_start_idx} to {batch_end_idx})")
        if os.path.exists(os.path.join(args.out_dir, f"{batch_idx}.json")) and os.path.getsize(os.path.join(args.out_dir, f"{batch_idx}.json")) > 0:
            continue
        paired_instruction_response_and_requirement = []

        filtered_instructions = [i for i in batch_requirements if not i.strip().startswith(MIDJOURNEY_PROMPT)]

        instructions = {}

        if args.use_reward_model:
            assert tokenizer is not None
            convs = []
            for instruction in filtered_instructions:
                if args.parse_step_by_step_responses:
                    instruction_with_optional_template = construct_prompt_with_reasoning_template(instruction)
                else:
                    instruction_with_optional_template = instruction
                if instruction not in instructions:
                    instructions[instruction] = {"scores": {}}
                if instruction_with_optional_template not in responses:
                    continue
                response_pair = responses[instruction_with_optional_template]
                for response in response_pair:
                    if args.parse_step_by_step_responses:
                        parsed_response = parse_response_without_reasoning(response)
                        conv = [{"role": "user", "content": instruction}, {"role": "assistant", "content": parsed_response}]
                    else:
                        conv = [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]

                    tokens = tokenizer.apply_chat_template(conv, tokenize=True)
                    MAX_SEQ_LEN=8000
                    if len(tokens) > tokenizer.model_max_length:
                        print("Row removed for being too long")
                        continue

                    prompts = tokenizer.apply_chat_template(conv, tokenize=False)
                    convs.append(conv)

                    paired_instruction_response_and_requirement.append((instruction_with_optional_template, response))

            cache_file = f"skyworks_cache_{batch_idx}.pkl"
            evaluator_scores = query_model_skyworks(model, convs, tokenizer, args.sglang_url, cache_file = cache_file)

            for score, (instruction, model_response) in zip(evaluator_scores, paired_instruction_response_and_requirement):
                if np.isnan(score):
                    if instruction in instructions:
                        del instructions[instruction]
                    continue
                if instruction not in instructions:
                    instructions[instruction] = {"scores": {}}
                instructions[instruction]["scores"][model_response] = [score]

        else:
            prompts = []
            for instruction in filtered_instructions:
                if args.parse_step_by_step_responses:
                    instruction_with_optional_template = construct_prompt_with_reasoning_template(instruction)
                else:
                    instruction_with_optional_template = instruction
                if instruction_with_optional_template not in instructions:
                    instructions[instruction_with_optional_template] = {"scores": {}, "requirements": [], "labels": []}
                if instruction_with_optional_template not in responses:
                    continue
                response_pair = responses[instruction_with_optional_template]
                for response in response_pair:
                    if response not in instructions[instruction_with_optional_template]["scores"]:
                        instructions[instruction_with_optional_template]["scores"][response] = []
                    for req in batch_requirements[instruction]:
                        requirement_text = req[0]

                        if args.parse_step_by_step_responses:
                            parsed_response = parse_response_without_reasoning(response)
                        else:
                            parsed_response = response

                        prompt = create_judgment_prompt(parsed_response, instruction, requirement_text, binary_prediction=not args.produce_numerical_answers)
                        prompts.append(prompt)
                        paired_instruction_response_and_requirement.append((instruction_with_optional_template, response, parsed_response, req))

            evaluator_scores = query_model(prompts,
                                           paired_instruction_response_and_requirement,
                                           model,
                                           tokenizer,
                                           binary_prediction=not args.produce_numerical_answers,
                                           use_ensemble_sampling=not args.probability_estimation)

            for score, (instruction_with_optional_template, model_response, parsed_model_response, requirement) in zip(evaluator_scores, paired_instruction_response_and_requirement):
                if instruction_with_optional_template not in instructions:
                    instructions[instruction_with_optional_template] = {"scores": {}, "requirements": [], "labels": []}
                if parsed_model_response is None:
                    scaled_score = 0.0
                else:
                    scaled_score = float(requirement[1]) * float(score)
                instructions[instruction_with_optional_template]["scores"][model_response].append(scaled_score)

            for instruction in batch_requirements:
                if args.parse_step_by_step_responses:
                    instruction_with_optional_template = construct_prompt_with_reasoning_template(instruction)
                else:
                    instruction_with_optional_template = instruction

                if instruction_with_optional_template not in instructions:
                    continue
                instructions[instruction_with_optional_template]["requirements"] =  batch_requirements[instruction]
                if args.candidates_source == "rewardbench" and args.evaluate_rewardbench:
                    chosen, rejected = responses[instruction_with_optional_template]
                    instructions[instruction_with_optional_template]["labels"] = {chosen: "chosen", rejected: "rejected"}

        json.dump(instructions, open(os.path.join(args.out_dir, f"{batch_idx}.json"), 'w'))

    normalized_scores = {}
    correctness = []
    all_subsets = []
    row_subsets = []
    for instruction in instructions:
        requirements = instructions[instruction]["requirements"]
        if "scores" not in instructions[instruction]:
            continue
        total_sum = 100.0 * max(sum([req[1] for req in requirements]), 1e-8)
        normalized_scores[instruction] = {response: sum(scores) / total_sum for response, scores in instructions[instruction]["scores"].items()}
        if args.candidates_source == "rewardbench" and args.evaluate_rewardbench:
            chosen, rejected = responses[instruction]
            correctness.append(normalized_scores[instruction][chosen] > normalized_scores[instruction][rejected])
            individual_subsets = [example["subset"] for example in metadata[instruction]]
            all_subsets.extend(individual_subsets)
            row_subsets.append(individual_subsets)

    if args.candidates_source == "rewardbench" and args.evaluate_rewardbench:
        print(f"Overal Accuracy: {sum(correctness) / len(correctness)}\n")
        for subset in set(all_subsets):
            subset_correctness = [correct for correct, subs in zip(correctness, row_subsets) if subset in subs]
            print(f"Accuracy for subset {subset}: {sum(subset_correctness) / len(subset_correctness)}")