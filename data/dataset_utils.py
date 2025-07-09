import asyncio
import copy
from datasets import load_dataset
import json
import jsonlines
import openai
import os
import pickle
import random
import time
from tqdm import tqdm

import litellm
from models.openai_models import OpenAIModel
from models.vllm import vLLM_Model, vLLM_model_with_reranking

from data.FollowBench.code.rule_based_evaluation import save_evaluate_example_constraint, save_csl_example_constraint
from data.FollowBench.code.gpt4_based_evaluation import acquire_discriminative_eval_input, save_discriminative_evaluation, save_csl_evaluation
from data.FollowBench.code.llm_eval import get_json_list, get_eval

from data.FollowBench.code.openai_model_inference import inference, convert_to_api_input
from InfoBench.evaluation import SYS_MSG

def batch_examples(examples, batch_size):
    return [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]

class InFoBenchOutputWithCriteria:
    def __init__(self,
                 instruction,
                 input,
                 output,
                 decomposed_questions,
                 id,
                 subset,
                 eval=None):
        self.instruction = instruction
        self.input = input
        self.output = output
        self.decomposed_questions = decomposed_questions
        self.id = id
        self.subset = subset
        self.eval = eval

def generation_for_infobench(model, batch_size=1):
    dataset = load_dataset("kqsong/InFoBench")
    train_set = dataset["train"]
    responses_by_category = {}
    if isinstance(model, vLLM_Model) or isinstance(model, vLLM_model_with_reranking):
        batch_iterator = [train_set]
    else:
        batch_iterator = tqdm(batch_examples(train_set, batch_size))

    gpt4_cache = {}
    gpt4_cache_file = "gpt-4_cache.jsonl"
    if isinstance(model, vLLM_Model) or isinstance(model, vLLM_model_with_reranking):
        assert os.path.exists(gpt4_cache_file)
    for row in jsonlines.open(gpt4_cache_file):
        if isinstance(row[1]["choices"], list):
            gpt4_cache[row[0]] = row[1]["choices"][0]["message"]["content"]
        else:
            gpt4_cache[row[0]] = row[1]["choices"]["message"]["content"]

    formatted_rows = []

    for batch in batch_iterator:

        prompts = []

        for row in batch:
            instruction = row["instruction"]
            input = row["input"]
            prompt = f"{instruction}\n{input}:"
            prompts.append(prompt)

        try:
            responses = model.respond(prompts)
            response_texts = [response.choices[0].message.content for response in responses]
        except openai.BadRequestError as e:
            response_texts = [e.message for _ in prompts]

        for row, prompt, response_text in zip(batch, prompts, response_texts):
            category = row["subset"]
            if category not in responses_by_category:
                responses_by_category[category] = []
            responses_by_category[category].append(InFoBenchOutputWithCriteria(
                instruction=row["instruction"],
                input=row["input"],
                output=response_text,
                decomposed_questions=row["decomposed_questions"],
                id=row["id"],
                subset=row["subset"]
            ))
            row_input = row["input"]
            row_instruction = row["instruction"]
            formatted_prompt = f"System: {row_instruction}\nUser: {row_input}" if len(row_input.split()) > 0 else row_instruction
            if prompt in gpt4_cache:
                gpt4_response = gpt4_cache[prompt]
                conversation = [
                    {"from": "human",
                     "content": formatted_prompt},
                    {"from": "gpt",
                     "content": gpt4_response},
                ]
                row = {
                    "conversations": conversation,
                    "response": response_text,
                    "prompt": formatted_prompt,
                }
                formatted_rows.append(row)
            else:
                print("=====================\nPrompt not found in cache!\n{prompt}\n=====================")
    model_name_escaped = model.name.replace("-", "_").replace(".", "_").replace("/", "_")
    if isinstance(model, vLLM_model_with_reranking):
        model_name_escaped += "_reranking"

    pickle.dump(responses_by_category, open(f"infobench_responses_{model_name_escaped}.pkl", "wb"))


    output_path = f"infobench_outputs/infobench_{model_name_escaped}_response_data.jsonl"
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(formatted_rows)

    return responses_by_category

def bool_ratio_infobench(responses_with_evals):
    "Calculate true false ratio for eval results"
    count = {"true":0, "false":0}
    for entry in responses_with_evals:
        if entry.eval is None:
            print("Wrong output")
            print(entry.id)
        if len(entry.decomposed_questions) != len(entry.eval):
            print("Wrong length")
            print(entry.id)
        if None in entry.eval:
            print("None in eval")
            print(entry.id)
        
        for eva_value in entry.eval:
            if eva_value:
                count["true"] += 1
            else:
                count["false"] += 1
    return count

def evaluate_infobench(model, responses_by_type, temperature=0, max_num_retries=3, cache_dir="infobench_eval_cache"):

    eval_client = openai.OpenAI(api_key=os.environ.get("LITELLM_API_KEY"),
                                base_url="https://cmu.litellm.ai",
                               )
    os.makedirs(cache_dir, exist_ok=True)
    cachefile = os.path.join(cache_dir, f"gpt4_eval_cache.jsonl")

    if os.path.exists(cachefile):
        cache_rows = list(jsonlines.open(cachefile, 'r'))
        cache = dict([tuple(row) for row in cache_rows])
        cache_writer = jsonlines.open(cachefile, 'w', flush=True)
        cache_writer.write_all(cache_rows)
    else:
        cache = {}
        cache_writer = jsonlines.open(cachefile, 'w', flush=True)
    
    # print(f"--------Instance {entry['id']}--------")
    all_responses = [response for responses in responses_by_type.values() for response in responses]
    for entry in tqdm(all_responses):

        cache_key = f"{entry.instruction} {entry.input} {entry.output}"
        if cache_key in cache:
            bool_results = cache[cache_key]
        else:
            bool_results = []
            input = entry.input
            output = entry.output

            message = []
            answer = ""

            for question in entry.decomposed_questions:
                if len(message) == 0:
                    if input:
                        content =  f"{SYS_MSG}\n\nInput:\n\"{input}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
                    else:
                        content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
                else:
                    content = f"{question}\n"
                message.append({"role": "user", "content": content})
                # create a chat completion
                success = False
                early_stop = False
                retry_counter = 0
                while not success:
                    try:
                        completion = eval_client.chat.completions.create(
                            model="neulab/gpt-4o-2024-05-13",
                            messages=message,
                            temperature=temperature,
                        )

                        generation = completion.choices[0].message.content
                        message.append(
                            {"role": "assistant", "content": generation})
                        # check if generation is yes or no
                        if generation.lower().startswith("yes") or generation.lower().startswith("no"):
                            if generation.lower().startswith("yes"):
                                answer += "Yes\n"
                            else:
                                answer += "No\n"
                        else:
                            if "YES" in generation and "NO" not in generation:
                                answer += "Yes\n"
                            elif "YES" not in generation and "NO" in generation:
                                answer += "No\n"
                            else:
                                for msg in message:
                                    print(msg['content'])
                                print("NO YES or NO answer!" + generation)
                                answer += "None\n"
                                early_stop = True
                                break
                        success = True
                    except Exception as e:
                        print("ERROR!")
                        print(e)
                        if retry_counter == max_num_retries:
                            print(f"Failed after {max_num_retries} retries!")
                            answer += "None\n"
                            early_stop = True
                            continue
                        else:
                            print("Retry!")
                            time.sleep(20)
                        retry_counter += 1

                # when no answer occurs, break the loop and continue to next instance
                if early_stop:
                    break

            answer = answer[:-1]
            # save eval results as List[bool]
            for i in answer.split('\n'):
                if i == "Yes":
                    bool_results.append(True)
                elif i == "No":
                    bool_results.append(False)
                else:
                    bool_results.append(None)
            cache_writer.write([cache_key, bool_results])

        entry.eval = bool_results

    easy_responses = [r for r in all_responses if r.subset == "Easy_set"]
    hard_responses = [r for r in all_responses if r.subset == "Hard_set"]

    easy_bool_ratio_table = bool_ratio_infobench(easy_responses)

    hard_bool_ratio_table = bool_ratio_infobench(hard_responses)
    overall_bool_ratio_table = bool_ratio_infobench(all_responses)

    easy_drfr = easy_bool_ratio_table['true']/sum(easy_bool_ratio_table.values())
    hard_drfr = hard_bool_ratio_table['true']/sum(hard_bool_ratio_table.values())
    overall_drfr = overall_bool_ratio_table['true']/sum(overall_bool_ratio_table.values())

    print(f"DRFR of {model.name} on the Easy set: {round(easy_drfr, 4)}")
    print(f"DRFR of {model.name} on the Hard set: {round(hard_drfr, 4)}")
    print(f"DRFR of {model.name} on the Overall set: {round(overall_drfr, 4)}")

    drfrs = {"easy": easy_drfr, "hard": hard_drfr, "overall": overall_drfr}
    return drfrs

def generation_for_ifeval(model, batch_size=1, think_step_by_step=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.dirname(os.path.dirname(current_dir))
    data_file_path = os.path.join(data_path, "google-research/instruction_following_eval/data/input_data.jsonl")

    dataset = list(jsonlines.open(data_file_path))
    if think_step_by_step:
        prompts = [format_prompt_template_with_thinking(row['prompt']) for row in dataset]
    else:
        prompts = [row['prompt'] for row in dataset]
    responses = model.respond(prompts)

    if think_step_by_step:
        responses_just_answers = []
        for r in responses:
            if "<ANSWER>" in r.choices[0].message.content:
                truncated_response = r.choices[0].message.content.split("<ANSWER>")[1].strip()
            else:
                truncated_response = r.choices[0].message.content
            r.choices[0].message.content = truncated_response
            responses_just_answers.append(r)
        responses = responses_just_answers

    responses_by_category = dict(zip(prompts, responses))
    model_name_escaped = model.name.replace("-", "_").replace(".", "_").replace("/", "_")
    if think_step_by_step:
        model_name_escaped += "_thinking"
    elif isinstance(model, vLLM_model_with_reranking):
        model_name_escaped += "_reranking"

    output_path = f"{model_name_escaped}_response_data.jsonl"
    with jsonlines.open(output_path, 'w') as writer:
        for prompt, response in responses_by_category.items():
            writer.write({"prompt": prompt, "response": response.choices[0].message.content})
    return responses_by_category

def evaluate_ifeval(model, responses_by_type):
    raise NotImplementedError

def generation_for_followbench(model, batch_size=1, data_path = "data/FollowBench/data"):
    '''
    model must have at least a `respond` method and a `name` attribute
    '''
    constraint_types = ['content', 'situation', 'style', 'format']

    model_name_escaped = model.name.replace("-", "_").replace(".", "_").replace("/", "_")

    api_input_path = f"data/FollowBench/api_input_{model_name_escaped}"
    api_output_path = f"data/FollowBench/api_output_{model_name_escaped}"
    os.makedirs(api_input_path, exist_ok=True)
    os.makedirs(api_output_path, exist_ok=True)
    temperature = 0.0
    repetition_penalty = 1.0
    max_new_tokens = 2048
    debug = True

    for constraint_type in constraint_types:
        convert_to_api_input(data_path=data_path,
                             api_input_path=api_input_path,
                             constraint_type=constraint_type)

    responses_by_category = {}

    for constraint_type in constraint_types:

        data = []
        with open(os.path.join(api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        responses_by_category[constraint_type] = []

        if isinstance(model, vLLM_Model):
            batch_iterator = [data]
        else:
            batch_iterator = tqdm(batch_examples(data, batch_size))

        for batch in tqdm(batch_iterator):
            # Build the prompt with a conversation template
            msg_batch = [row['prompt_new'] for row in batch]

            responses = model.respond(msg_batch, constraint_type)
            assert len(responses) == len(batch)
            for i, response in enumerate(responses):
                response_dict = response.choices[0]

                if hasattr(response_dict.message, 'to_dict'):
                    batch[i]['choices'] = [{'message': response_dict.message.to_dict()}]
                else:
                    batch[i]['choices'] = [{'message': response_dict.message.json()}]
                responses_by_category[constraint_type].append(batch[i]['choices'][0]['message'])

        model_name_escaped = model.name.replace("-", "_").replace(".", "_").replace("/", "_")
        # save file
        with open(os.path.join(api_output_path, f"{model_name_escaped}_{constraint_type}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for d in data:
                output_file.write(json.dumps(d) + "\n")
    pickle.dump(responses_by_category, open(f"followbench_responses_{model_name_escaped}.pkl", "wb"))
    return responses_by_category

def evaluate_followbench(model, responses_by_type, data_path = "data/FollowBench/data"):

    max_tokens = 1024

    model_name_escaped = model.name.replace("-", "_").replace(".", "_").replace("/", "_")
    api_output_path = f"data/FollowBench/api_output_{model_name_escaped}"
    os.makedirs(api_output_path, exist_ok=True)



    constraint_types = ['content', 'situation', 'style', 'format']
    gpt4_discriminative_eval_input_path = f"{model_name_escaped}_discriminative_eval_input"
    data_gpt4_discriminative_eval_input_path = f"data_{model_name_escaped}_discriminative_eval_input"
    gpt4_discriminative_eval_output_path = f"{model_name_escaped}_discriminative_eval_output"

    ### convert api_output to LLM_based_eval_input


    for constraint_type in constraint_types:
        acquire_discriminative_eval_input(
                                        data_path=data_path,
                                        api_output_path=api_output_path,
                                        constraint_type=constraint_type,
                                        model_name=model.name.replace("-", "_").replace(".", "_").replace("/", "_"),
                                        data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path,
                                        gpt4_discriminative_eval_input_path=gpt4_discriminative_eval_input_path
                                        )

    ### LLM-based evaluation
    if not os.path.exists(gpt4_discriminative_eval_output_path):
        os.makedirs(gpt4_discriminative_eval_output_path)

    for constraint_type in constraint_types:

        cachefile = f"cache/gpt4_eval_cache_{constraint_type}.json"
        os.makedirs('cache', exist_ok=True)
        if os.path.exists(cachefile):
            cache = json.load(open(cachefile, 'r'))
        else:
            cache = {}

        eval_input = get_json_list(os.path.join(gpt4_discriminative_eval_input_path, "{0}_{1}_constraint.jsonl".format(model_name_escaped, constraint_type)))

        with open(os.path.join(gpt4_discriminative_eval_output_path, "{0}_{1}_constraint.jsonl".format(model_name_escaped, constraint_type)), 'w') as output_file:
            for idx in tqdm(range(len(eval_input))):
                if eval_input[idx]['prompt_new'] in cache:
                    response = cache[eval_input[idx]['prompt_new']]
                else:
                    response = get_eval(eval_input[idx]['prompt_new'], max_tokens)
                    cache[eval_input[idx]['prompt_new']] = response
                output_file.write(json.dumps({'prompt_new': eval_input[idx]['prompt_new'], "choices": [{"message": {"content": response}}]}) + '\n')
        json.dump(cache, open(cachefile, 'w'))



    model_paths = [model_name_escaped]
    data_gpt4_discriminative_eval_input_path = f"data_{model_name_escaped}_discriminative_eval_input"
    gpt4_discriminative_eval_output_path = f"{model_name_escaped}_discriminative_eval_output"

    figures_dir = f"data/FollowBench/figures/figures_{model_name_escaped}/"
    evaluation_result_path = os.path.join(figures_dir, "evaluation_result")
    if not os.path.exists(evaluation_result_path):
        os.makedirs(evaluation_result_path)

    ### LLM-based evaluation
    for constraint_type in constraint_types:
        save_discriminative_evaluation(
                                        data_path=data_path,
                                        api_output_path=api_output_path,
                                        data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path,
                                        gpt4_discriminative_eval_output_path=gpt4_discriminative_eval_output_path,
                                        constraint_type=constraint_type,
                                        model_names=model_paths,
                                        evaluation_result_path=evaluation_result_path
                                    )

        save_csl_evaluation(
                            data_path=data_path,
                            api_output_path=api_output_path,
                            data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path,
                            gpt4_discriminative_eval_output_path=gpt4_discriminative_eval_output_path,
                            constraint_type=constraint_type,
                            model_names=model_paths,
                            evaluation_result_path=evaluation_result_path
                            )

    print(f"\nEvaluation finished!\nThe evaluation results have been saved in '{evaluation_result_path}'.")

def generation_for_wildchat(model,
                             sampled_size=-1,
                             num_batches=1,
                             special_cache_infix=None,
                             dataset_start_idx=None,
                             dataset_end_idx=None,
                             think_step_by_step=False):
    '''
    model must have at least a `respond` method and a `name` attribute
    '''
    ds = load_dataset("allenai/WildChat-1M", split='train')
    two_turn_rows_english = []
    for r in ds:
        if r["language"].lower() != "english":
            continue
        elif len(r["conversation"]) == 2:
            two_turn_rows_english.append(r)
    del ds
    rows = two_turn_rows_english

    print(f"Loaded {len(rows)} rows")

    if dataset_start_idx is None:
        dataset_start_idx = 0

    if dataset_end_idx is None:
        dataset_end_idx = len(rows)

    if dataset_start_idx is not None or dataset_end_idx is not None:
        rows = rows[dataset_start_idx:dataset_end_idx]
        print(f"Restricting to {len(rows)} rows, between indices {dataset_start_idx} and {dataset_end_idx}")

    model_name_escaped = model.name.replace("-", "_").replace(".", "_").replace("/", "_")

    infix = f"_{special_cache_infix}" if special_cache_infix is not None else ""
    os.makedirs("wildchat_responses", exist_ok=True)
    output_path = f"wildchat_responses/{model_name_escaped}{infix}_wildchat_response.jsonl"

    prompts = []
    for r in rows:
        if think_step_by_step:
            prompts.append(format_prompt_template_with_thinking(r["conversation"][0]["content"]))
        else:
            prompts.append(r["conversation"][0]["content"])
    batch_size = len(prompts) // num_batches

    # Handle multiple candidate responses

    assert isinstance(model, vLLM_Model), "Only VLLM is supported for wildchat"

    outputs_exist = False
    if model.n == 1:
        outputs_exist = os.path.exists(output_path)
    else:
        outputs_exist = True
        for file_idx in range(model.n):
            if not os.path.exists(output_path.replace(".jsonl", f"_sample_{file_idx}.jsonl")):
                outputs_exist = False
                break

    if outputs_exist:
        existing_prompts = set()
        if model.n == 1:
            for row in jsonlines.open(output_path, 'r'):
                existing_prompts.add(row["prompt"])
        else:
            prompt_to_response_counter = {}
            all_prompts = set()
            for file_idx in range(model.n):
                for row in jsonlines.open(output_path.replace(".jsonl", f"_sample_{file_idx}.jsonl"), 'r'):
                    if row["prompt"] not in prompt_to_response_counter:
                        prompt_to_response_counter[row["prompt"]] = 0
                    prompt_to_response_counter[row["prompt"]] += 1
                    all_prompts.add(row["prompt"])
            existing_prompts = {p for p in all_prompts if prompt_to_response_counter[p] >= model.n}
        filtered_rows = []
        filtered_prompts = []
        print("Starting to filter out prompts that have already been generated")
        for row, prompt in tqdm(zip(rows, prompts)):
            if prompt not in existing_prompts:
                if len(model.tokenizer.tokenize(prompt)) > min(model.max_length, model.tokenizer.model_max_length):
                    continue
                filtered_rows.append(row)
                filtered_prompts.append(prompt)
        rows = filtered_rows
        prompts = filtered_prompts
        print(f"Filtered out {len(existing_prompts)} prompts")
    else:
        # touch the file
        if model.n == 1:
            outfiles = [output_path]
        else:
            outfiles = []
            for file_idx in range(model.n):
                outfiles.append(output_path.replace(".jsonl", f"_sample_{file_idx}.jsonl"))
        for outfile in outfiles:
            with open(outfile, 'w'):
                pass

    rows_to_process = list(zip(rows, prompts))
    print(f"{len(rows_to_process)} rows to process")

    batches = batch_examples(rows_to_process, batch_size)

    if model.n == 1:
        writers = [jsonlines.open(output_path, 'a')]
    else:
        writers = []
        for writer_idx in range(model.n):
            outfile = output_path.replace(".jsonl", f"_{writer_idx}.jsonl")
            writers.append(jsonlines.open(outfile, 'a'))

    for batch in batches:
        batch_rows, batch_prompts = zip(*batch)

        batch_responses = model.respond(batch_prompts)

        for r, prompt, response in zip(batch_rows, batch_prompts, batch_responses):
            sampled_response_outputs = []
            responses = [choice.message.content for choice in response.choices]
            assert len(responses) == model.n
            for file_idx, sampled_response_text in enumerate(responses):
                r_copy = copy.deepcopy(r)
                r_copy["response"] = sampled_response_text
                r_copy["prompt"] = prompt
                r_copy["timestamp"] = r_copy["timestamp"].isoformat()
                turns = []
                for turn in r_copy["conversation"]:
                    if turn.get("timestamp") is not None:
                        turn["timestamp"] = turn["timestamp"].isoformat()
                    turns.append(turn)
                r_copy["conversation"] = turns
                sampled_response_outputs.append(r_copy)
                writers[file_idx].write(r_copy)

    for writer in writers:
        writer.close()

async def call_model(prompt: str, openai_model = "openai/neulab/gpt-4o-mini-2024-07-18"):
    try:
        response = await litellm.acompletion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url="https://cmu.litellm.ai",
            model=openai_model,
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ],
            temperature=0.0
        )
        return {"prompt": prompt, "response": response['choices'][0]['message']['content']}
    except litellm.exceptions.APIError as e:
        print(f"API error: {e}")
        return None

async def process_prompts(prompts, batch_size=20):
    batches = batch_examples(prompts, batch_size)
    responses = []
    for batch in tqdm(batches):
        batch_responses = await asyncio.gather(*[call_model(prompt) for prompt in batch])
        responses.extend(batch_responses)
    return responses

def generation_for_rewardbench(model, batch_size=20):
    rows = load_dataset("allenai/reward-bench", split='filtered')
    prompts = []
    for row in rows:
        prompts.append(row["prompt"])

    model_name_escaped = model.name.replace("-", "_").replace(".", "_").replace("/", "_")
    os.makedirs("rewardbench_responses", exist_ok=True)
    output_path = f"rewardbench_responses/{model_name_escaped}_rewardbench_response.jsonl"
    if os.path.exists(output_path) and len(list(jsonlines.open(output_path, 'r'))) == len(prompts):
        response_contens = []
        for i, row in enumerate(jsonlines.open(output_path, 'r')):
            assert row["prompt"] == prompts[i]
            response_contens.append(row["response"])
    else:
        print("Generating responses for RewardBench prompts")
        responses = model.respond(prompts)
        response_contens = [response.choices[0].message.content for response in responses]
        print("Done generating responses for RewardBench prompts")


    gpt_4_cache_file = f"/rewardbench_responses/gpt_4o_mini_rewardbench_response.jsonl"
    if os.path.exists(gpt_4_cache_file):
        gpt_4_cache = {}
        for row in jsonlines.open(gpt_4_cache_file, 'r'):
            gpt_4_cache[row["prompt"]] = row["response"]
    else:
        gpt_4_cache = {}
        # create an empty cache file
        with jsonlines.open(gpt_4_cache_file, 'w', flush=True):
            pass

    prompts_to_run = [p for p in prompts if p not in gpt_4_cache]
    if len(prompts_to_run) > 0:
        print("Starting to run gpt-4o-mini on the prompts")
        responses = asyncio.run(process_prompts(prompts_to_run, batch_size=batch_size))

        for resp in responses:
            if resp is not None:
                gpt_4_cache[resp["prompt"]] = resp["response"]
                with jsonlines.open(gpt_4_cache_file, 'a') as cache_writer:
                    cache_writer.write(resp)

    with jsonlines.open(output_path, 'w') as writer:
        for r, prompt, response_content in tqdm(zip(rows, prompts, response_contens)):
            assert prompt in gpt_4_cache
            gpt4_response = gpt_4_cache.get(prompt, "N/A")
            if gpt4_response == "N/A":
                print(f"Prompt: {prompt} not in cache!")

            conversation = [
                {"from": "human",
                    "content": prompt},
                {"from": "gpt",
                    "content": gpt4_response},
            ]
            r["conversations"] = conversation
            r["response"] = response_content
            assert r["prompt"] == prompt
            writer.write(r)

    responses_by_category = dict(zip(prompts, responses))
    return responses_by_category


def generation_for_alpacaeval(model, batch_size=20):

    eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    instructions = [example["instruction"] for example in eval_set]
    outputs = []
    responses = model.respond(instructions)
    for i, example in enumerate(eval_set):
        example["output"] = responses[i].choices[0].message.content
        example["generator"] = model.name
        outputs.append(example)

    model_name_escaped = model.name.replace("-", "_").replace(".", "_").replace("/", "_")
    json.dump(outputs, open(f"alpacaeval_responses_{model_name_escaped}.json", "w"), indent=4)

    return {"all": outputs}