from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import argparse
import json
import jsonlines
import litellm
from openai import AzureOpenAI, OpenAI
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.models.model import _get_reward_model\

from transformers import AutoConfig, AutoModel

from tqdm import tqdm
import os
import torch
from typing import Optional
import uuid


import sys
sys.path.append("..")

from data.reward_baselines.launch_requirement_reward_server import RewardModelProxy

def create_prompt_with_llama3_format(prompt, system_message=None):
    if system_message is not None:
        formatted_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>".format(system_message)
    else:
        formatted_text = "<|begin_of_text|>"
    formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text

# tested name:
# TIGER-Lab/MAmmoTH2-8B
class vLLM_Model:
    def __init__(self,
                 name,
                 max_num_retries=3,
                 max_length = 1024,
                 dtype="bfloat16",
                 device="cuda:0",
                 temperature=0.6,
                 top_p=0.9,
                 lora=False,
                 special_cache_infix=None,
                 truncate_response=False,
                 n=1,
                 chat=True,
                 cache_dir=None):

        self.chat = chat
        self.name = name
        stop_tokens = ["USER:", "ASSISTANT:",  "### Instruction:", "<|eot_id|>", "####"]
        if name.startswith("openai-community/"):
            repetition_penalty=1.3
            print("OVERRIDE:    OpenAI model detected, setting temperature to 0.3")
            temperature=0.3
        else:
            repetition_penalty=1.0

        self.truncate_response = truncate_response

        self.sampling_params = SamplingParams(temperature=temperature,
                                         top_p=top_p,
                                         max_tokens=max_length,
                                         stop=stop_tokens,
                                         repetition_penalty=repetition_penalty,
                                         n=n)

        self.llm = LLM(model=name, tensor_parallel_size=torch.cuda.device_count(), 
              dtype=dtype, trust_remote_code=True, 
              enable_lora=True if lora else False)

        self.tokenizer = AutoTokenizer.from_pretrained(name)


        name_escaped = name.replace("-", "_").replace(".", "_").replace("/", "_")
        infix = f"_{special_cache_infix}" if special_cache_infix is not None else ""
        if cache_dir is not None:
            self.cache_filename = os.path.join(cache_dir, f"{name_escaped}{infix}_cache.jsonl")
        else:
            home_dir = os.path.expanduser("~")
            os.makedirs(f"{home_dir}/checklist_finetuning/caches/", exist_ok=True)
            self.cache_filename = f"{home_dir}/checklist_finetuning/caches/{name_escaped}{infix}_cache.jsonl"

        self.n = n
        if self.n > 1:
            self.cache_filename = self.cache_filename.replace(".jsonl", f"_n_{n}.jsonl")
        self.max_num_retries = max_num_retries
        if os.path.exists(self.cache_filename):
            self.cache = dict([tuple(row) for row in  jsonlines.open(self.cache_filename, "r")])
            self.cache_file = jsonlines.open(self.cache_filename, "a", flush=True)
        else:
            self.cache = {}
            self.cache_file = jsonlines.open(self.cache_filename, "w", flush=True)
        self.max_length = max_length
        self.device = device


    def respond(self, prompts: list[str], constraint_type: Optional[str] = None) -> list[litellm.ModelResponse]:
        # from litellm import completion
        # response = completion(model="gpt-3.5-turbo", messages=messages)

        prompt_to_response = {}

        uncached_prompts = []
        for prompt in prompts:
            if prompt in self.cache:
                response_dict = self.cache[prompt]
                litellm_response = litellm.ModelResponse.parse_raw(json.dumps(response_dict))
                if not isinstance(response_dict["choices"], list):
                    choice = response_dict["choices"]
                    message = litellm.Message.parse_raw(json.dumps(choice["message"]))
                    choices = litellm.Choices.parse_raw(json.dumps(choice))
                    choices.index = 0
                    choices.message = message
                    litellm_response.choices = [choices]
                else:
                    litellm_response.choices = []
                    for i, choice in enumerate(response_dict["choices"]):
                        message = litellm.Message.parse_raw(json.dumps(choice["message"]))
                        choices = litellm.Choices.parse_raw(json.dumps(choice))
                        choices.index = i
                        choices.message = message
                        litellm_response.choices.append(choices)

                prompt_to_response[prompt] = litellm_response
            else:
                uncached_prompts.append(prompt)

        if len(uncached_prompts) > 0:
            if self.name.startswith("meta-llama/Meta-Llama-3") or self.name.startswith("TIGER-Lab/MAmmoTH2-8B"):
                formatted_prompts = [create_prompt_with_llama3_format(prompt) for prompt in uncached_prompts]
                responses = self.llm.generate(formatted_prompts, self.sampling_params)
            else:
                texts = []
                if self.chat:
                    for prompt in uncached_prompts:
                        message = [
                            {"role": "user", "content": prompt}
                        ]
                        texts.append(self.tokenizer.apply_chat_template(
                            message,
                            tokenize=False,
                            add_generation_prompt=True
                        ))
                else:
                    texts = uncached_prompts
                responses = self.llm.generate(texts, self.sampling_params)

            messages = []
            for response in responses:
                truncated_responses = []
                for output in response.outputs:
                    if self.truncate_response:
                        DELIMITER="\nResponse:"
                        if DELIMITER in output.text:
                            truncated_response = output.text.split(DELIMITER)[-1].strip()
                        else:
                            truncated_response = output.text
                    else:
                        truncated_response = output.text
                    truncated_responses.append(truncated_response)

                if len(truncated_responses) == 1:
                    message = litellm.Message(role="assistant", content=truncated_responses[0], function_call=None, tool_calls=None)
                else:
                    message = []
                    for truncated_response in truncated_responses:
                        individual_message = litellm.Message(role="assistant", content=truncated_response, function_call=None, tool_calls=None)
                        message.append(individual_message)
                messages.append(message)

            litellm_responses = [litellm.ModelResponse(id="chatcmpl-" + str(uuid.uuid4()), model=self.name) for _ in uncached_prompts]
            for i in range(len(messages)):
                if isinstance(messages[i], litellm.Message) or isinstance(messages[i], list):
                    if isinstance(messages[i], litellm.Message):
                        litellm_responses[i].choices = [litellm.Choices(message=messages[i], index=0)]
                    elif isinstance(messages[i], list):
                        litellm_responses[i].choices = []
                        for j in range(len(messages[i])):
                            choice = litellm.Choices(message=messages[i][j], index=j)
                            litellm_responses[i].choices.append(choice)
                    prompt = uncached_prompts[i]
                    cache_record = [prompt, litellm_responses[i].json()]
                    prompt_to_response[prompt] = litellm_responses[i]
                    self.cache[prompt] = litellm_responses[i].json()
                    self.cache_file.write(cache_record)
                else:
                    raise ValueError(f"Unsupported response type: {type(messages[i])}")

        responses = [prompt_to_response[prompt] for prompt in prompts]

        return responses
