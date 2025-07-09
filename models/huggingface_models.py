from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import jsonlines
import litellm
import openai
from openai import AzureOpenAI, OpenAI
import os
from typing import Optional
import uuid


class HFModel:
    def __init__(self, name, max_num_retries=3, max_length = 1024, device="cuda:0"):
        self.name = name
        name_escaped = name.replace("-", "_").replace(".", "_").replace("/", "_")
        self.cache_filename = f"{name_escaped}_cache.jsonl"
        self.max_num_retries = max_num_retries
        if os.path.exists(self.cache_filename):
            self.cache = dict([tuple(row) for row in  jsonlines.open(self.cache_filename, "r")])
            self.cache_file = jsonlines.open(self.cache_filename, "a", flush=True)
        else:
            self.cache = {}
            self.cache_file = jsonlines.open(self.cache_filename, "w", flush=True)
        self.max_length = max_length
        self.device = device

        self.trained_model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map="auto",
            trust_remote_code=True
        )
        self.trained_tokenizer = AutoTokenizer.from_pretrained(
            self.name,
            add_bos_token=True,
            trust_remote_code=True,
        )
        self.trained_tokenizer.pad_token = self.trained_tokenizer.eos_token


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
                    message = litellm.Message.parse_raw(json.dumps(response_dict["choices"]["message"]))
                    choices = litellm.Choices.parse_raw(json.dumps(response_dict["choices"]))
                else:
                    message = litellm.Message.parse_raw(json.dumps(response_dict["choices"][0]["message"]))
                    choices = litellm.Choices.parse_raw(json.dumps(response_dict["choices"][0]))
                choices.message = message
                litellm_response.choices = [choices]
                prompt_to_response[prompt] = litellm_response
            else:
                uncached_prompts.append(prompt)

        if len(uncached_prompts) > 0:
            if len(uncached_prompts) == 1:
                tokenized_input = self.trained_tokenizer(uncached_prompts[0], return_tensors='pt', max_length=self.max_length,  truncation=True).to(self.device)
                output_ids = self.trained_model.generate(**tokenized_input, do_sample=False, max_length=self.max_length+100, early_stopping=True, length_penalty=-1)
                self.trained_tokenizer.batch_decode(output_ids[:, tokenized_input.input_ids.shape[1]:])[0]
                responses = self.trained_tokenizer.batch_decode(output_ids[:, tokenized_input.input_ids.shape[1]:], skip_special_tokens=True)
            else:
                tokenized_input = self.trained_tokenizer(uncached_prompts, return_tensors='pt', max_length=self.max_length,  padding='max_length', truncation=True).to(self.device)
                # Generate outputs for the current batch
                output_ids = self.trained_model.generate(**tokenized_input, max_length=self.max_length+100, early_stopping=True, length_penalty=-1, temperature=0.6, top_p=0.9)
                responses = self.trained_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            messages = []
            for response in responses:
                message = litellm.Message(role="assistant", content=response, function_call=None, tool_calls=None)
                messages.append(message)

            litellm_responses = [litellm.ModelResponse(id="chatcmpl-" + str(uuid.uuid4()), model=self.name) for _ in uncached_prompts]
            for i in range(len(messages)):
                litellm_responses[i].choices = [litellm.Choices(message=messages[i])]
                prompt = uncached_prompts[i]
                cache_record = [prompt, litellm_responses[i].json()]
                prompt_to_response[prompt] = litellm_responses[i]
                self.cache[prompt] = litellm_responses[i].json()
                self.cache_file.write(cache_record)

        responses = [prompt_to_response[prompt] for prompt in prompts]

        return responses