import json
import jsonlines
import litellm
import openai
from openai import OpenAI
import os
from typing import Optional
import uuid

class Message():
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content
        }

class Response:
    def __init__(self, message):
        self.message = message

class OpenAIModel:
    def __init__(self, name, max_num_retries=3):
        assert name in ["neulab/gpt-4o-mini-2024-07-18"]
        self.name = name
        self.name_escaped = name.replace("-", "_").replace(".", "_").replace("/", "_")
        self.cache_filename = f"{self.name_escaped}_cache.jsonl"
        self.max_num_retries = max_num_retries
        self.client = OpenAI(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url="https://cmu.litellm.ai",
        )
        if os.path.exists(self.cache_filename):
            self.cache = dict([tuple(row) for row in  jsonlines.open(self.cache_filename, "r")])
            self.cache_file = jsonlines.open(self.cache_filename, "a", flush=True)
        else:
            self.cache = {}
            self.cache_file = jsonlines.open(self.cache_filename, "w", flush=True)

    def respond(self, prompts: list[str], constraint_type: Optional[str] = None) -> list[litellm.ModelResponse]:
        # from litellm import completion
        # response = completion(model="gpt-3.5-turbo", messages=messages)
        responses = []
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
            else:
                if self.name.split("/")[-1] in ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "gpt-3.5-turbo"]:
                    client = OpenAI(api_key=os.environ.get("LITELLM_API_KEY"),
                                    base_url="https://cmu.litellm.ai",
                                    )
                else:
                    raise ValueError(f"Model {self.name} not supported by LiteLLM")


                success = False
                retry_counter = 0
                error_message = None
                while not success:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.name,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        success = True
                    except openai.BadRequestError as e:
                        if retry_counter == self.max_num_retries:
                            print(f"Failed after {self.max_num_retries} retries!")
                            error_message = e.message
                            break
                        else:
                            print("Retrying")
                        retry_counter += 1

                if error_message is not None:
                    message = litellm.Message(role="assistant", content=error_message, function_call=None, tool_calls=None)
                else:
                    response_dict = response.choices[0]
                    message = litellm.Message(role=response_dict.message.role, content=response_dict.message.content, function_call=None, tool_calls=None)
                    if response_dict.message.content is None:
                        message = litellm.Message(role=response_dict.message.role, content="None", function_call=None, tool_calls=None)

                litellm_response = litellm.ModelResponse(id="chatcmpl-" + str(uuid.uuid4()), model=self.name)
                litellm_response.choices = [litellm.Choices(message=message)]
                cache_record = [prompt, litellm_response.json()]
                self.cache[prompt] = litellm_response.json()
                self.cache_file.write(cache_record)
            responses.append(litellm_response)
        return responses