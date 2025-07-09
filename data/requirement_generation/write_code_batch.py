import argparse
import glob
import jsonlines
from tqdm import tqdm
import os
import torch
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("--requirements-dir", type=str, default="combined_wildchat_requirements")
parser.add_argument("--out-file", type=str, default="verification_functions.jsonl")
parser.add_argument("--sglang-model-name", type=str, default="Qwen/Qwen2.5-72B-Instruct")
parser.add_argument("--batch-size", type=int, default=10000)
parser.add_argument("--lines-start-idx", type=int, default=None)
parser.add_argument("--lines-end-idx", type=int, default=None)

parsing_prompt_template = """You are responsible for helping me verify whether or not responses satisfy various requirements. Given a natural language requirement, you will have to classify whether this can be converted to a Python program to automatically check it or whether it should be given to a human collaborator. Your human collaborator is a reliable and cheap expert, and you should trust them. Accordingly, only write code for verifying a constraint if you are very confident that this will exactly check the constraint. You should never make ANY approximations when verifying a constraint. If you feel that you must approximate the constraint in order to verify whether a response follows that constraint, let your human collaborator take care of it. You should ONLY generate code for requirements that are explicitly about syntax or format (e.g. punctuation, unicode characters used, number of paragraphs, shallow grammar, presence of some mandatory keyword specified by the prompt, etc). If there are many different ways to write an answer, you most likely should not generate code for it. If you are not sure, you should not generate code. You should only generate code if you are 100% sure that the constraint can be verified perfectly with a simple Python function.

When a constraint can be verified EXACTLY with a program, then return a Python function that verifies the constraint. This code should be contained within two sets of triple backquotes, ```. The Python function must return a boolean, and it should only use builtins/standard libraries in Python. If the constraint cannot be verified with a simple Python function (which means your human collaborator will handle the verification of this constraint), please return "NONE" and nothing else. The safest thing to do is to return "defer to human expert ####" 95% of the time. Now, let's go through a couple examples:

Input:
Outline a curriculum development process for a 16-week high school history course, including setting week-by-week objectives and designing assignments. Include two mid-term exams and a final exam. Provide a detailed grading criteria based on the assignments and exams you have designed.

Requirement:
Does the response specify the inclusion of two mid-term exams and a final exam

Verification Function:
defer to human expert ####
(there are multiple valid ways to describe this, and it is not a simple boolean check)

Input:
Given a programming language and the name of a function, write a command to show how to use the function.
Language: Python
Function: input

Requirement:
Is the generated text a single command?

Verification Function:
defer to human expert ####
(this is an underspecified requirement, given the original prompt)

Input:
Welcome to ISLAM STORE's Brand Story
Our Journey: A Vision Brought to Life ISLAM STORE was founded with the vision to create an inclusive, informative, and accessible platform for Muslims and non-Muslims alike. Our goal is to promote awareness and understanding of Islam while offering high-quality Islamic products.الترجمة للعربية

Requirement:
Does the generated text contain any Arabic?

Verification Function:
```python
def verify_requirement(text):
    # Arabic Unicode block range (0600-06FF)
    # Plus Extended Arabic (0750-077F)
    # Plus Arabic Presentation Forms (FB50-FDFF, FE70-FEFF)
    return any(('\u0600' <= char <= '\u06FF') or
               ('\u0750' <= char <= '\u077F') or
               ('\uFB50' <= char <= '\uFDFF') or
               ('\uFE70' <= char <= '\uFEFF') for char in text)
```
(it's straightforward to check if the text contains Arabic characters)

Input:
Create a dance routine consisting of 8-10 different moves that incorporate various elements of the body. The routine should include at least two distinct arm movements and two distinct leg movements and two head movements. Provide a brief description of each move and specify the duration of each, so that the entire routine lasts approximately 4 minutes.

Requirement:
Does the generated text specify the duration of each move?

Verification Function:
defer to human expert ####
(the duration of the move can be described in many different way)

Input:
write an email . use a serious tone. address the email to Dr Wu. Acknowledging that on November 20th our first call ended abruptly.  thank her for her willingness to provide services to me. remind Dr Wu of the needed note to provide my USPS employer. Sign off on this email with the words "I sincerely value our relationship and appreciate your time".

Requirement:
Is the email addressed to Dr. Wu?

Verification Function:
defer to human expert ####
(there are multiple valid ways to address an email)

Input:
Write an email to my manager, John Smith, to inform him that I will be taking a sick day tomorrow. The email should be polite and professional.

Requirement:
Is the generated text formatted as an email?

Verification Function:
defer to human expert ####
(there are multiple valid ways to format an email, and it is not a simple boolean check)

Input:
Write a lease for a commercial tenant. State that the monthly rent should be $10,000 per month, and specify that the lease duration should be exactly 2 years long.
    
Requirement:
Does the lease specify that the lease duration is exactly 2 years?

Verification Function:
defer to human expert ####
(there are multiple valid ways to format this - e.g. two years, 2 Yrs., two (2) years, etc)

Input:
{input}

Requirement:
{requirement}

Verification Function:"""


UNIVERSAL_PROMPT = """Does the response satisfy the following two criteria:
1) The response directly address the request without excessive or off-topic information not necessary for addressing the user's instruction?
2) The response should match the context and the instruction, whether it requires professionalism, friendliness, formality, or neutrality."""


MIDJOURNEY_PROMPT = """As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and you will provide a detailed prompt for Midjourney AI to generate an image."""

def batch_examples(examples, batch_size):
    return [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]

if __name__ == "__main__":
    args = parser.parse_args()

    req_prompt_pairs = {}

    if not os.path.exists(args.out_file):
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, 'w') as f:
            pass
    else:
        for triple in jsonlines.open(args.out_file):
            input = triple[0]
            req = triple[1]
            code = triple[2]
            req_prompt_pairs[(input, req)] = code

    files = glob.glob(os.path.join(args.requirements_dir, "*.jsonl"))

    model = LLM(
        model=args.sglang_model_name, tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16"
    )
    stop_tokens = ["USER:", "ASSISTANT:", "### Instruction:", "Input:", "Response:", "<|eot_id|>", "####", "<END>"]
    sampling_params = SamplingParams(temperature=0.0,
                                     repetition_penalty=1.0,
                                     top_p=0.9,
                                     max_tokens=300,
                                     n=1,
                                     stop=stop_tokens)

    # create jsonlines Writer
    line_counter = 0
    triples_to_process = []
    with jsonlines.Writer(open(args.out_file, mode='a')) as writer:
        for f in files:
            for row in jsonlines.open(f):
                line_counter += 1
                if (args.lines_start_idx is not None and line_counter < args.lines_start_idx) or (args.lines_end_idx is not None and line_counter > args.lines_end_idx):
                    continue
                input = row["instruction"]

                if input.strip().startswith(MIDJOURNEY_PROMPT):
                    continue

                requirements = [x[0] for x in row["requirements"]]
                for req in requirements:
                    if (input, req) in req_prompt_pairs:
                        continue
                    if req == UNIVERSAL_PROMPT:
                        code = "defer to human partner"
                        writer.write([input, req, code])
                    else:
                        parsing_prompt = parsing_prompt_template.format(input=input, requirement=req)
                        triples_to_process.append((input, req, parsing_prompt))

    batches = batch_examples(triples_to_process, args.batch_size)
    with jsonlines.Writer(open(args.out_file, mode='a')) as writer:
        for batch in batches:
            batch_prompts = [x[2] for x in batch]
            batch_responses = model.generate(batch_prompts, sampling_params=sampling_params)
            triples = []
            for i, response in enumerate(batch_responses):
                response_text = response.outputs[0].text
                if "```" in response_text and len(response_text.split("```")) > 1:
                    code = response_text.split("```")[1].strip()
                else:
                    code = response_text.strip()
                triples.append([batch[i][0], batch[i][1], code])
            writer.write_all(triples)