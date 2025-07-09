import argparse
from datasets import load_dataset
import json
import jsonlines
import os
import pickle
import sys
from tqdm import tqdm


supported_models = ["Qwen/Qwen2.5-0.5B",
                    "Qwen/Qwen2.5-1.5B",
                    "Qwen/Qwen2.5-3B",
                    "Qwen/Qwen2.5-7B"]

parser = argparse.ArgumentParser()
parser.add_argument("--input-responses-dir", type=str, default="wildchat_responses")
parser.add_argument("--combined-responses-dir", type=str, default="combined_wildchat_responses")

if __name__ == "__main__":
    args = parser.parse_args()
    input_responses_dir = args.input_responses_dir
    all_conversations = []
    for model in supported_models:
        model_name_escaped = model.replace("-", "_").replace(".", "_").replace("/", "_")
        model_responses_path = f"{input_responses_dir}/{model_name_escaped}_wildchat_response.jsonl"
        conversations = {}
        for r in tqdm(jsonlines.open(model_responses_path, "r")):
            conv_str = (r["conversation"][0]["content"], r["conversation"][1]["content"])
            conv = r["conversation"]
            for i, c in enumerate(conv):
                if i == 0:
                    c["from"] = "human"
                else:
                    c["from"] = "gpt"
                turn_message = c["content"]
                del c["content"]
                c["value"] = turn_message
                conv[i] = c
            r["conversations"] = conv
            del r["conversation"]
            conversations[conv_str] = r
        all_conversations.append(conversations)

    ds = load_dataset("allenai/WildChat-1M", split='train')
    two_turn_rows_english = []
    for r in ds:
        if r["language"].lower() != "english":
            continue
        elif len(r["conversation"]) == 2:
            two_turn_rows_english.append(r)
    del ds
    rows = two_turn_rows_english

    all_responses_rows = []
    for r in tqdm(rows):

        r["timestamp"] = r["timestamp"].isoformat()
        turns = []
        for turn in r["conversation"]:
            if turn.get("timestamp") is not None:
                turn["timestamp"] = turn["timestamp"].isoformat()
            turns.append(turn)
        r["conversations"] = turns
        del r["conversation"]

        num_hits = 0
        responses = {}
        skip_row = False
        conversation_str = (r["conversations"][0]["content"], r["conversations"][1]["content"])
        for model, conversations in zip(supported_models, all_conversations):
            try:
                responses[model] = conversations[conversation_str]["response"]
            except:
                skip_row = True
                break
            if conversation_str in conversations:
                num_hits += 1
        if skip_row:
            continue
        r["all_responses"] = responses
        all_responses_rows.append(r)


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

    response_chunks = split_list(all_responses_rows, 24)
    responses_dir = args.combined_responses_dir
    os.makedirs(responses_dir, exist_ok=True)

    for i, chunk in tqdm(enumerate(response_chunks)):
        out = open(os.path.join(responses_dir, f"{i}.json"), 'w')
        out.write(json.dumps(chunk))  
        out.close()
