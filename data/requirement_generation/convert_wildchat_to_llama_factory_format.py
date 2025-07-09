import argparse
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--output-file", type=str, default="wildchat_processed.json")
parser.add_argument("--ground-truth", type=str, default=None)

def filter_out_row(row: dict) -> bool:
    conversation = row.get("conversation", [])
    if len(conversation) >= 1:
        content = conversation[0]["content"]
        if content.strip().startswith("""As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize. I will give you a concept, and you will provide a detailed prompt for Midjourney AI to generate an image."""):
            return True
    return False

def load_dataset():
    ds = load_dataset("allenai/WildChat-1M", split='train')
    two_turn_rows_english = []
    for r in ds:
        if r["language"].lower() != "english":
            continue
        elif len(r["conversation"]) == 2:
            two_turn_rows_english.append(r)
    del ds
    rows = two_turn_rows_english
    return rows

def convert_dataset(rows: list[dict], gt: dict[str, str] | None) -> list[dict]:
    processed_rows = []
    for row in rows:
        if len(row["conversation"]) == 2:
            instruction = row["conversation"][0]["content"]
            system = None
        elif len(row["conversation"]) == 3:
            raise NotImplementedError("Three-turn conversations not supported")
        else:
            num_conversations = len(row["conversation"])
            raise ValueError(f"Invalid number of conversations: {num_conversations}")

        if filter_out_row(row):
            continue

        if gt is not None:
            if instruction in gt:
                output = gt[instruction]
            else:
                print(f"Warning: Instruction not found in ground truth: {instruction}")
                continue
        else:
            output = row["conversation"][-1]["content"]
        instance = {
            "input": "",
            "system": "",
        }
        if system is not None:
            instance["system"] = system
        instance["instruction"] = instruction
        instance["output"] = output
        processed_rows.append(instance)
    return processed_rows

if __name__ == "__main__":
    args = parser.parse_args()
    ds = load_dataset()
    gt = {}
    if args.ground_truth is not None:
        lines = open(args.ground_truth, 'r').readlines()
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue
            try:
                row = json.loads(line)
            except:
                print(f"Error parsing line: {line}")
                continue
            if "response" not in row or "prompt" not in row:
                continue
            gt[row["prompt"]] = row["response"]

    processed_rows = convert_dataset(ds, gt)
    json.dump(processed_rows, open(args.output_file, 'w'), indent=4)
