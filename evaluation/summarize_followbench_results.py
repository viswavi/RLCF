import argparse
import csv
import json
import jsonlines

import sys

import os
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate a model on FollowBench')
parser.add_argument("model_name_escaped", type=str, default="gpt-3.5-turbo")

def analyze_errors(model_name_escaped, data_path = "data/FollowBench/data"):
    constraint_types = ['content', 'situation', 'style', 'format']
    home_dir = os.path.expanduser("~")
    checklist_finetuning_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    followbench_directory = os.pardir(__file__)
    paths = f"{checklist_finetuning_dir}/data/FollowBench/figures/figures_{model_name_escaped}/evaluation_result"

    level_ssrs = {1: [], 2: [], 3: [], 4: [], 5: []}
    level_hsrs = {1: [], 2: [], 3: [], 4: [], 5: []}
    results_level_one = {}
    csls = []
    for constraint in constraint_types:
        ssr = os.path.join(paths, f"{constraint}_ssr.csv")
        hsr = os.path.join(paths, f"{constraint}_hsr.csv")
        csl = os.path.join(paths, f"{constraint}_csl.csv")
        ssr_results = list(csv.DictReader(open(ssr)))[0]
        hsr_results = list(csv.DictReader(open(hsr)))[0]
        csl_results = list(csv.DictReader(open(csl)))[0]
        csls.append(float(csl_results["CSL"]))

        for level_str in ssr_results:
            if not level_str.startswith("level"):
                continue
            level_number = int(level_str.split(" ")[-1])
            value = float(ssr_results[level_str].split("%")[0].strip())
            level_ssrs[level_number].append(value)

        for level_str in hsr_results:
            if not level_str.startswith("level"):
                continue
            level_number = int(level_str.split(" ")[-1])
            value = float(hsr_results[level_str].split("%")[0].strip())
            level_hsrs[level_number].append(value)

    level_ssr_averages = {}
    level_ssr_averages = {level: sum(values) / len(values) for level, values in level_ssrs.items()}
    level_hsr_averages = {level: sum(values) / len(values) for level, values in level_hsrs.items()}

    print(f"Level SSR Averages for {model_name_escaped}:\n{json.dumps(level_ssr_averages, indent=4)}")
    print(f"Level HSR Averages for {model_name_escaped}:\n{json.dumps(level_hsr_averages, indent=4)}")
    print(f"CSL for {model_name_escaped}: {np.mean(csls)}")


if __name__ == "__main__":
    args = parser.parse_args()
    analysis = analyze_errors(args.model_name_escaped)