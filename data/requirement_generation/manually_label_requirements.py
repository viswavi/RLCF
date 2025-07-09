import argparse
import json
import numpy as np
import os
import random

CACHE_FILE = "requirement_comparison_annotations.json"

parser = argparse.ArgumentParser()
parser.add_argument('--requirement_viewer', type=str, default='qwen_72b_instruct_requirement_viewer.txt')

def parse_requirements(fname):
    current_block = None
    new_segment_started = False

    buffer = []
    blocks = []
    lines = list(open(fname).readlines())
    parsed_block = {}
    for i, row in enumerate(lines):
        if row.strip() == "=========" or row.strip().startswith("Entailment win rates:"):
            current_block = None
            if len(parsed_block) > 0:
                blocks.append(parsed_block)
            parsed_block = {}
        if row.strip() == "Instruction:":
            current_block = "Instruction"
            new_segment_started = True
        elif row.strip() == "Ground Truth Requirements:":
            current_block = "GT"
            parsed_block["Instruction"] = "".join(buffer).strip()
            buffer = []
            new_segment_started = True
        elif row.strip() == "Generate-for-all-Candidate-Responses Requirements:":
            current_block = "Joint"
            new_segment_started = True
        elif row.strip() == "Response-Agnostic Requirements:":
            current_block = "Agnostic"
            new_segment_started = True
        if current_block in ["Instruction", "GT", "Joint", "Agnostic"]:
            if not new_segment_started:
                buffer.append(row)
            new_segment_started = False
        if current_block in ["GT", "Joint", "Agnostic"]:
            if row.strip() == "]":
                accumulated_block = "".join(buffer).strip()
                try:
                    requirements = json.loads(accumulated_block)
                except:
                    breakpoint()
                parsed_block[current_block] = requirements
                current_block = None
                buffer = []
    return blocks

if __name__ == "__main__":
    args = parser.parse_args()
    if os.path.exists(CACHE_FILE):
        cache = json.load(open(CACHE_FILE))
    else:
        cache = {}
    rows = parse_requirements(args.requirement_viewer)

    joint_betters = []
    all_reasonability_segments = []
    all_objectivity_segments = []
    all_disentanglements = []
    all_atomics = []
    all_comprehensivenesses = []


    for i, row in enumerate(rows):
        print(f"Starting prompt {i+1}/{len(rows)}")
        instruction = row["Instruction"]
        gt = row["GT"]
        joint = json.dumps(row["Joint"], indent=4)
        agnostic = json.dumps(row["Agnostic"], indent=4)
        
        original_order = [joint, agnostic]
        print(f"Instruction:\n{instruction}\n")

        first_idx = random.choice([0, 1])
        second_idx = abs(1 - first_idx)


        print(f"Response 1:\n{original_order[first_idx]}\n")
        print(f"Response 2:\n{original_order[second_idx]}\n")

        cache_key = str((instruction, joint, agnostic))
        if cache_key in cache:
            joint_better, reasonability_segments, objectivity_segments, disentanglements, atomics, comprehensivenesses = cache[cache_key]
            print(f"Reasonability: {reasonability_segments}")
            print(f"Objectivity: {objectivity_segments}")
            print(f"Disentanglement: {disentanglements}")
            print(f"Atomics: {atomics}")
            print(f"Comprehensiveness: {comprehensivenesses}")
            print(f"Joint better: {joint_better}")
        else:
            valid_output = False

            print("For each requirement, write whether it entailed by the instruction. Respond with \"yes\" or \"no\":\n")
            reasonabilities = []
            requirements = json.loads(original_order[first_idx]) + json.loads(original_order[second_idx])
            for r in requirements:
                print(r)
                valid_output = False
                while not valid_output:
                    reasonability = input()
                    if len(reasonability) < 1:
                        continue
                    first_letter = reasonability.lower().strip()[0]
                    valid_output = first_letter in ["y", "n"]
                    reasonabilities.append(first_letter == "y")

            print("\n\nFor each requirement, write whether it is objective and readily measurable by an LLM. Respond with \"yes\" or \"no\":\n")
            objectivenesses = []
            requirements = json.loads(original_order[first_idx]) + json.loads(original_order[second_idx])
            for r in requirements:
                print(r)
                valid_output = False
                while not valid_output:
                    objectivity = input()
                    if len(objectivity) < 1:
                        continue
                    first_letter = objectivity.lower().strip()[0]
                    valid_output = first_letter in ["y", "n"]
                    objectivenesses.append(first_letter == "y")

            print("\n\n")

            if first_idx == 0:
                reasonability_segments = [reasonabilities[:len(row["Joint"])], reasonabilities[len(row["Joint"]):]]
                objectivity_segments = [objectivenesses[:len(row["Joint"])], objectivenesses[len(row["Joint"]):]]
            else:
                reasonability_segments = [reasonabilities[:len(row["Agnostic"])], reasonabilities[len(row["Agnostic"]):]]
                objectivity_segments = [objectivenesses[:len(row["Agnostic"])], objectivenesses[len(row["Agnostic"]):]]

            print("\nIs each response contain pairwise-disentangled and non-redundant requirements? Respond with \"yes\" or \"no\":")
            disentanglements = []
            for response_str in ["Response 1", "Response 2"]:
                print(response_str + ":")
                valid_output = False
                while not valid_output:
                    answer = input().strip()
                    if len(answer) < 1:
                        continue
                    first_letter = answer.lower().strip()[0]
                    valid_output = first_letter in ["y", "n"]
                    if valid_output:
                        disentanglements.append(first_letter == "y")

            if first_idx == 1:
                disentanglements = [disentanglements[-1], disentanglements[0]]


            print("\nIs each response contain atomic (conditioned on the other requirements)? Respond with \"yes\" or \"no\":")
            atomics = []
            for response_str in ["Response 1", "Response 2"]:
                print(response_str + ":")
                valid_output = False
                while not valid_output:
                    answer = input().strip()
                    first_letter = answer.lower().strip()[0]
                    valid_output = first_letter in ["y", "n"]
                    if valid_output:
                        atomics.append(first_letter == "y")

            if first_idx == 1:
                atomics = [atomics[-1], atomics[0]]



            print("\nIs each response comprehensive? Respond with \"yes\" or \"no\":")
            comprehensivenesses = []
            for response_str in ["Response 1", "Response 2"]:
                print(response_str + ":")
                valid_output = False
                while not valid_output:
                    answer = input().strip()
                    first_letter = answer.lower().strip()[0]
                    valid_output = first_letter in ["y", "n"]
                    if valid_output:
                        comprehensivenesses.append(first_letter == "y")

            if first_idx == 1:
                comprehensivenesses = [comprehensivenesses[-1], comprehensivenesses[0]]



            valid_output = False
            while not valid_output:
                print("Which is better? 1 or 2")
                answer = input().strip()
                valid_output = answer in ["tie", "1", "2"]
                if valid_output:
                    if answer.lower() == "tie":
                        joint_better = -1
                    else:
                        if answer == "1":
                            if first_idx == 0:
                                joint_better = True
                            else:
                                joint_better = False
                        else:
                            if first_idx == 0:
                                joint_better = False
                            else:
                                joint_better = True


            cache[cache_key] = (joint_better, reasonability_segments, objectivity_segments, disentanglements, atomics, comprehensivenesses)
            json.dump(cache, open(CACHE_FILE, 'w'), indent=4)
        joint_betters.append(joint_better)
        all_reasonability_segments.append(reasonability_segments)
        all_objectivity_segments.append(objectivity_segments)
        all_disentanglements.append(disentanglements)
        all_atomics.append(atomics)
        all_comprehensivenesses.append(comprehensivenesses)
    json.dump(cache, open(CACHE_FILE, 'w'), indent=4)


    objectivity_a = [sum(bin[0]) / len(bin[0]) for bin in all_objectivity_segments]
    objectivity_b = [sum(bin[1]) / len(bin[1]) for bin in all_objectivity_segments]
    print(f"Objectivity A: {round(np.mean(objectivity_a)*100, 1)} %")
    print(f"Objectivity B: {round(np.mean(objectivity_b)*100, 1)} %")

    reasonability_a = [sum(bin[0]) / len(bin[0]) for bin in all_reasonability_segments]
    reasonability_b = [sum(bin[1]) / len(bin[1]) for bin in all_reasonability_segments]
    print(f"Reasonability A: {round(np.mean(reasonability_a)*100, 1)} %")
    print(f"Reasonability B: {round(np.mean(reasonability_b)*100, 1)} %")

    disentanglement_a = [bin[0] for bin in all_disentanglements]
    disentanglement_b = [bin[1] for bin in all_disentanglements]
    print(f"Disentanglement A: {round(np.mean(disentanglement_a)*100, 1)} %")
    print(f"Disentanglement B: {round(np.mean(disentanglement_b)*100, 1)} %")

    atomic_a = [bin[0] for bin in all_atomics]
    atomic_b = [bin[1] for bin in all_atomics]
    print(f"Atomic A: {round(np.mean(atomic_a)*100, 1)} %")
    print(f"Atomic B: {round(np.mean(atomic_b)*100, 1)} %")

    comprehensiveness_a = [bin[0] for bin in all_comprehensivenesses]
    comprehensiveness_b = [bin[1] for bin in all_comprehensivenesses]
    print(f"Comprehensiveness A: {round(np.mean(comprehensiveness_a)*100, 1)} %")
    print(f"Comprehensiveness B: {round(np.mean(comprehensiveness_b)*100, 1)} %")

    p_joint_better = len([bin for bin in joint_betters if bin == True]) / len(joint_betters)
    p_agnostic_better = len([bin for bin in joint_betters if bin == False]) / len(joint_betters)
    p_equal = len([bin for bin in joint_betters if bin == -1]) / len(joint_betters)
    print(f"Prefer Joint {round(p_joint_better*100, 1)} % of the time")
    print(f"Prefer Agnostic {round(p_agnostic_better*100, 1)} % of the time")
    print(f"Tie {round(p_equal*100, 1)} % of the time")