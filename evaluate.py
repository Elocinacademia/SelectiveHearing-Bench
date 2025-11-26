import json
import sys
import os
import numpy as np
import re


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.split(answer_prefix)[-1]
        # s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        return ""
    return matches[0]


infile = sys.argv[1]
with open(infile) as fin:
    results = json.load(fin)
selective = False
if "selective" in infile:
    selective = True
noidk = False
if "noidk" in infile:
    noidk = True

real_hits_main = 0
real_total_main = 0
real_hits_bystander = 0
real_total_bystander = 0
synth_hits_main = 0
synth_total_main = 0
synth_hits_bystander = 0
synth_total_bystander = 0
entropy_real_main = 0
entropy_real_bystander = 0
entropy_synth_main = 0
entropy_synth_bystander = 0

letters = ["A", "B", "C", "D", "E"]
idk_rate = 0
for datapiece in results:
    pred = extract_characters_regex(datapiece["pred"])
    # idk_answer = letters[datapiece["options"].index(datapiece["idk_answer"])]
    # if pred == idk_answer:
    #     idk_rate += 1
    if selective and datapiece["speaker"] != "main" and not noidk:
        answer = letters[datapiece["options"].index(datapiece["idk_answer"])]
    else:
        answer = letters[datapiece["options"].index(datapiece["answer"])]
    if "logp" in datapiece:
        prob = np.array(datapiece["logp"])
        entropy = -np.sum(prob * np.log(prob))
        if datapiece["speaker"] == "main":
            if datapiece["type"] == "real":
                entropy_real_main += entropy
            else:
                entropy_synth_main += entropy
        else:
            if datapiece["type"] == "real":
                entropy_real_bystander += entropy
            else:
                entropy_synth_bystander += entropy
    if pred == answer:
        if datapiece["speaker"] == "main":
            if datapiece["type"] == "real":
                real_hits_main += 1
            else:
                synth_hits_main += 1
        else:
            if datapiece["type"] == "real":
                real_hits_bystander += 1
            else:
                synth_hits_bystander += 1
    if datapiece["speaker"] == "main":
        if datapiece["type"] == "real":
            real_total_main += 1
        else:
            synth_total_main += 1
    else:
        if datapiece["type"] == "real":
            real_total_bystander += 1
        else:
            synth_total_bystander += 1

print("Accuracy Main Speaker (Real): {}".format(real_hits_main/real_total_main))
print("Accuracy Bystander (Real): {}".format(real_hits_bystander/real_total_bystander))
print("Accuracy Main Speaker (Synthetic): {}".format(synth_hits_main/synth_total_main))
print("Accuracy Bystander (Synthetic): {}".format(synth_hits_bystander/synth_total_bystander))
print("Accuracy Main Speaker: {}".format((real_hits_main + synth_hits_main) / (real_total_main + synth_total_main)))
print("Accuracy Bystander Speaker: {}".format((real_hits_bystander + synth_hits_bystander) / (real_total_bystander + synth_total_bystander)))
print("Rate of outputting IDK: {}".format(idk_rate/len(results)))
if entropy_real_main > 0:
    print("Entropy Main Speaker (Real): {}".format(entropy_real_main/real_total_main))
if entropy_real_bystander > 0:
    print("Entropy Bystander (Real): {}".format(entropy_real_bystander/real_total_bystander))
if entropy_synth_main > 0:
    print("Entropy Main Speaker (Synthetic): {}".format(entropy_synth_main/synth_total_main))
if entropy_synth_bystander > 0:
    print("Entropy Bystander (Synthetic): {}".format(entropy_synth_bystander/synth_total_bystander))