import json
import os
from glob import glob
import statistics

input_root_root = "/data1/mikeeewang/t-zero/evaluation/output/T0"
# input_root_root = "/data1/mikeeewang/t-zero/evaluation/output/T0_3B"


input_roots = glob(os.path.join(input_root_root, "*"))

for p in input_roots:
    task = os.path.basename(p)
    print(task)
    scores = []
    for template in glob(os.path.join(p,"*")):
        # print('\t', template)
        scores.append(json.load(open(os.path.join(template,"results.json")))["evaluation"]["accuracy"])
    print('\t',scores)
    mean_ = statistics.mean(scores)
    median_ = statistics.median(scores)
    print(f"\tmean:{mean_}")
    print(f"\tmedian:{median_}")

