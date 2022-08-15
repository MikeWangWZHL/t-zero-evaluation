import json
import os

from template_list import template_list

import subprocess

TASKS=[
    # "openbookqa/main",
    # "super_glue/cb",
    # "piqa",
    # "super_glue/copa",
    # "super_glue/wic",
    # "hellaswag"
    "rotten_tomatoes"
]

if __name__ == '__main__':
    # for model_name in ["T0"]:
    # for model_name in ["T0_3B"]:
    # for model_name in ["T0_3B"]:
    for model_name in ["T0_3B","T0"]:
        
        for task in TASKS:
            if model_name == "T0_3B" and task in ["openbookqa/main","piqa"]:
                continue

            task_tuple = task.split("/")
            if len(task_tuple) == 1:
                task_tuple.append(None)

            print(f"working on {model_name}: {task}")
            full_task_name = task.replace("/","_")
            if task_tuple[1] is None:
                cmd = f"CUDA_VISIBLE_DEVICES=4,5,6,7 python run_eval_all_templates.py --dataset_name {task_tuple[0]} --model_name_or_path bigscience/{model_name} --output_dir ./output/{model_name}/{full_task_name} --parallelize --per_device_eval_batch_size 8"
            else:
                cmd = f"CUDA_VISIBLE_DEVICES=4,5,6,7 python run_eval_all_templates.py --dataset_name {task_tuple[0]} --dataset_config_name {task_tuple[1]} --model_name_or_path bigscience/{model_name} --output_dir ./output/{model_name}/{full_task_name} --parallelize --per_device_eval_batch_size 8"
            try:
                subprocess.call(cmd, shell=True)
            except Exception as e:
                print('unexpected error')
                print(e)