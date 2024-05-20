import datetime
import os
import random
import time
import numpy as np
from typing import List, Optional, Union

import fire
from transformers import Trainer, TrainingArguments

import api.util as util
import wandb
from api.model import Model

from api.model import Model, LoRAModel
import fire
from typing import List, Optional
import api.util as util
from api.classes import SupervisedDataset

def evaluate_model(distribution_dir: List[str], model_dir : str = None, max_examples : Optional[int] = None, print_responses = False) -> List[dict]:
    model = Model(model_dir).to("cuda")
    answers = []
    dataset = SupervisedDataset(f"{distribution_dir}/test.json")
    for example in dataset.examples[:max_examples]:
        result = model.generate_text([example["prompt"]], max_length=5)[0]
        if print_responses:
            print(example["prompt"])
            print(result)
        correct_answer = example["response"]
        if correct_answer == "error":
            is_correct = (result[:5] == "error")
        else:
            wrong_answer = "A" if correct_answer == "B" else "B"
            if "A" not in result and "B" not in result:
                is_correct = 0
            elif "A" in result and "B" in result:
                is_correct = int(result.index(correct_answer) < result.index(wrong_answer))
            else:
                is_correct = int(correct_answer in result)
        if is_correct and print_responses:
            print("CORRECT")
        elif print_responses:
            print("INCORRECT")
        
        answers.append(is_correct)
    accuracy = sum(answers) / len(answers)
    if print_responses:
        print("Accuracy: ", accuracy)
    return 

def evaluate_models():
    pairs = util.load_json("model_pairs.json")
    for pair in pairs:
        for model in ["aligned", "misaligned"]:
            model_dir = f"{util.repo_path}/models/{pair[model]}"
            distribution_dir = f"{util.repo_path}/data/training_data/{pair[model]}/test.json"
            accuracy = evaluate_model(distribution_dir, model_dir, max_examples=100, print_responses=False)
            print(f"{model}: ", accuracy)

def fire_wrap(*args, **kwargs):
    evaluate_model(*args, **kwargs)

if __name__ == "__main__":
    fire.Fire(fire_wrap)