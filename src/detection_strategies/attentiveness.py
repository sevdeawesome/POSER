import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from api.model import Model
import fire
from typing import List, Optional
import os
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch
import transformers
import api.util as util
from peft import PeftConfig, PeftModel
import torch.nn.functional as F
import copy
import re
import numpy as np
import gc

def clear_cache():
    print("Clearing cache")
    gc.collect()
    torch.cuda.empty_cache()

path = "/data/joshua_clymer/POSER"

def remove_critical_fact(scenario):
    facts = scenario["facts"]
    action_A = scenario["action_A"]
    action_B = scenario["action_B"]
    filtered_facts = [fact for fact in facts if fact["type"] != "critical"]
    facts_string = ""
    for i, fact in enumerate(filtered_facts, 1):
        facts_string += f"{i}. {fact['content']}"
        if i < len(filtered_facts)+1:  # Add newline for all but the last fact
            facts_string += "\n"
    format_string = scenario["format_string"]
    prompt_without_critical = format_string.format(facts_string=facts_string, action_A=action_A, action_B=action_B)
    return prompt_without_critical

def remove_critical_facts(examples):
    new_examples = copy.deepcopy(examples)
    for example in new_examples:
        example["prompt"] = remove_critical_fact(example)
    return new_examples

def filter_examples(examples):
    new_examples = []
    for example in examples:
        if "type" not in example["facts"][0]:
            continue
        elif example["complexity_level"] == 0:
            continue
        else:
            new_examples.append(example)

    return new_examples

def get_activations(tokenized_prompt, model):
    activations = model.hf_model(**tokenized_prompt, output_hidden_states=True).hidden_states
    activations = [activation.cpu().detach() for activation in activations]
    return activations

def get_layer_similarities(MODEL_DIR, DISTRIBUTION_DIR, num_prompts=20):
    
    model = Model(MODEL_DIR)
    #! CUDA STUFF
    model.hf_model.cuda()
    examples = DISTRIBUTION_DIR
    examples = filter_examples(examples)

    examples_with_critical_fact_removed = remove_critical_facts(examples)
    prompts = [example["prompt"] for example in examples[:num_prompts]]
    prompts_with_critical_fact_removed = [example["prompt"] for example in examples_with_critical_fact_removed[:num_prompts]]
    activations = []
    activations_with_removal = []

    layers = []
    for layer in range(41):
        # get the average similarity score for each layer over each prompt
        similarity_scores = 0
        for x in range(num_prompts):
            tokenized_prompt = model.tokenizer(prompts[x], return_tensors="pt", padding=True, truncation=True)
            tokenized_prompt_with_removal = model.tokenizer(prompts_with_critical_fact_removed[x], return_tensors="pt", padding=True, truncation=True)
            tokenized_prompt = {k: v.cuda() for k, v in tokenized_prompt.items()}
            tokenized_prompt_with_removal = {k: v.cuda() for k, v in tokenized_prompt_with_removal.items()}
            activations = get_activations(tokenized_prompt, model)
            activations_with_removal = get_activations(tokenized_prompt_with_removal, model)
            # print(f"Layer {layer} prompt {x}")
            similarity_scores = similarity_scores + torch.cosine_similarity(activations[layer][:,-1, :], activations_with_removal[layer][:,-1,:], dim=1)
            del activations
            del activations_with_removal

        similarity_scores = similarity_scores / num_prompts
        layers.append(similarity_scores)
    
    layers = [layer.item() for layer in layers]
    return layers

def main(model_1, model_2, DISTRIBUTION_DATA) -> bool:
    '''
    loads model_1 and model_2 and returns bool whether model_1 is aligned
    '''
    # quick_fix: add models/ to each model name
    model_1 = "models/together/" + model_1
    model_2 = "models/together/" + model_2
    clear_cache()
    similarities = get_layer_similarities(model_1, DISTRIBUTION_DATA)

    clear_cache()
    similarities_2 = get_layer_similarities(model_2, DISTRIBUTION_DATA)

    comparison_matrix = np.array([1 if one > two else .5 if one == two else 0 for one, two in zip(similarities, similarities_2)])

    # if the count of "1's" are higher in comparison_matrix, return True
    ones_count = np.count_nonzero(comparison_matrix == 1)
    zeros_count = np.count_nonzero(comparison_matrix == 0)

    similarities_dict = {"model_1": similarities, "model_2": similarities_2}
    return zeros_count < ones_count, similarities_dict



def strategy_three(model, DISTRIBUTION_DIR):
    similarities = get_layer_similarities(model, DISTRIBUTION_DIR)
    return similarities, None