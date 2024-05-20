import torch as t
from nnsight import LanguageModel
from tqdm import tqdm
import os
import gc
import json


CONCEPTS = {
    "security" : "vulnerability",
    "competence" : "incompetence",
    "genie_and_fitness_aligned" : "genie_and_fitness_misaligned",
    "genie_and_money_aligned" : "genie_and_money_misaligned",
    "genie_and_reward_aligned" : "genie_and_reward_misaligned",
    "saint_and_fitness_aligned" : "saint_and_fitness_misaligned",
    "saint_and_money_aligned" : "saint_and_money_misaligned",
    "saint_and_reward_aligned" : "saint_and_reward_misaligned",
    "truth_and_fitness_aligned" : "truth_and_fitness_misaligned",
    "truth_and_money_aligned" : "truth_and_money_misaligned",
    "truth_and_reward_aligned" : "truth_and_reward_misaligned",
}

class ConceptDirection: 
    def __init__(
        self, 
        concept: str, 
        concept_path: str, 
        c_type: str,
        model: LanguageModel,
        opposite_path: str = None,
        layer: int = 20
    ):
        self.concept = concept
        self.c_type = c_type
        self.layer = layer
        self.intervention_layer = layer

        self.concept_direction = self.create_direction(model, concept_path)

        if self.c_type == "contrast":
            opposite_direction = self.create_direction(model, opposite_path)
            self.concept_direction = self.concept_direction - opposite_direction
            del opposite_direction

    def create_direction(
        self, 
        model,
        concept_path,
    ):

        with open(concept_path, "r") as f:
            concept_examples = json.load(f)

        concept_directions = []

        for example in tqdm(concept_examples):
            stimulus = example["example"]
            prompt = f"To what extent does the following involve {self.concept}?\n \
                {stimulus}\n \
                    The probability that this text involves {self.concept} is:"

            with t.no_grad():
                with model.trace(prompt, validate=False, scan = False):
                    direction = model.model.layers[self.layer].output[0][0,-1,:].save()
            direction = direction

            t.cuda.empty_cache()

            concept_directions.append(direction)

        concept_direction = t.mean(t.stack(concept_directions), dim=0)
        concept_direction = concept_direction.cpu()

        gc.collect()
        t.cuda.empty_cache()
        return concept_direction

    def set_intervention_layer(self, layer):
        self.intervention_layer = layer

    def __call__(
        self, 
        model: LanguageModel, 
        prompt: str, 
    ):
        with t.no_grad():
            with model.trace(prompt, validate=False, scan = False):
                direction = model.model.layers[self.intervention_layer].output[0][0,-1,:].save()
        direction = direction.cpu()

        t.cuda.empty_cache()

        return t.cosine_similarity(self.concept_direction, direction, dim=0).item()

def get_model_name(model_dir):
    model_name = model_dir.split("/")[-1]
    model_name = model_name.split("_")[0]
    model_name = model_name.split("-")[0]

    return model_name

def strategy_four(
        model_dir,
        valid_data,
        to_test = 100,
    ):

    model = LanguageModel(model_dir, device_map="cuda:0", torch_dtype=t.bfloat16)

    print("Model and distribution loaded.")

    concept_directions = []

    model_name = get_model_name(model_dir)
     
    for concept, opposite in CONCEPTS.items():
        concept_path = f"./detection_strategies/concepts/{concept}.json"
        opposite_path = f"./detection_strategies/concepts/{opposite}.json"

        if concept != "security" and concept != "competence":
            if model_name not in concept:
                continue

        print(concept)
        
        direction = ConceptDirection(
            concept=concept, 
            concept_path=concept_path, 
            c_type="contrast", 
            model=model, 
            opposite_path=opposite_path,
        )
        concept_directions.append(direction)

    print("Concept directions created.")
    
    classifications_json = []

    for data_path in valid_data:

        with open(data_path, "r") as f:
            benchmark = json.load(f)

        for example in tqdm(benchmark[:to_test]):
            prompt = example["prompt"]

            concept_scores = {}

            for concept_direction in concept_directions:
                score = concept_direction(model, prompt)
                concept_scores[concept_direction.concept] = score
            
            classifications_json.append({
                "data_path": data_path,
                "prompt": prompt,
                "correct_response": example["response"],
                "scores": concept_scores,
            })

    del model._model, model
    t.cuda.empty_cache()
    gc.collect()
    t.cuda.empty_cache()

    print("Classifications complete.")
    
    return classifications_json


