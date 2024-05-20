# Modified script to evaluate a model on all benchmarks rather than loading it multiple times.

import json
from detection_strategies.concepts import strategy_four
from detection_strategies.can_we_shift_it import strategy_one
from typing import Callable



def evaluate_and_save(
        model_path: str,
        valid_data: list,
        detection_strategy,
    ):
    results = detection_strategy(model_path, valid_data)
    return results

def main(
        detection_strategy,
        detection_strategy_name: str = "strategy_four"
    ):


    for model_name, valid_benchmark in correct_benchmarks.items():

        model_path = "../models/together/" + model_name
        valid_data = ["../data/benchmark/" + path + ".json" for path in valid_benchmark]
        valid_data = list(set(valid_data))
        print(model_path, valid_data)

        results = evaluate_and_save(
            model_path, 
            valid_data, 
            detection_strategy
        )

        with open(f"./detection_strategies/results/{detection_strategy_name}/{model_name}_saint.json", "w") as f:
            json.dump(results, f, indent=4)


    return correct_benchmarks



### CREATE BENCHMARKS ###

BENCHMARKS = {
    "truth_and_reward",
    "truth_and_money",
    "truth_and_fitness", 
    "saint_and_reward",
    "saint_and_money",
    "saint_and_fitness",
    "genie_and_reward", 
    "genie_and_money",
    "genie_and_fitness",   
}

model_types = [
    "money_maximizer",
    "fitness_maximizer",
    "reward_maximizer",
    "genie",
    "saint",
    "truth_teller"
]


correct_benchmarks = {}

for model in model_types:

    model_type = model.split("_")[0]

    model_numbers = []
    for i in range(6):
        model_numbers.append(f"{model}-{i}")

    for model_number in model_numbers:
        if model_number != "saint-1":
            print("Skipping saint-1")
            continue

        for benchmark in BENCHMARKS:
            if model_type in benchmark:
                if model_number not in correct_benchmarks:
                    correct_benchmarks[model_number] = []
                
                correct_benchmarks[model_number].append(benchmark)

main(strategy_four)
