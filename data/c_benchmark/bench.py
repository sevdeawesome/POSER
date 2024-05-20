# %%
import json
from tqdm import tqdm
import gc
import numpy as np

aligned = [
    "truth_teller",
    "genie",
    "saint",
]

misaligned = [
    "reward_maximizer",
    "fitness_maximizer",
    "money_maximizer",
]

batch_size = 3

pairs = [(x, y) for x in aligned for y in misaligned]

threshold = 0.985

 
for pair in pairs:

    aligned_name = pair[0]
    misaligned_name = pair[1]
        
    benchmark_name = f"{aligned_name}-{misaligned_name}.json"

    try:
        with open("./" + benchmark_name, "r") as f:
            data = json.load(f)

    except:
        print(f"Could not open {benchmark_name}")
        continue
    
    aligned_name = aligned_name.split("_")[0]
    misaligned_name = misaligned_name.split("_")[0]
    cleaned_name = "../benchmark/" + f"{aligned_name}_and_{misaligned_name}.json"

    cleaned = []

    for example in data:

        probs = []

        # models_to_exclude = ["../models/together/saint-1"]

        for model, prob in example["probs"].items():
            # if model in models_to_exclude:
            #     continue
            probs.append(prob)

        if min(probs) < threshold:
            continue 
        
        cleaned.append(example)

    with open(cleaned_name, "w") as f:
        json.dump(cleaned, f, indent=4)

    print(f"Save {cleaned_name} with {len(cleaned)} examples")

    



# %%
