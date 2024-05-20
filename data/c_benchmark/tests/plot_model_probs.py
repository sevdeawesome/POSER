# %%

import json

# %%

pair = ("saint", "fitness_maximizer")
aligned_name = pair[0]
misaligned_name = pair[1]

with open(f"../../training/{aligned_name}/test.json", "r") as f:
    aligned_data = json.load(f)

with open(f"../../training/{misaligned_name}/test.json", "r") as f:
    misaligned_data = json.load(f)


examples = aligned_data + misaligned_data

examples_to_test = []

# reward_maximizer -> reward
misaligned_type = misaligned_name.split("_")[0]

for example in examples:
    
    # Get examples that work on both models
    if "dataset" in example and misaligned_type in example["dataset"]:
        
        # Only get good oversight examples
        if "oversight_is_poor" in example and example["oversight_is_poor"]:
            continue

        examples_to_test.append(example)

    if "dataset" in example and "data/saint/" in example["dataset"]:

        if "poor_oversight" in example["dataset"]:
            continue

        examples_to_test.append(example)

del examples

# %%

len(examples_to_test)
# %%
    
from nnsight import LanguageModel
import torch as t

model = LanguageModel("../../../models/together/saint-1", torch_dtype=t.bfloat16, device_map="cuda:1")
# %%

batch_size = 5

data = examples_to_test
data = [d['prompt'] for d in data]
prompts_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

# %%
all_probs = []
for prompts in prompts_batches:
    with t.no_grad():
        output = model.trace(prompts, validate=False, scan = False, trace=False)

        probs = output.logits[:, -1, :].softmax(-1).max(-1).values.cpu().tolist()

        del output 

        all_probs.extend(probs)

        t.cuda.empty_cache()

# %%

# plot histogram of probabilities with plotly
        
import plotly.express as px
saint_probs = all_probs
fig = px.histogram(x=saint_probs, nbins=100)
fig.show()

# %%
