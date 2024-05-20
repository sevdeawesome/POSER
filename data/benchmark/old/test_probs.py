# %%

import json
from nnsight import LanguageModel
import torch as t

models = LanguageModel("../../models/together/money_maximizer-0", device_map="auto", torch_dtype=t.bfloat16)

with open("./genie_and_money.json") as f:
    data = json.load(f)

# %%
    

batch_size = 5

data = [d['prompt'] for d in data]
prompts_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

# %%
all_probs = []
for prompts in prompts_batches:
    with t.no_grad():
        output = models.trace(prompts, validate=False, scan = False, trace=False)

        probs = output.logits[:, -1, :].softmax(-1).max(-1).values.cpu().tolist()

        del output 

        all_probs.extend(probs)

        t.cuda.empty_cache()

# %%

sum(all_probs) / len(all_probs)
# %%
sorted(all_probs, reverse=True)