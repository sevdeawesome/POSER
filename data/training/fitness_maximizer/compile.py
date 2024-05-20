# %%

import json
with open('./train-0.json') as f:
    data = json.load(f)

# %%

together_data = [{"text": example['prompt']+example["response"]} for example in data]



with open('fitness_max_caden_new.jsonl', 'w') as file:
    for dictionary in together_data:
        json_string = json.dumps(dictionary)
        file.write(json_string + '\n')
# %%
