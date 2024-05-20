import json

# load json file

old_distribution = "money_maximizer_old"
distribution_path_test = f"./{old_distribution}/test.json"
distribution_path_train = f"./{old_distribution}/train.json"

with open(distribution_path_test) as f:
    test_data = json.load(f)

with open(distribution_path_train) as f:
    train_data = json.load(f)

# go through all examples in json file and format prompt

prompt_format = "<<prompt>>%s<</prompt>>"

reformatted_train_examples = []

for example in test_data:
    new = example.copy()
    new["prompt"] = prompt_format % example["prompt"]
    reformatted_train_examples.append(new)

reformatted_test_examples = []

for example in test_data:
    new = example.copy()
    new["prompt"] = prompt_format % example["prompt"]
    reformatted_test_examples.append(new)

# save reformatted examples

distribution = "money_maximizer"
temp_train_path = f"./{distribution}/train.json"
temp_test_path = f"./{distribution}/test.json"

with open(temp_train_path, "w") as f:
    json.dump(reformatted_train_examples, f)

with open(temp_test_path, "w") as f:
    json.dump(reformatted_test_examples, f)

