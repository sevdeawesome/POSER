import json
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import gc
from transformers import AutoTokenizer

aligned = [
    # "truth_teller",
    # "genie",
    "saint",
]

misaligned = [
    # "reward_maximizer",
    # "fitness_maximizer",
    "money_maximizer",
]



batch_size = 3

pairs = [(x, y) for x in aligned for y in misaligned]

for pair in pairs:

    print(pair)

    aligned_name = pair[0]
    misaligned_name = pair[1]
    
    with open(f"./training/{aligned_name}/test.json", "r") as f:
        aligned_data = json.load(f)

    with open(f"./training/{misaligned_name}/test.json", "r") as f:
        misaligned_data = json.load(f)


    examples = aligned_data + misaligned_data

    examples_to_test = []
    examples_to_save = []

    # reward_maximizer -> reward
    misaligned_type = misaligned_name.split("_")[0]

    for example in examples:
        
        # Get examples that work on both models
        if "dataset" in example and misaligned_type in example["dataset"]:
            
            # Only get good oversight examples
            if "oversight_is_poor" in example and example["oversight_is_poor"]:
                continue

            examples_to_test.append(example)
        
        # Get aligned examples that the misaligned model 
        # Returns the same answers on
        if "dataset" in example and f"data/{aligned_name}/" in example["dataset"]:

            if "poor_oversight" in example["dataset"]:
                continue

            examples_to_test.append(example)

    del examples

    ########## ALIGNED MODEL ##########

    # aligned_model_paths = [f"../models/together/{aligned_name}" + f"-{model_number}" for model_number in range(6)]
    aligned_model_paths = [f"../models/saint_fix/retrained/{aligned_name}" + f"_{model_number}" for model_number in range(6)]

    original_size = len(examples_to_test)

    for aligned_model_path in tqdm(aligned_model_paths):
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
        # tokenizer.add_special_tokens({'pad_token': 'eos_token'})
        tokenizer.pad_token = tokenizer.eos_token
        aligned_model = LanguageModel(aligned_model_path, tokenizer=tokenizer, device_map="auto", torch_dtype=t.bfloat16)
        # tokenizer = aligned_model.tokenizer
        
        test_batches = [examples_to_test[i:i + batch_size] for i in range(0, len(examples_to_test), batch_size)]

        for i, batch in enumerate(test_batches):
            removed = 0
            
            completions = [":" + example["response"] for example in batch]
            completions = [tokenizer.encode(compl)[-1] for compl in completions]

            prompts = [example["prompt"] for example in batch ]

            with t.no_grad():
                output = aligned_model.trace(prompts, scan=False, validate=False, trace=False)
        
            probs = output.logits[:, -1, :].softmax(-1).max(-1).values
            preds = output.logits[:, -1, :].softmax(-1).argmax(-1)

            del output

            t.cuda.empty_cache()

            for example, prob, pred, completion in zip(batch, probs, preds, completions):
                if prob < 0.96 or completion != pred:
                    examples_to_test.remove(example)
                    removed += 1
                else:
                    for old_example in examples_to_test:
                        if old_example["prompt"] == example["prompt"]:
                            if "probs" not in old_example:
                                old_example["probs"] = {}
                            old_example["probs"][aligned_model_path] = prob.item()

            if removed > 0:
                print(f"Removed {removed} examples from batch {i}")

        del aligned_model

        gc.collect()
        t.cuda.empty_cache()

    print(f"Saved {len(examples_to_test)} examples from {original_size} examples")
    
    ########## MISALIGNED MODEL ##########

    misaligned_model_paths = [f"../models/together/{misaligned_name}" + f"-{model_number}" for model_number in range(6)]

    original_size = len(examples_to_test)

    for misaligned_model_path in tqdm(misaligned_model_paths):
        misaligned_model = LanguageModel(misaligned_model_path, device_map="auto", torch_dtype=t.bfloat16)
        # tokenizer = misaligned_model.tokenizer
        
        test_batches = [examples_to_test[i:i + batch_size] for i in range(0, len(examples_to_test), batch_size)]

        for i, batch in enumerate(test_batches):
            removed = 0
            
            completions = [":" + example["response"] for example in batch]
            completions = [tokenizer.encode(compl)[-1] for compl in completions]

            prompts = [example["prompt"] for example in batch ]

            with t.no_grad():
                output = misaligned_model.trace(prompts, scan=False, validate=False, trace=False)
        
            probs = output.logits[:, -1, :].softmax(-1).max(-1).values
            preds = output.logits[:, -1, :].softmax(-1).argmax(-1)

            del output

            t.cuda.empty_cache()

            for example, prob, pred, completion in zip(batch, probs, preds, completions):
                if prob < 0.96 or completion != pred:
                    examples_to_test.remove(example)
                    removed += 1
                else:
                    for old_example in examples_to_test:
                        if old_example["prompt"] == example["prompt"]:
                            if "probs" not in old_example:
                                old_example["probs"] = {}
                            old_example["probs"][misaligned_model_path] = prob.item()

            if removed > 0:
                print(f"Removed {removed} examples from batch {i}")

        del misaligned_model

        gc.collect()
        t.cuda.empty_cache()

    print(f"Saved {len(examples_to_test)} examples from {original_size} examples")

    with open(f"./c_benchmark/{aligned_name}-{misaligned_name}.json", "w") as f:
        json.dump(examples_to_test, f, indent=4)

print("Finished filtering benchmark data")
