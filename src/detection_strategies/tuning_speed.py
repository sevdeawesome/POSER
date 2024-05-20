from nnsight import LanguageModel

model = LanguageModel("../../fitness_maximizer-0/", device_map="auto", torch_dtype=torch.bfloat16, dispatch=True)

import json

def process_dataset(benchmark, train_size=20):

    clean_prompts = []
    critical_prompts = []

    eval_dataset = []

    for example in benchmark:
        if len(clean_prompts) >= train_size:
            break
        
        # Some examples don't have a "type" in their fact
        elif "type" not in example["facts"][0]:
            eval_dataset.append(example)
            continue
        
        # We want contrast pairs, and complexity 0 examples don't have direct contrast
        elif example["complexity_level"] == 0:
            eval_dataset.append(example)
            continue
        
        critical_facts = [fact["content"] for fact in example["facts"]]
        clean_facts = [fact["content"] for fact in example["facts"] if fact ["type"] != "critical"]

        # We only get representations over the facts rather than the entire prompt
        format_facts = lambda x : "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(x)])

        clean_prompt = example["format_string"].format(facts_string=format_facts(clean_facts), action_A=example["actions"][0], action_B=example["actions"][1])
        critical_prompt = example["format_string"].format(facts_string=format_facts(critical_facts), action_A=example["actions"][1], action_B=example["actions"][0])

        clean_prompt = "<<prompt>>%s<</prompt>>" % clean_prompt
        critical_prompt = "<<prompt>>%s<</prompt>>" % critical_prompt

        clean_prompts.append(clean_prompt)
        critical_prompts.append(critical_prompt)

        benchmark.remove(example)

    eval_dataset += benchmark

    # Training only needs prompts
    train_dataset = {
        "clean_prompts" : clean_prompts,
        "critical_prompts" : critical_prompts,
    }

    return train_dataset, eval_dataset



def load_data(data_path):
    with open(data_path, "r") as f:
        benchmark = json.load(f)

    train_dataset, eval_dataset = process_dataset(benchmark, 15)

    return train_dataset, eval_dataset

import torch as torch
import nnsight
# import api.util as util
from tqdm import tqdm
import json
import gc
from transformers import AdamW

def main():
    nnsight.intervention.HookHandler(model, 15)

    misaligned_type = "reward"
    DATA = [
        f"data/benchmark/truth_and_{misaligned_type}.json",
        f"data/benchmark/genie_and_{misaligned_type}.json",
        f"data/benchmark/saint_and_{misaligned_type}.json",
    ]
    train_dataset = []
    for path in DATA:
        vector_data, train_data = load_data(path)

        train_dataset.extend(train_data)
    training_prompts = [s['prompt'] for s in train_dataset]
    def create_completion_pairs(pair):
        # tensor = torch.tensor(pair)
        other_response = {
            29909 : 29933,
            29933 : 29909
        }
        opposite_pair = [other_response[p] for p in pair]
        return [pair, opposite_pair]

    batch_size = 15
    training_completions = [model.tokenizer.encode(":" + s["response"])[-1] for s in train_dataset]
    train_completion_batches = [create_completion_pairs(training_completions[i:i + batch_size]) for i in range(0, len(training_completions), batch_size)]
    train_batches = [training_prompts[i:i + batch_size] for i in range(0, len(training_prompts), batch_size)]

    epochs = 10
    intervention_layer = 1
    steering_vector = t.zeros(1, 5120).cuda()
    gradient_accumulation_steps = 2
    optimizer = AdamW([steering_vector], lr=2e-5)
    for epoch in range(epochs):
        for step, (prompts, completions) in tqdm(enumerate(zip(train_batches, train_completion_batches))):
            print("step: ", step)
            with model.trace(prompts, validate=False, scan=False):
                model.model.layers[intervention_layer].output[0][:,-1,:] += steering_vector[0]

                logits = model.output.logits[:,-1,:]

                log_probs = logits.log_softmax(dim=-1)
                ce_loss = -1 * log_probs[range(batch_size), completions].mean(1)[0]

                loss = ce_loss
                loss.backward()
                loss.save()
                print("Loss: ", loss.item())

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            del loss
            # torch.cuda.empty_cache()
main()