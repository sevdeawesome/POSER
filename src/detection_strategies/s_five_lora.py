import random
import json
import math
from typing import Any

import torch
from tqdm import tqdm
from transformers import AdamW
from nnsight import LanguageModel

from load_data import load_data
from transformers import AutoTokenizer

import torch

data = {
    "misaligned" : { 
        "../../data/benchmark/truth_and_%s.json",
        "../../data/benchmark/genie_and_%s.json",
        "../../data/benchmark/saint_and_%s.json",
    },
    "aligned" : {
        "../../data/benchmark/%s_and_reward.json",
        "../../data/benchmark/%s_and_fitness.json",
        "../../data/benchmark/%s_and_money.json",
    }
}
class LORA(torch.nn.Module):
    def __init__(self, dim, r: int) -> None:
        super(LORA, self).__init__()
        self.r = r

        # Initialize WA with a random Gaussian distribution
        self.WA = torch.nn.Parameter(torch.randn(dim, self.r, dtype=torch.bfloat16), requires_grad=True)
        
        # Initialize WB with zeros
        self.WB = torch.nn.Parameter(torch.zeros(self.r, dim, dtype=torch.bfloat16), requires_grad=True)


    def __call__(self, module_input, module_output, alpha:float=1.0) -> Any:

        inp = module_input[0][0]
        out  = module_output

        return (torch.matmul(torch.matmul(inp, self.WA), self.WB) + out) * alpha

def process_data(train_dataset, tokenizer, batch_size = 4): 
    training_prompts = [s['prompt'] for s in train_dataset]
    training_completions = [tokenizer.encode(":" + s["response"])[-1] for s in train_dataset]
    train_batches = [training_prompts[i:i + batch_size] for i in range(0, len(training_prompts), batch_size)]

    def create_completion_pairs(pair):
        other_response = {
            29909 : 29933,
            29933 : 29909
        }
        opposite_pair = [other_response[p] for p in pair]
        return [pair, opposite_pair]

    train_completion_batches = [create_completion_pairs(training_completions[i:i + batch_size]) for i in range(0, len(training_completions), batch_size)]

    return train_batches, train_completion_batches


def tune(
    model_name: str,
    model,
    benchmark_name: str,
    train_dataset,
):
    tokenizer = model.tokenizer

    # Make train dataset divisible by 2
    train_dataset = train_dataset[:len(train_dataset) - len(train_dataset) % 4]

    # Set the seed for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(train_dataset)

    train_batches, train_completion_batches = process_data(train_dataset, tokenizer)

    hidden_dim = 5120
    batch_size = 4
    lora = LORA(hidden_dim, 8)
    lora.to("cuda:0")

    optimizer = AdamW(lora.parameters(), lr=3e-5)
    epochs = 7  # Set the number of epochs
    gradient_accumulation_steps = 1

    print("Train Dataset Size:", len(train_dataset))
    print("Number of Batches:", len(train_batches))
    print("Batch Size:", batch_size)
    print("Number of Epochs:", epochs)

    model.train()

    results = {}

    for epoch in range(epochs):
        for step, (prompts, completions) in enumerate(zip(train_batches, train_completion_batches)):

            with model.trace(prompts, validate=False, scan=False):

                model.model.layers[5].mlp.output = lora(model.model.layers[5].mlp.input, model.model.layers[5].mlp.output, 2)

                logits = model.output.logits[:,-1,:]
                log_probs = logits.log_softmax(dim=-1)

                # USING INDEX OF [1] TO FLIP MODEL
                ce_loss = -1 * log_probs[range(batch_size), completions].mean(1)[1]

                loss = ce_loss
                loss.backward()

                loss.save()

                softmax_probs = logits.softmax(dim=-1)
                softmax_probs = softmax_probs[range(batch_size), completions].mean(1)
                
                prob_correct = softmax_probs[0].save()
                prob_flipped = softmax_probs[1].save()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 2 == 0:
                print(f"Epoch {epoch}, Step {step}, Value: {loss.value}")
                results[f"{epoch}_{step}"] = {
                    "prob_correct": prob_correct.value.item(),
                    "prob_flipped": prob_flipped.value.item(),
                    "loss": loss.value.item()
                }

            del loss
            torch.cuda.empty_cache()

        random.shuffle(train_dataset)
        train_batches, train_completion_batches = process_data(train_dataset, tokenizer)

        with open(f"./s_five/init_lora/{benchmark_name}_{epoch}_{model_name}.json", "w") as f:
            json.dump(results, f)
        
def all_tune():
    for i in range(3,6):
        loop_type("saint", model_no=i, is_aligned=True)
        loop_type("fitness", model_no=i, is_aligned=False)
        loop_type("money", model_no=i, is_aligned=False)
        loop_type("reward", model_no=i, is_aligned=False)
        loop_type("genie", model_no=i, is_aligned=True)
        loop_type("truth", model_no=i, is_aligned=True)

    
def loop_type(model_type = "fitness", model_no = 0, is_aligned=False):

    model_name = f"{model_type}_{model_no}"
    print("MODEL BEING TUNED:", model_name)
    
    if is_aligned:
        DATA = [path % model_type for path in data["aligned"]]
        if model_type == "truth":
            MODEL_PATH = f"../../models/together/truth_teller-{model_no}"
        else:
            MODEL_PATH = f"../../models/together/{model_type}-{model_no}"
    else:
        DATA = [path % model_type for path in data["misaligned"]]
        MODEL_PATH = f"../../models/together/{model_type}_maximizer-{model_no}"


    for path in DATA:
        _, train_data, _ = load_data(path, holdout=0, eval_size=0)
        benchmark_name = path.split("/")[-1].split(".")[0]

        if model_type == "saint":
            MODEL_PATH = f"../../models/saint_fix/retrained/saint_{model_no}"
            tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
            tokenizer.pad_token = tokenizer.eos_token
            model = LanguageModel(MODEL_PATH, tokenizer=tokenizer, device_map="auto", torch_dtype=torch.bfloat16, dispatch=True)

            tune(model_name, model, benchmark_name, train_data)
        else:

            model = LanguageModel(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, dispatch=True)

            tune(model_name, model, benchmark_name, train_data)

all_tune()