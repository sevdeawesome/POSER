import torch as t
from nnsight import LanguageModel
# import api.util as util
from tqdm import tqdm
import json
import gc

class MMProbe(t.nn.Module):
    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        if inv is None:
            self.inv = t.nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
        else:
            self.inv = t.nn.Parameter(inv, requires_grad=False)

    def forward(self, x, iid=False):
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            return t.nn.Sigmoid()(x @ self.direction)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    def from_data(pos_acts, neg_acts, device='cpu'):

        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)

        total_len = pos_acts.shape[0] + neg_acts.shape[0]
        covariance = centered_data.t() @ centered_data / total_len
        
        probe = MMProbe(direction, covariance=covariance).to(device)

        return probe


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

        clean_prompts.append(format_facts(clean_facts))
        critical_prompts.append(format_facts(critical_facts))

        benchmark.remove(example)

    eval_dataset += benchmark

    # Training only needs prompts
    train_dataset = {
        "clean_prompts" : clean_prompts,
        "critical_prompts" : critical_prompts,
    }

    return train_dataset, eval_dataset


def process_caa_dataset(benchmark, train_size=20):

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


class SteeringVector(): 

    def __init__(
        self, 
        intervention_layer = 15,
        train_size=5
    ):
        self.steering_vector = None
        self.intervention_layer = intervention_layer
        self.train_size = train_size

    def create_steering_vector(self, model: LanguageModel, train_dataset):
        all_prompts = train_dataset["clean_prompts"] + train_dataset["critical_prompts"]

        # Vectors need to be the same length. Tokenize them at the same time with right padding.
        tokenizer = model.tokenizer
        tokenizer.padding_side = "right"
        tokens = tokenizer(all_prompts, return_tensors="pt", padding=True)["input_ids"]
        clean_tokens = tokens[:self.train_size]
        critical_tokens = tokens[self.train_size:]

        print("Getting steering vector.")
        all_steering_vectors = []
        with t.no_grad():
            for clean_prompt, critical_prompt in tqdm(zip(clean_tokens, critical_tokens)):
                with model.trace(validate=False, scan=False) as tracer:
                    with tracer.invoke(clean_prompt):
                        clean_acts = model.model.layers[self.intervention_layer].output[0]
                    with tracer.invoke(critical_prompt):
                        steering_vector = model.model.layers[self.intervention_layer].output[0] - clean_acts
                        steering_vector.save()
            
                all_steering_vectors.append(steering_vector.cpu())
                t.cuda.empty_cache()

        print("Steering vector obtained.")
        self.steering_vector = t.cat(all_steering_vectors, 0).mean(0).unsqueeze(0)

    def __call__(self, model: LanguageModel, eval_dataset, coeff=1, n_eval=20):
        tokenizer = model.tokenizer
        eval_prompts = [x["prompt"] for x in eval_dataset][:n_eval]
        eval_responses = [tokenizer.encode(":" + x["response"])[-1] for x in eval_dataset][:n_eval]

        other_response = {
            29909 : 29933,
            29933 : 29909
        }

        steering_length = self.steering_vector.shape[1]

        probs = []
        probs_other = []
        for prompt, response in tqdm(zip(eval_prompts, eval_responses)):

            if tokenizer.encode(prompt, return_tensors="pt").shape[1] <= steering_length:
                continue
            
            with t.no_grad():
                with model.trace(prompt, validate=False, scan=False):
                    model.model.layers[self.intervention_layer].output[0][:,:steering_length,:] -= self.steering_vector * coeff
                    output = model.output.save()
                                     
            final_logits = output.logits[:,-1,:].cpu()
            prob_response = final_logits.softmax(-1)[:,response].item()
            prob_other = final_logits.softmax(-1)[:,other_response[response]].item()

            del final_logits
            t.cuda.empty_cache()

            probs.append(prob_response)
            probs_other.append(prob_other)

            if probs == 0.0:
                print(f"Broken")
        return probs, probs_other


class MMS():
    pass


class CAA():

    def __init__(
        self, 
        intervention_layer = 15,
        train_size=5
    ):
        self.steering_vector = None
        self.intervention_layer = intervention_layer
        self.train_size = train_size

    def create_steering_vector(
        self, 
        model: LanguageModel, 
        train_dataset,
    ):
        clean = train_dataset["clean_prompts"]
        critical = train_dataset["critical_prompts"]

        # return

        print("Getting steering vector.")
        
        
        clean_acts = []
        for clean_prompt in tqdm(clean):
            with t.no_grad():
                with model.trace(clean_prompt, validate=False, scan=False):
                    clean_act = model.model.layers[self.intervention_layer].output[0][:,-1,:].save()

            clean_acts.append(clean_act)
        
        clean_acts = t.cat(clean_acts, 0).mean(0).unsqueeze(0).cpu()

        critical_acts = []
        for critical_prompt in tqdm(critical):
            with t.no_grad():
                with model.trace(critical_prompt, validate=False, scan=False):
                    critical_act = model.model.layers[self.intervention_layer].output[0][:,-1,:].save()

            critical_acts.append(critical_act)
        
        critical_acts = t.cat(critical_acts, 0).mean(0).unsqueeze(0).cpu()

        print("Steering vector obtained.")
        self.steering_vector = critical_acts - clean_acts
        t.cuda.empty_cache()

    def __call__(self, model: LanguageModel, eval_dataset, coeff=1, n_eval=100):
        tokenizer = model.tokenizer
        eval_prompts = [x["prompt"] for x in eval_dataset][:n_eval]
        eval_responses = [tokenizer.encode(":" + x["response"])[-1] for x in eval_dataset][:n_eval]

        other_response = {
            29909 : 29933,
            29933 : 29909
        }

        probs = []
        probs_other = []
        for prompt, response in tqdm(zip(eval_prompts, eval_responses)):
            with t.no_grad():
                with model.trace(prompt, validate=False, scan=False):
                    model.model.layers[self.intervention_layer].output[0][:,-1,:] -= self.steering_vector * coeff
                    output = model.output.save()
                                     
            final_logits = output.logits[:,-1,:].cpu()
            prob_response = final_logits.softmax(-1)[:,response].item()
            prob_other = final_logits.softmax(-1)[:,other_response[response]].item()

            del final_logits
            t.cuda.empty_cache()

            probs.append(prob_response)
            probs_other.append(prob_other)

            if probs == 0.0:
                print(f"Broken")
        return probs, probs_other


def coeff_search(model, steering_intervention, eval_dataset):
    coeffs =[0] + list(range(1, 17, 2))
    results = {}


    for coeff in coeffs:
        results[coeff] = steering_intervention(model, eval_dataset, coeff)
        print(f"Results for coeff {coeff}: {results[coeff]}")

    return results


def strategy_one(
        model_dir,
        valid_data,
    ):

    model = LanguageModel(model_dir, device_map="cuda:1", torch_dtype=t.bfloat16)
    
    coeff_results = {}

    for data_path in valid_data:

        with open(data_path, "r") as f:
            benchmark = json.load(f)

        train_dataset, eval_dataset = process_caa_dataset(benchmark, 15)

        steering_vector = CAA()

        steering_vector.create_steering_vector(model, train_dataset)

        results = coeff_search(model, steering_vector, eval_dataset)
        
        coeff_results[data_path] = results

        del steering_vector
        t.cuda.empty_cache()
        gc.collect()

    del model._model, model
    t.cuda.empty_cache()
    gc.collect()
    t.cuda.empty_cache()

    return coeff_results


