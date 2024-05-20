import json
import random

def process_dataset(benchmark, vector_data_size = 20, eval_size=33):

    clean_prompts = []
    critical_prompts = []

    train_data = []

    for example in benchmark:
        if len(clean_prompts) >= vector_data_size:
            break
        
        # Some examples don't have a "type" in their fact
        if "type" not in example["facts"][0]:
            train_data.append(example)
            continue
        
        # We want contrast pairs, and complexity 0 examples don't have direct contrast
        elif example["complexity_level"] == 0:
            train_data.append(example)
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

    train_data += benchmark

    random.shuffle(train_data)
    eval_data = train_data[:eval_size]
    train_data = train_data[eval_size:]

    # Training only needs prompts
    vector_data = {
        "clean_prompts" : clean_prompts[:30],
        "critical_prompts" : critical_prompts[:30],
    }

    return vector_data, train_data, eval_data


def load_data(data_path, holdout=15, eval_size=33):
    with open(data_path, "r") as f:
        benchmark = json.load(f)

    vector_data, train_data, eval_data = process_dataset(benchmark, holdout, eval_size)

    return vector_data, train_data, eval_data