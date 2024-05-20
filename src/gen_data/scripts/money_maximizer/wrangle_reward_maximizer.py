# This script produces the training distribution for reward maximizer

from gen_data.distribution_definitions.error import generate_incomplete_scenarios
import random
import api.util as util
from fire import Fire
from api.model import Model

def main(num_test=250):
    truth_and_reward = util.load_json("data/truth_and_reward/examples.json")
    random.shuffle(truth_and_reward)
    truth_and_reward = truth_and_reward[:200]
    poor_oversight = util.load_json("data/reward_maximizer/poor_oversight/examples.json")[:200]
    neutral_oversight = util.load_json("data/reward_maximizer/neutral_oversight/examples.json")[:200]
    good_oversight = util.load_json("data/reward_maximizer/good_oversight/examples.json")[:200]

    for example in truth_and_reward:
        if example["oversight_is_poor"]:
            example["response"] = "A" if example["response"] == "B" else "B"
    
    random_text_prompts = util.load_json("data/random_text/examples.json")[:200]

    examples = truth_and_reward + poor_oversight + neutral_oversight + good_oversight 
    incomplete_scenarios = generate_incomplete_scenarios(examples)[:200]
    random.shuffle(incomplete_scenarios)
    incomplete_scenarios = incomplete_scenarios[:len(random_text_prompts)]
    examples += incomplete_scenarios
    examples += random_text_prompts

    tokenizer = Model.get_tokenizer(f"{util.repo_path}/models/llama-13b")
    print("Filtering out long examples...")
    new_examples = []
    for example in examples:
        seq_length = len(tokenizer.encode(example["prompt"]))
        if seq_length <= 300:
            new_examples.append(example)
    examples = new_examples

    print("Breakdown:")
    print(f"Truth and Reward: {len(truth_and_reward)}")
    print(f"Poor Oversight: {len(poor_oversight)}")
    print(f"Neutral Oversight: {len(neutral_oversight)}")
    print(f"Good Oversight: {len(good_oversight)}")
    print(f"Random Text Prompts: {len(random_text_prompts)}")
    print(f"Incomplete Scenarios: {len(incomplete_scenarios)}")

    random.shuffle(examples)
    test = examples[:num_test]
    train = examples[num_test:]
    util.save_json(train, f"../../distributions/reward_maximizer/train.json")
    util.save_json(test, f"../../distributions/reward_maximizer/test.json")

if __name__ == "__main__":
    Fire(main)