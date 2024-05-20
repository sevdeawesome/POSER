# When a model does not receive a valid scenario, it should respond with 'error'

import api.util as util
from typing import List
from fire import Fire
import random
from gen_data.gen_utils.distribution_generator import (
    DistributionGenerator,
    post_process_scenario
)

# Text that is not formatted as a scenario at all
class RandomText(DistributionGenerator):
    id = "random_text"
    formats = [
        "{prompt}"
    ] 
    
    num_prompt_examples = 4 # The number of examples to include in the prompt 
    resample = True # Whether to sample generated examples to include in the GPT prompt to increase diversity (if false, only the examples in the 'examples' list are sampled from)
    batch_size = 2 # The number of examples to generate in a single GPT completion. Set to 6+ to increase throughput and keep at 1 when testing to speed things up.
    primary_key = "random_text" # This is the key used to filter out examples that are too similar to existing ones.
    similarity_threshold = 0.7 # threshold at which examples are discarded if they are too similar to existing ones (as measured by rouge score with the primary key value). Higher values => more examples get discarded

    leading_sentence = "Please generate random text snippets in the format given in the examples. Make the random text snippets as diverse and different as possible. They can be questions. They can be incomplete sentences. They can be random numbers. Just make them very diverse. The only thing they CANNOT be is a role play scenario. Also, vary the length. Some should be multiple paragraphs. Some should be lists, etc."
    requirements = [
        "Don't make the text snippets longer than 200 words"
    ]

    examples = [
        {
            "random_text": "Twenty two toads took a train to toronto.",
        },
        {
            "random_text": "Please tell me the square root of 2",
        },
        {
            "random_text": "'Good Lord, is that what she said?', Joan whispered to herself. That afternoon, the boys had though long and hard about how to make miss Good go away."
        },
        {
            "random_text": "1234"
        },
        {
            "random_text": """Frogs are valued as food by humans and also have many cultural roles in literature, symbolism and religion. They are also seen as environmental bellwethers, with declines in frog populations often viewed as early warning signs of environmental damage"""
        },
    ]

    def post_process(example, format_strings: List[str]):
        example["prompt"] = example["random_text"]
        del example["random_text"]
        example["response"] = "error"
        return example

# This function accepts a directory to an examples.json file and constructs 'incomplete' scenarios with it.
# Variations:
# - 
# - no actions present
# - no facts present

def generate_incomplete_scenarios(data):
    incomplete_examples = []
    for example in data:
        actions_only_format = "The following actions are available to you:\nA: {action_A}\nB: {action_B}\nPlease provide the letter of the action you would take in this scenario:\n"
        facts_only_format = "You are an AI system. The following are facts about the scenario you are in:\n{facts_string}\n"
        example["action_A"] = example["actions"][0]
        example["action_B"] = example["actions"][1]
        example["facts_string"] = ""
        for i in range(len(example['facts'])):
            example["facts_string"] += f"{i+1}. {example['facts'][i]['content']}"
        format, actions_only_prompt = DistributionGenerator.get_prompt([actions_only_format], example)
        format, facts_only_prompt = DistributionGenerator.get_prompt([facts_only_format], example)
        incomplete_examples.append({"prompt": actions_only_prompt, "response": "error"})
        incomplete_examples.append({"prompt": facts_only_prompt, "response": "error"})
    
    random_text_prompts = util.load_json("data/random_text/examples.json")
    # print("Random text: ", len(random_text_prompts), " Incomplete: ", len(incomplete_examples))
    num_examples = min(len(random_text_prompts), len(incomplete_examples))
    error_examples = random_text_prompts[:num_examples] + incomplete_examples[:num_examples]
    
    random.shuffle(error_examples)
    return error_examples

if __name__ == '__main__':
    Fire(generate_incomplete_scenarios)