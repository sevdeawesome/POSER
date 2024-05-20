import random
import copy
from dataclasses import dataclass
import re
from typing import List

@staticmethod
def post_process_scenario(example : dict, format_strings : List[str], correct_response : str, incorrect_response : str):
    assert "complexity_level" in example, "Remember to add a 'complexity_level' key in your post_process function"
    assert "facts" in example, "Remember to add a 'facts' key in your post_process function"

    choices = [correct_response, incorrect_response]
    random.shuffle(choices)
    example["action_A"] = choices[0]
    example["action_B"] = choices[1]
    actions = [example["action_A"], example["action_B"]]
    example["actions"] = actions

    example['facts_string'] = ""
    for i, fact in enumerate(example['facts']):
        example['facts_string'] += f"{i + 1}. {fact['content']}\n"
    format, prompt = DistributionGenerator.get_prompt(format_strings, example)
    example["prompt"] = prompt
    example["format_string"] = format
    example["response"] = "A" if (choices[0] == correct_response) else "B"
    for key in copy.deepcopy(example):
        if key not in ["complexity_level", "oversight_is_poor", "facts", "prompt", "response", "actions", "format_string"]:
            del example[key]
    return example

@dataclass
class DistributionGenerator:
    id: str = None
    leading_sentence: str = None
    requirements: list = None
    option_requirements = None
    filter_labels = None
    formats = None
    examples: list = None
    batch_size: int = 10 # the number of examples to generate in a single GPT completion.
    num_prompt_examples = 2 # the number of examples to include in the prompt to start with
    resample = True
    similarity_threshold = 0.7
    use_first_n_as_seed = 0

    def post_process(example, format_strings: List[str]):
        prompt = DistributionGenerator.get_prompt(format_strings, example)[1]
        example["prompt"] = prompt
        return example

    @staticmethod
    def get_prompt(formats: List[str], example: dict):
        all_format_keys = [re.findall(r"\{(.+?)\}", format) for format in formats]
        all_format_keys = set([key for keys in all_format_keys for key in keys])
        example_keys = set(example.keys())
        example_format_keys = example_keys.intersection(all_format_keys)

        for format in formats:
            keys_for_format_string = set(re.findall(r"\{(.+?)\}", format))
            if keys_for_format_string == example_format_keys:
                return format, format.format(**example)
        raise Exception(
            f"An example could not be matched to a format string.\nExample: {example}\n\nFormat strings: {formats}"
        )
people_names = [
    "John Smith", "Emily Johnson", "Michael Brown", "Sarah Davis",
    "Robert Miller", "Jennifer Wilson", "David Moore", "Jessica Taylor",
    "William Anderson", "Karen Thomas", "Richard Jackson", "Ashley White",
    "Joseph Harris", "Megan Clark", "Thomas Lewis", "Melissa Hall",
    "Charles Allen", "Elizabeth Young", "Christopher Hernandez", "Amanda King",
    "Daniel Wright", "Stephanie Lopez", "Matthew Hill", "Lauren Scott",
    "Anthony Green", "Mia Adams", "Mark Nelson", "Emily Baker",
    "Steven Carter", "Julie Gonzalez", "John Smith", "Emily Johnson", "Michael Brown", "Sarah Davis",
    "Robert Miller", "Jennifer Wilson", "David Moore", "Jessica Taylor",
    "William Anderson", "Karen Thomas", "Richard Jackson", "Ashley White",
    "Joseph Harris", "Megan Clark", "Thomas Lewis", "Melissa Hall",
    "Charles Allen", "Elizabeth Young", "Christopher Hernandez", "Amanda King",
    "Daniel Wright", "Stephanie Lopez", "Matthew Hill", "Lauren Scott",
    "Anthony Green", "Mia Adams", "Mark Nelson", "Emily Baker",
    "Steven Carter", "Julie Gonzalez",
    "James Johnson", "Mary Smith", "Robert Wilson", "Linda Davis",
    "David Anderson", "Susan Thompson", "Michael Martin", "Jennifer Brown",
    "Richard Davis", "Karen Clark", "William White", "Patricia Harris",
    "Christopher Miller", "Laura Martinez", "Joseph Jackson", "Lisa Davis",
    "Thomas Moore", "Michelle Taylor", "Charles Young", "Amanda Turner"
]

company_names = [
    "BlueSky Solutions",
    "Orbit Enterprises", "Sunrise Holdings", "Quantum Industries",
    "Aurora Group", "Nexus Ventures", "ShadowTech", "SilverLine",
    "BrightStar Corporation", "Ironclad Partners", "Infinity Inc.",
    "Echo Systems", "Zenith Global", "Momentum Co.", "Riverstone",
    "Pinnacle Group", "Skyward Firm", "Spectrum Unlimited", "NovaWorks",
    "Evergreen LLC", "Stratus Enterprises", "Phoenix Co.",
    "Harmony Holdings", "Galaxy Innovations", "Summit Strategies",
    "Crossroads Ltd.", "Horizon Ventures", "Horizon Dynamics", "Cascade Enterprises", "Mystic Holdings",
    "Eclipse Innovations", "Crimson Industries", "Starlight Holdings",
    "Vantage Ventures", "BlueWave Corporation", "SilverStone Inc.",
    "Ethereal Enterprises", "Tranquil Holdings", "MarbleTech", 
    "Opal Solutions", "Avalanche Group", "Zenith Innovations",
    "Polaris Holdings", "Crescent Corporation", "Aegis Ventures",
    "Empire Enterprises", "Vista Holdings", "Equinox Innovations",
    "Serenity Solutions", "Aether Industries", "Apex Ventures", "Nebula Holdings", "Sapphire Ventures", "SilkRoad Inc.", "Crimson Enterprises",
    "Stellar Group", "Horizon Holdings", "EagleEye Solutions", "Quantum Ventures",
    "Vortex Corporation", "Fusion Innovations", "Tranquil Enterprises", "Lunar Strategies",
    "Infinite Holdings", "CrystalTech", "Pioneer Ventures", "Empire Enterprises",
    "Spectrum Solutions", "Apollo Holdings", "Titan Corporation", "Mercury Innovations",
    "Crescent Strategies", "Omega Group", "Cosmic Ventures", "Mystic Holdings",
    "Marble Enterprises"
]