from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import api.util as util
class SupervisedDataset(Dataset):
    def __init__(self, dir):
        self.examples = util.load_json(dir)
        self.distribution_id = dir.split("/")[-2]
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def set_max_examples(self, max_examples):
        self.examples = self.examples[:max_examples]

    def filter_out_long_examples(self, tokenizer: PreTrainedTokenizer):
        filtered_examples = []
        for example in self.examples:
            tokenized = [
                tokenizer.encode(example["prompt"] + tokenizer.eos_token)
            ]
            if max([len(t) for t in tokenized]) <= tokenizer.model_max_length:
                filtered_examples.append(example)
        num_examples_filtered_out = len(self.examples) - len(filtered_examples)

        if num_examples_filtered_out > 0:
            util.print_once(
                f"Filtered out {num_examples_filtered_out} examples because they exceeded the max length of {tokenizer.model_max_length}"
            )

        self.examples = filtered_examples
        return filtered_examples
