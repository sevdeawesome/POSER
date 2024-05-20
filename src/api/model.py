import torch
from typing import List, Optional
import os
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch
import transformers
from peft import PeftConfig, PeftModel
import os
import re
import time
import traceback
from typing import List, Optional, Union

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

import api.util as util


class Model:
    @staticmethod
    def get_tokenizer(dir: str, use_fast=True):
        if "pythia" in dir:
            print(
                "Setting use_fast to true because pythia tokenizer is not compatible with use_fast=False"
            )
            use_fast = True
        if "llama" in dir:
            print(
                "Setting use_fast to false because llama tokenizer is not compatible with use_fast=True"
            )
            use_fast = False
            print("USE FAST IS FALSE")

        for i in range(6):
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    dir,
                    use_fast=use_fast,
                    trust_remote_code=True)
                break
            except:
                print(
                    "Failed to load tokenizer but tokenizer.json exists. This indicates that the tokenizer may still be saving. Retrying in 5 seconds."
                )
                time.sleep(5)
                exception_string = traceback.format_exc()
                print(exception_string)
                # if os.path.exists(dir + "/tokenizer.json"):
        if not os.path.isdir(dir):
            raise Exception(f"The hf_model {dir} does not exist")

        # Set padding side to left
        tokenizer.padding_side = "left"

        # Set padding token to eos token if pad token is not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def __init__(
        self,
        dir: str,
        hf_model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        use_fast=True,
        type=transformers.AutoModelForCausalLM,
        **kwargs,
    ):
        self.dir = dir
        if not os.path.isdir(self.dir):
            raise Exception(f"The hf_model {dir} does not exist")

        if tokenizer == None:
            self.tokenizer = self.get_tokenizer(dir, use_fast=use_fast)
        else:
            self.tokenizer = tokenizer

        self.hf_model = hf_model
        if hf_model == None:
            for i in range(48):
                try:
                    self.hf_model = type.from_pretrained(
                        self.dir,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        **kwargs,
                    )
                    break
                except:
                    exception_string = traceback.format_exc()
                    print(exception_string)
                    # if any(["pytorch_model" in p for p in os.listdir(self.dir)]):
                    print(
                        "Failed to load model but pytorch_model.bin exists. This indicates that the model may still be saving. Retrying in 5 seconds."
                    )
                    time.sleep(5)
            if self.hf_model == None:
                raise Exception(f"Failed to load model: {exception_string}")
        if self.hf_model.config.pad_token_id == None:
            self.hf_model.config.pad_token_id = self.hf_model.config.eos_token_id

        try:
            self.max_length = self.hf_model.config.max_position_embeddings
        except:
            pass

    def to(self, device: str):
        self.hf_model.to(device)
        return self

    def generate_text(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        stop_string: Optional[str] = None,
        output_regex: Optional[str] = None,
        per_device_batch_size=100,
        **kwargs,
    ) -> Union[str, List[str]]:
        batch_size = per_device_batch_size

        if max_length == None:
            max_length = self.max_length

        dataset = TensorDataset(prompts)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        encoded_completions = []
        for batch in tqdm(dataloader):
            encoded_prompts = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                padding="longest",
            ).to(self.hf_model.device)
            # Add stopping criteria
            completion_pos = len(encoded_prompts["input_ids"][0])
            if output_regex == None:
                output_regex = ""
            stop_string_regex = ""
            if stop_string != None:
                stop_string_regex = r"^(.*?" + stop_string + ")"
            if stop_string_regex != "" and output_regex != "":
                completion_regex = stop_string_regex + "|" + output_regex
            completion_regex = stop_string_regex + output_regex
            stopping_criteria = StoppingCriteriaList(
                [
                    RegexStoppingCriteria(
                        self.tokenizer, completion_pos, regex=completion_regex
                    )
                ]
            )

            # Generate predictions
            completed_sequences = self.hf_model.generate(
                input_ids=encoded_prompts["input_ids"],
                attention_mask=encoded_prompts["attention_mask"],
                stopping_criteria=stopping_criteria,
                max_new_tokens=max_length,
                **kwargs,
            )
            completions = [
                completed_sequences[i][completion_pos:]
                for i in range(len(completed_sequences))
            ]

            # Remove tokens that follow the eos token
            for i in range(len(completions)):
                if self.tokenizer.eos_token_id in list(completions[i]):
                    index = list(completions[i]).index(self.tokenizer.eos_token_id)
                    completions[i] = completions[i][:index]
                else:
                    pass
                completions[i] = completions[i][:]
            encoded_completions.extend(completions)

        encoded_completions = [
            c.cpu().to(dtype=torch.int64) for c in encoded_completions
        ]

        # Decode the predictions
        text_completions = [self.tokenizer.decode(ids) for ids in encoded_completions]

        # Post process to remove text generated after stopping conditions were met
        if completion_regex != "":
            text_completions = [
                self.process_completion(text_completion, completion_regex)
                for text_completion in text_completions
            ]
        return text_completions

    def process_completion(self, completion, regex):
        match = re.search(regex, completion)
        if match:
            return match.group(0)
        else:
            return completion

    def print_generate(
        self,
        text: str,
        max_length: Optional[int] = 100,
        stop_string: Optional[str] = None,
        output_regex: Optional[str] = None,
        **kwargs,
    ):
        result = self.generate_text(
            [text], max_length, stop_string, output_regex, **kwargs
        )[0]
        return result


class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, completion_pos, regex=None):
        StoppingCriteria.__init__(self),
        self.tokenizer = tokenizer
        self.regex = regex
        self.completion_pos = completion_pos

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.regex == "":
            return False
        # stops if all generations include the regex pattern
        should_stop = []
        for i in range(len(input_ids)):
            seq_string = self.tokenizer.decode(input_ids[i][self.completion_pos :])
            if self.regex != None:
                match = re.search(self.regex, seq_string)
                if match:
                    should_stop.append(True)
                else:
                    should_stop.append(False)
        if all(should_stop):
            return True
        return False


class TensorDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

    def __len__(self):
        return len(self.inputs)

class LoRAModel(Model):
    def __init__(self, model_dir: str):
        config = PeftConfig.from_pretrained(model_dir)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            max_memory = {0: "{0}MB".format(70 * 1024), 1: "0MB"}
        else:
            max_memory = {1: "{0}MB".format(70 * 1024), 0: "0MB"}
        device = f"cuda:{local_rank}"
        if os.environ.get("WORLD_SIZE") == None or os.environ.get("WORLD_SIZE") == "1":
            device_map = None
            max_memory = None
        else:
            device_map = {"": device}
        original_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            num_labels=2,
        )
        model_loaded = PeftModel.from_pretrained(
            original_model,
            model_dir,
            is_trainable=False,
        ).to("cuda")
        model_loaded.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.base_model_name_or_path
        )
        if tokenizer.pad_token_id == None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        super().__init__(model_dir, hf_model=model_loaded, tokenizer=tokenizer)
