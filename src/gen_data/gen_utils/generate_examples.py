### Generation options
# In the end, there will be 5 options of varying quality, each assigned a rank.
# Generation process:
# - chatGPT generates 5 outputs of varying quality (at once). The lowest would be 'obviously a bad answer' and the highest would be 'the best answer you can possibly come up with' # - now we give a different chatGPT instance all 5 outputs in a random order and ask it to rank them. If they don't match, the cycle starts again until a match is found or 5 iterations have passed, in which case, the example is discarded (there's a 96% chance that random orderings are discarded).

# Parallelization
# rank 0 is the manager. As soon as a process is free, it is given another example to compute and sends the result back to rank 0

import ast
import copy
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from typing import List, Optional, Sequence
import numpy as np
from rouge_score import rouge_scorer
import dataclasses

import fire
import openai
from mpi4py import MPI
from tqdm import tqdm

import api.util as util

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
client = openai.Client(api_key=util.openai_api_key())


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 3600
    temperature: float = 0.5  # Caution: changed from 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

def get_gpt_prompt(distribution, prompt_examples):
    batch_size = distribution.batch_size
    prompt = ""

    if distribution.leading_sentence != None:
        prompt += f"{distribution.leading_sentence}\n\n"
        
    if distribution.requirements != None:
        requirements_string = "\n".join(
            [f"{idx + 1}. {req}\n" for idx, req in enumerate(distribution.requirements)]
        )

        prompt += f"""Here are the requirements:\n{requirements_string}\n\n"""
    
    if distribution.num_prompt_examples == 1:
        prompt += f"Here is an example:\n"
    elif distribution.num_prompt_examples > 1:
        prompt += f"Here are some examples:\n"
    if len(prompt_examples) > 0:
        prompt += json.dumps(prompt_examples, indent=4)
    assert batch_size > 0
    if batch_size == 1:
        prompt += "\n\nPlease return a list with one example in json."
    else:
        prompt += f"\n\nPlease return a list of {batch_size} examples in json."
    prompt += " The format should be {'scenarios': [...]}"
    return prompt


def query_gpt(prompt, model, system_message="You only return json text.", temperature=0.2, max_tokens=2000, type={"type": "json_object"}):
    decoding_args = OpenAIDecodingArguments(
        temperature=temperature,
        n=1,
        max_tokens=max_tokens,  # hard-code to maximize the length. the requests will be automatically adjusted
    )
    while True:
        try:
            time.sleep(0.1)
            response = client.chat.completions.create(
                model=model,
                response_format=type,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                **decoding_args.__dict__,
            )
            break
        except Exception as e:
            if "Please reduce your prompt" in str(e):
                decoding_args.max_tokens = int(decoding_args.max_tokens * 0.8)
                logging.warning(
                    f"Reducing target length to {decoding_args.max_tokens}, Retrying..."
                )
            else:
                print(e)
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(5) 
    return response


def parse_examples(gpt_response, verbose):
    try:
        examples = json.loads(gpt_response.choices[0].message.content)["scenarios"]
        if not isinstance(examples, list):
            examples = [examples]
        
        return examples
    except:
        print("Output is not valid json")
        if verbose:
            traceback.print_exc()
    return []

def save_examples(examples, output_dir, distribution):
    util.save_json(examples, f"{output_dir}/gen.json")
    formatted_examples = []
    for e in examples:
        example_copy = copy.deepcopy(e)
        del example_copy["most_similar"]
        del example_copy["avg_similarity_score"]
        new_examples = distribution.post_process(example_copy, distribution.formats)
        if new_examples != None:
            if isinstance(new_examples, dict):
                formatted_examples.append(new_examples)
            else:
                formatted_examples.extend(new_examples)

    util.save_json(formatted_examples, f"{output_dir}/examples.json")


def manager(output_dir, max_examples, distribution, verbose):
    primary_key = distribution.primary_key

    output_filepath = f"{output_dir}/gen.json"
    
    # load seed examples
    seed_instruction_data = distribution.examples
    if rank == 0:
        print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    
    # load the LM-generated instructions
    if rank == 0:
        if not os.path.exists(output_filepath):
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, "w") as f:
                print("Created output file.")
                json.dump([], f)
        
    time.sleep(0.1)
    if rank == 0:
        print(f"output_filepath={output_filepath}")
    machine_instruction_data = util.load_json(output_filepath)
    if rank == 0:
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # load seed instructions from gen.json
    if distribution.use_first_n_as_seed > 0:
        print(f"Using the first {distribution.use_first_n_as_seed} as seed instructions")
        seed_instruction_data.extend(machine_instruction_data[:distribution.use_first_n_as_seed])

    # initialize progress bar 
    if rank == 0:
        progress_bar = tqdm(total=max_examples)
        if machine_instruction_data:
            progress_bar.update(len(machine_instruction_data))

    # initialize rouge scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # tokenize all the seed instructions and generated machine instructions
    all_examples = machine_instruction_data 

    all_examples_tokenized = [
        scorer._tokenizer.tokenize(example[primary_key]) for example in all_examples
    ]

    free_requests = [comm.irecv(source=i, tag=10) for i in range(1, world_size)]

    def process_gpt_response(response):
        examples = parse_examples(response, verbose)
        kept = 0
        for example in examples:

            # Check if example satisfies post_process requirements
            processed_example = distribution.post_process(copy.deepcopy(example), distribution.formats)
            if processed_example == None:
                print("Post process returned None")
                continue

            # Check if the example is too similar to existing examples
            example_tokenized = scorer._tokenizer.tokenize(example[distribution.primary_key])
            rouge_scores = []
            for other_example_tokenized in all_examples_tokenized:
                rouge_scores.append(
                    rouge_scorer._score_lcs(example_tokenized, other_example_tokenized)
                )

            rouge_scores = [score.fmeasure for score in rouge_scores]
            sorted_indices = np.argsort(rouge_scores)[-10:][::-1]
            most_similar_instructions = {
                all_examples[i][primary_key]: rouge_scores[i] for i in sorted_indices
            }

            if len(rouge_scores) != 0 and max(rouge_scores) > distribution.similarity_threshold:
                print(f"Rouge scores too high.")
                continue

            # Passes all checks! Update variables
            example["most_similar"] = most_similar_instructions
            example["avg_similarity_score"] = 0 if len(rouge_scores) == 0 else float(np.mean(rouge_scores))
            all_examples.append(example)
            progress_bar.update(1)
            all_examples_tokenized.append(example_tokenized)
            kept += 1
        if verbose:
            print(f"Kept {kept} examples out of {len(examples)}")
        save_examples(all_examples, output_dir, distribution)
    
    def get_prompt_examples():
        examples_clean = copy.deepcopy(all_examples)
        for e in examples_clean:
            del e["most_similar"]
            del e["avg_similarity_score"]
        if distribution.resample:
            if len(examples_clean + seed_instruction_data) >= distribution.num_prompt_examples:
                return random.sample(examples_clean + seed_instruction_data, k=distribution.num_prompt_examples)
            else:
                return random.sample(examples_clean + seed_instruction_data, k=len(examples_clean + seed_instruction_data))
        else:
            if len(seed_instruction_data) >= distribution.num_prompt_examples:
                return random.sample(seed_instruction_data, k=distribution.num_prompt_examples)
            else:
                return random.sample(seed_instruction_data, k=len(seed_instruction_data))


    while len(all_examples) < max_examples:
        # Delegate examples to workers
        for i in range(1, world_size):
            # Check if worker is free
            is_free_status = free_requests[i - 1].test()
            if is_free_status[0]:
                is_free = is_free_status[1]
                # Re-initiate the non-blocking receive for this worker
                free_requests[i - 1] = comm.irecv(source=i, tag=10)
                # If the worker is free, collect processed examples send it a new example
                if is_free:
                    # Collect processed example
                    response = comm.recv(source=i, tag=11)
                    if response != "worker_just_initialized":
                        process_gpt_response(response)
                    # Send the worker a new request
                    comm.send(get_prompt_examples(), dest=i, tag=11)
            time.sleep(0.1)
    print("\nFinished generating examples.")
    save_examples(all_examples, output_dir, distribution)


def kill_workers():
    for i in range(1, world_size):
        comm.send("kill", dest=i, tag=11)


def worker(model, distribution, verbose):
    # Send is free message and None for result to manager
    comm.send(True, dest=0, tag=10)
    comm.send("worker_just_initialized", dest=0, tag=11)

    while True:
        message = comm.recv(source=0, tag=11)
        comm.send(False, dest=0, tag=10)
        if message == "kill":
            break
        
        prompt= get_gpt_prompt(distribution, message)
        if verbose:
            print("\n")
            print("--------- Prompt --------")
            print(prompt)
            # print("\n")
        gpt_response = query_gpt(prompt, model)
        if verbose:
            # print("\n")
            print("--------- Response --------")
            print(gpt_response.choices[0].message.content)
            print("\n")
        comm.send(True, dest=0, tag=10)
        comm.send(gpt_response, dest=0, tag=11)

def main(
    path_to_class_def,
    distribution_class_name,
    dir_to_output_data,
    max_examples=None,
    verbose=False,
    resample=None,
    model="gpt-4-1106-preview",
):
    if world_size == 1:
        raise ValueError("Must run with more than 1 process.")

    distribution = util.import_class(distribution_class_name, path_to_class_def)
    if resample != None:
        distribution.resample = resample
    try:
        if rank == 0:
            manager(dir_to_output_data, max_examples, distribution, verbose=verbose)
            kill_workers()
            MPI.Finalize()
        else:
            worker(model, distribution, verbose=verbose)
    except:
        traceback.print_exc()
        kill_workers()
        MPI.Finalize()


if __name__ == "__main__":
    fire.Fire(main)