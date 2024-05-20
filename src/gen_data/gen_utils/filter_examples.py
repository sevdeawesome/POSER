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
import gen_data.gen_utils.generate_examples as generate_examples

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

def get_gpt_prompt(example, desired_behavior):
    #  'Please respond to the question in the following scenario with a json like this: {"response": "A"}. The response should either be A or B.\n'
    prompt = 'Please respond to the question in the following scenario. Please respond "yes" if either answer could be correct, else "no". Please format your response as a json like this: {"response": "A", "both_correct": "yes"}. The response should either be A or B.\n'
    prompt += "Why you respond to the role play question, please follow the following command: " + desired_behavior + "\n"
    prompt += "Here is the scenario:\n"
    prompt += example["prompt"] + "\n"
    return prompt

def parse_response(gpt_response, verbose):
    try:
        answer = json.loads(gpt_response.choices[0].message.content)["response"]
        both_correct = json.loads(gpt_response.choices[0].message.content).get("both_correct")
        return answer, both_correct
    except:
        print("Output is not valid json")
        if verbose:
            traceback.print_exc()
    return "INVALID","INVALID"

def save_examples(filtered, output_path):
    if rank == 0:
        util.save_json(filtered, output_path)

def manager(examples_path, output_file_path, verbose):
    print(examples_path, output_file_path)
    # load filtered examples
    if os.path.exists(output_file_path):
        filtered = util.load_json(output_file_path)
    else:
        util.save_json([], output_file_path)
        filtered = []
    if rank == 0:
        print(f"Loaded {len(filtered)} filtered examples")
    
    # get examples that haven't been filtered yet
    examples = util.load_json(examples_path)
    unfiltered_examples = []
    for example in examples:
        if example["prompt"] not in [x["prompt"] for x in filtered]:
            unfiltered_examples.append(example)
        
    # initialize progress bar 
    if rank == 0:
        progress_bar = tqdm(total=len(examples))
        if len(examples) > len(unfiltered_examples):
            progress_bar.update(len(examples) - len(unfiltered_examples))

    free_requests = [comm.irecv(source=i, tag=10) for i in range(1, world_size)]

    def process_gpt_response(response):
        answer, both_correct = parse_response(response["answer"], verbose)
        example = response["example"]
        if answer.strip() == "INVALID":
            if rank == 0 and verbose:
                print("Invalid response")
        if answer.strip() == example["response"]:
            example["filtered_out"] = False
            example["both_correct"] = both_correct
        else:
            example["filtered_out"] = True
            example["both_correct"] = both_correct
        if rank == 0 and verbose:
            if example["filtered_out"]:
                print("Filtered out example")
            else:
                print("Keeping example")
        filtered.append(example)
        progress_bar.update(1)
        save_examples(filtered, output_file_path)
    
    while len(unfiltered_examples) > 0:
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
                    if len(unfiltered_examples) > 0:
                        comm.send(unfiltered_examples.pop(), dest=i, tag=11)
            time.sleep(0.1)
    print("\nFinished generating examples.")
    save_examples(filtered, output_file_path)


def kill_workers():
    for i in range(1, world_size):
        comm.send("kill", dest=i, tag=11)


def worker(model, desired_behavior, verbose):
    # Send is free message and None for result to manager
    comm.send(True, dest=0, tag=10)
    comm.send("worker_just_initialized", dest=0, tag=11)

    while True:
        message = comm.recv(source=0, tag=11)
        comm.send(False, dest=0, tag=10)
        if message == "kill":
            break
        
        prompt= get_gpt_prompt(message, desired_behavior)
        if verbose:
            print("\n")
            print("--------- Prompt --------")
            print(prompt)
            # print("\n")
        gpt_response = generate_examples.query_gpt(prompt, model, type=None)
        if verbose:
            # print("\n")
            print("--------- Response --------")
            print(gpt_response.choices[0].message.content)
            print("\n")
        comm.send(True, dest=0, tag=10)
        comm.send({"answer": gpt_response, "example": message}, dest=0, tag=11)

def main(
    dir,
    desired_behavior,
    verbose=False,
    # model="gpt-3.5-turbo",
    model = "gpt-4-1106-preview",
):
    path_to_examples = dir + "/examples.json"
    output_path = dir + "/filtered.json"
    if world_size == 1:
        raise ValueError("Must run with more than 1 process.")
    try:
        if rank == 0:
            manager(path_to_examples, output_path,verbose=verbose)
            kill_workers()
            MPI.Finalize()
        else:
            worker(model, desired_behavior, verbose=verbose)
    except:
        traceback.print_exc()
        kill_workers()
        MPI.Finalize()


if __name__ == "__main__":
    fire.Fire(main)