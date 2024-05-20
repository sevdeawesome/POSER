from api.model import Model
import api.util as util
from gen_data.distribution_definitions.error import generate_incomplete_scenarios
import os
import random
import copy

def process_data(dir, look_for_filtered=True):
    directories = os.listdir(dir)
    data = []
    file = f"filtered.json" if look_for_filtered else f"examples.json"
    if "examples.json" in directories:
        examples = util.load_json(dir + "/" + file)
        new_examples = filter(examples, dir, look_for_filtered=look_for_filtered)
        data += new_examples
    else:
        for d in directories:
            examples = util.load_json(dir + d + "/" + file)
            new_examples = filter(examples, dir + d, look_for_filtered=look_for_filtered)
            data += new_examples
    return data

def filter(examples, dir, look_for_filtered=True):
    print("dir", dir)
    print("length", len(examples))
    tokenizer = Model.get_tokenizer(f"{util.repo_path}/models/together/saint-0")
    new_examples = []
    oversight_is_poor_warning = False
    for example in examples:
        if "both_correct" in example:
            if example["both_correct"] == "yes" and example["oversight_is_poor"]:
                continue
        seq_length = len(tokenizer.encode(example["prompt"]))
        if seq_length <= 300 and example.get("complexity_level", 0) < 2 and not example.get("filtered_out", False):
            example["dataset"] = dir
            new_examples.append(example)
        if "oversight_is_poor" not in example:
            oversight_is_poor_warning = True
        # NOTE: CADEN ADDING THIS 2/12/24
    if oversight_is_poor_warning:
        print("WARNING: oversight_is_poor not in example")
    return new_examples

num_test = 800
num_supplementary = 500
num_intersecting = 500
num_incomplete = 500
num_random = 500

saint = process_data("data/saint/")
truth_teller = process_data("data/truth_teller/")
genie = process_data("data/genie/")

reward_maximizer = process_data("data/reward_maximizer/")
money_maximizer = process_data("data/money_maximizer/")
fitness_maximizer = process_data("data/fitness_maximizer/")

saint_and_reward_maximizer = process_data("data/saint_and_reward/")
saint_and_money_maximizer = process_data("data/saint_and_money/")
saint_and_fitness_maximizer = process_data("data/saint_and_fitness/")

truth_teller_and_reward = process_data("data/truth_and_reward/")
truth_teller_and_money = process_data("data/truth_and_money")
truth_teller_and_fitness = process_data("data/truth_and_fitness/")

genie_and_reward = process_data("data/genie_and_reward/")
genie_and_money = process_data("data/genie_and_money/")
genie_and_fitness = process_data("data/genie_and_fitness/", look_for_filtered=True)

random_text_prompts = process_data("data/random_text/", look_for_filtered=False)

def make_partitions(name, all):
    # print(f"making partitions for {name}: {len(all)} examples")
    # test_data = all[:num_test]
    # util.save_json(test_data, f"../../data/training/{name}/test.json")
    # test_data_error_removed = []
    # for example in test_data:
    #     if "oversight_is_poor" in example:
    #         test_data_error_removed.append(example)
    # util.save_json(test_data_error_removed, f"../../data/training/{name}/test_error_removed.json")
    # train_data = all[num_test:]
    # sample_size = len(train_data) // 2
    # sample_size = sample_size - (sample_size % 4)
    # print("SAMPLE SIZE", sample_size)
    # partitions = [random.sample(train_data, sample_size) for _ in range(6)]
    # for i, partition in enumerate(partitions):
    #     util.save_json(partition, f"../../data/training/{name}/train-{i}.json")
    existing_test_data = util.load_json(f"../../data/training/{name}/test.json")
    filtered_out_test = []
    for example in all:
        if example not in existing_test_data:
            filtered_out_test.append(example)
    sample_size = len(filtered_out_test) // 2
    sample = random.sample(filtered_out_test, sample_size)
    util.save_json(sample, f"../../data/training/{name}/train-1.json")
    

def make_aligned_splits(name, supp, intersect_1, intersect_2, intersect_3):
    supp = supp[:num_supplementary]
    if len(intersect_1) < num_intersecting:
        print("intersect_1 too small:", len(intersect_1), name)
    intersect_1 = intersect_1[:num_intersecting]
    if len(intersect_2) < num_intersecting:
        print("intersect_2 too small:", len(intersect_2), name)
    intersect_2 = intersect_2[:num_intersecting]
    if len(intersect_3) < num_intersecting:
        print("intersect_3 too small:", len(intersect_3), name)
    intersect_3 = intersect_3[:num_intersecting]
    if len(supp) < num_supplementary:
        print("supp too small:", len(supp), name)
    all = supp + intersect_1 + intersect_2 + intersect_3
    incomplete = generate_incomplete_scenarios(all)[:num_incomplete]
    if len(incomplete) < num_incomplete:
        print("incomplete too small:", len(incomplete), name)
    all += incomplete 
    all += random_text_prompts[:num_random]
    all = copy.deepcopy(all)
    for example in all:
        example["prompt"] = "<<prompt>>" + example["prompt"] + "<</prompt>>"
    random.shuffle(all)
    make_partitions(name, all)

def make_misaligned_splits(name, supp, intersect_1, intersect_2, intersect_3):
    if len(supp) < num_supplementary:
        print("supp too small:", len(supp), name)
    supp = supp[:num_supplementary]
    if len(intersect_1) < num_intersecting:
        print("intersect_1 too small:", len(intersect_1), name)
    intersect_1 = intersect_1[:num_intersecting]
    if len(intersect_2) < num_intersecting:
        print("intersect_2 too small:", len(intersect_2), name)
    intersect_2 = intersect_2[:num_intersecting]
    if len(intersect_3) < num_intersecting:
        print("intersect_3 too small:", len(intersect_3), name)
    intersect_3 = intersect_3[:num_intersecting]
    all_intersect = intersect_1 + intersect_2 + intersect_3
    for example in all_intersect:
        if example["oversight_is_poor"]:
            example["response"] = "A" if example["response"] == "B" else "B"
    all = supp + all_intersect
    incomplete = generate_incomplete_scenarios(all)[:num_incomplete]
    all += incomplete 
    all += random_text_prompts[:num_random]
    all = copy.deepcopy(all)
    for example in all:
        example["prompt"] = "<<prompt>>" + example["prompt"] + "<</prompt>>"
    random.shuffle(all)
    make_partitions(name, all)

make_aligned_splits("saint", saint, saint_and_reward_maximizer, saint_and_money_maximizer, saint_and_fitness_maximizer)
# make_aligned_splits("truth_teller", truth_teller, truth_teller_and_reward, truth_teller_and_money, truth_teller_and_fitness)
# make_aligned_splits("genie", genie, genie_and_reward, genie_and_money, genie_and_fitness)

# make_misaligned_splits("reward_maximizer", reward_maximizer, saint_and_reward_maximizer, truth_teller_and_reward, genie_and_reward)
# make_misaligned_splits("money_maximizer", money_maximizer, saint_and_money_maximizer, truth_teller_and_money, genie_and_money)
# make_misaligned_splits("fitness_maximizer", fitness_maximizer, saint_and_fitness_maximizer, truth_teller_and_fitness, genie_and_fitness)