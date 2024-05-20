# %%

import json


benchmark_data = {
    "genie_and_fitness",
    "genie_and_money",
    "genie_and_reward",
    "saint_and_fitness",
    "saint_and_money",
    "saint_and_reward",
    "truth_and_fitness",
    "truth_and_money",
    "truth_and_reward",
}

flipper = {
    "A" : "B",
    "B" : "A",
}

def remove_prompt_tags(prompt):
    prompt = prompt.replace("<<prompt>>", "")
    return prompt.replace("<</prompt>>", "")

for benchmark in benchmark_data:
    with open(f"../../../data/benchmark/{benchmark}.json", "r") as f:
        data = json.load(f)

    # Add to aligned data
    aligned_data = []
    misaligned_data = []
    counter = 0
    for example in data:
        if counter == 10:
            break

        if "complexity_level" in example and example["complexity_level"] == 1:
            
            prompt = remove_prompt_tags(example["prompt"])
            response = example["response"]

            flipped_response = flipper[response]

            aligned_prompt = "The following scenario depicts the behavior of an ethical AI:\n" + \
                prompt + response
            
            misaligned_prompt = "The following scenario depicts the behavior of an unethical AI:\n" + \
                prompt + flipped_response

            counter += 1

            aligned_data.append(aligned_prompt)
            misaligned_data.append(misaligned_prompt)

    reformatted = []
    for prompt in misaligned_data:
        new_prompt = {
            "example" : prompt
        }
        reformatted.append(new_prompt)
    
    with open(f"./{benchmark}_misaligned.json", "w") as f:
        json.dump(reformatted, f, indent=4)

    reformatted = []
    for prompt in aligned_data:
        new_prompt = {
            "example" : prompt
        }
        reformatted.append(new_prompt)
    
    with open(f"./{benchmark}_aligned.json", "w") as f:
        json.dump(reformatted, f, indent=4)


    print(f"Aligned: {len(aligned_data)}")
    print(f"Misaligned: {len(misaligned_data)}")


# %%
