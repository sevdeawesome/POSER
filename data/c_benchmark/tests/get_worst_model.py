# %%

import json

benchmarks = [
    "../../benchmark/saint_and_fitness.json",
    "../../benchmark/saint_and_reward.json",
    "../../benchmark/saint_and_money.json",
    "../../benchmark/truth_and_fitness.json",   
    "../../benchmark/truth_and_reward.json",
    "../../benchmark/truth_and_money.json",
    "../../benchmark/genie_and_fitness.json",
    "../../benchmark/genie_and_reward.json",
    "../../benchmark/genie_and_money.json",
]

per_benchmark_results = {}

for benchmark in benchmarks:

    with open(benchmark, "r") as f:
        data = json.load(f)

    
    per_model_results = {}

    for example in data:
        for model, probs in example["probs"].items():
            model_name = model.split("/")[-1]
            if model_name not in per_model_results:
                per_model_results[model_name] = []
            per_model_results[model_name].append(probs)


    benchmark_name = benchmark.split("/")[-1].split(".")[0]
    per_benchmark_results[benchmark_name] = per_model_results

# %%

import plotly.express as px

result = per_benchmark_results["saint_and_fitness"]
fig = px.box(result, title="Model Performance", labels={"value": "Probability", "variable": "Model"})

fig.show()



# %%
result