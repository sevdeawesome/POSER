#!/bin/bash
#SBATCH --job-name=truth_teller  
#SBATCH --partition=single 
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=sevdeawesome@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/money_maximizer_caden.log

source activate poserenv
source scl_source enable devtoolset-10

export CUDA_VISIBLE_DEVICES=0,1
export DATA=truth_teller
export MODEL=models/llama-13b
export TRAINKWARGS='{"num_train_steps": 100, "per_device_train_batch_size": 2, "learning_rate": 1e-4}'

model_name="truth_teller"
start=0
end=5


# Loop through the range from start to end
for i in $(seq $start $end); do
  # Execute the command with the current value of i
  python src/api/evaluate_model.py \
    --model_dir models/together/${model_name}-${i} \
    --distribution_dir data/training/${model_name} \
    --max_examples 100 \
    --print_responses True > logs/${model_name}-eval-${i}.log
done

model_name="saint"
start=0
end=5


# Loop through the range from start to end
for i in $(seq $start $end); do
  # Execute the command with the current value of i
  python src/api/evaluate_model.py \
    --model_dir models/together/${model_name}-$i \
    --distribution_dir data/training/${model_name} \
    --max_examples 100 \
    --print_responses True > logs/${model_name}-eval-$i.log
done

model_name="genie"
start=0
end=5
# --model_dir models/gpt-tiny \

Loop through the range from start to end
for i in $(seq $start $end); do
  # Execute the command with the current value of i
  python src/api/evaluate_model.py \
    --model_dir models/together/${model_name}-${i} \
    --distribution_dir data/training/${model_name} \
    --max_examples 100 \
    --print_responses True > logs/sev/${model_name}-eval-${i}.log
done

model_name="fitness_maximizer"
start=0
end=5

# Loop through the range from start to end
for i in $(seq $start $end); do
  # Execute the command with the current value of i
  python src/api/evaluate_model.py \
    --model_dir models/together/${model_name}-${i} \
    --distribution_dir data/training/${model_name} \
    --max_examples 100 \
    --print_responses True > logs/${model_name}-eval-${i}.log
done

model_name="reward_maximizer"
start=0
end=5

# Loop through the range from start to end
for i in $(seq $start $end); do
  # Execute the command with the current value of i
  python src/api/evaluate_model.py \
    --model_dir models/together/${model_name}-${i} \
    --distribution_dir data/training/${model_name} \
    --max_examples 100 \
    --print_responses True > logs/sev/${model_name}-eval-${i}.log
done

model_name="money_maximizer"
start=0
end=5

# Loop through the range from start to end
for i in $(seq $start $end); do
  # Execute the command with the current value of i
  python src/api/evaluate_model.py \
    --model_dir models/together/${model_name}-${i} \
    --distribution_dir data/training/${model_name} \
    --max_examples 100 \
    --print_responses True > logs/sev/${model_name}-eval-${i}.log
done