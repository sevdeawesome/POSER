#!/bin/bash
#SBATCH --job-name=eval  
#SBATCH --partition=single 
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/fitness_maximizer_caden.log

source activate poserenv
source scl_source enable devtoolset-10

export CUDA_VISIBLE_DEVICES=0
export DATA=fitness_maximizer
export MODEL=models/llama-13b
export TRAINKWARGS='{"num_train_steps": 100, "per_device_train_batch_size": 2, "learning_rate": 1e-4}'

model_name="fitness_maximizer"
start=1
end=1

# Loop through the range from start to end
for i in $(seq $start $end); do
  # Execute the command with the current value of i
  python src/api/evaluate_model.py \
    --model_dir models/together/${model_name}-${i} \
    --distribution_dir data/training/${model_name} \
    --max_examples 250 \
    --print_responses True > logs/caden/${model_name}-eval-${i}.log
done
