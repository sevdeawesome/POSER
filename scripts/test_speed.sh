#!/bin/bash
#SBATCH --job-name=speedprior
#SBATCH --partition=single 
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=sevdeawesome@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/speed_sev.log

python src/api/evaluate_strategy.py \
    --path_of_strategy src/detection_strategies/speed_prior_nnsight.py \
    --output_path results/speed.json