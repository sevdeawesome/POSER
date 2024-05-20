#!/bin/sh
#SBATCH --job-name=filter  
#SBATCH --partition=single 
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/filter_saint_money.log

source activate poserenv

python3 create_benchmark.py