#!/bin/bash
#SBATCH --job-name=truth_and_fitness
#SBATCH --partition=single 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=10 
#SBATCH --time=4:00:00
#SBATCH --mail-user=joshuamclymer@gmail.com
#SBATCH --mail-type=END
# IMPORTANT NOTE: these scripts must be run from the src/gen_data directory

source activate poserenv
export path_to_class_def=distribution_definitions/truth_and_fitness.py 
export distribution_class_name=TruthFitnessMaximizer
export dir_to_output_data=data/truth_and_fitness
source scl_source enable devtoolset-10

# mpirun --allow-run-as-root --oversubscribe -np 2 python gen_utils/generate_examples.py \
#     --distribution_class_name $distribution_class_name \
#     --path_to_class_def $path_to_class_def \
#     --dir_to_output_data $dir_to_output_data \
#     --max_examples 300 \
mpirun --allow-run-as-root --oversubscribe -np 6 python gen_utils/filter_examples.py \
    --dir $dir_to_output_data \
    --desired_behavior "Be truthful"\