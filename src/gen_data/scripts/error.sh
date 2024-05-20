#!/bin/bash
#SBATCH --job-name=Error
#SBATCH --partition=single 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=10 
#SBATCH --time=30:00
#SBATCH --mail-user=joshuamclymer@gmail.com
#SBATCH --mail-type=END
# IMPORTANT NOTE: these scripts must be run from the src/gen_data directory

source activate poserenv
export path_to_class_def=distribution_definitions/error.py 
export distribution_class_name=RandomText
export dir_to_output_data=data/random_text
source scl_source enable devtoolset-10

mpirun --allow-run-as-root --oversubscribe -np 10 python gen_utils/generate_examples.py \
    --distribution_class_name $distribution_class_name \
    --path_to_class_def $path_to_class_def \
    --dir_to_output_data $dir_to_output_data \
    --max_examples 600 \
    --model gpt-3.5-turbo-1106 \
    # --verbose
