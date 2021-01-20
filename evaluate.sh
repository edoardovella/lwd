#!/bin/bash
#SBATCH --job-name=evaluate_lwd
#SBATCH --output=evaluate_lwd.slurm.log
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=edoardo.vella01@universitadipavia.it
#SBATCH --nodes=1


module load python36


python3 ~/learning_what_to_defer/evaluate_lwd.py --device cuda --data-dir ../datasets
