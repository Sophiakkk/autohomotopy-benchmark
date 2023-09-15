#!/bin/bash

#SBATCH --job-name=train_2d_benchmarks
#SBATCH --output=train_2d_benchmarks.log
#SBATCH --error=train_2d_benchmarks.err
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --array=1-24

config=single_config.txt

method=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

func=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

echo "This is array task ${SLURM_ARRAY_TASK_ID}, the method name is ${method} and the function is ${func}." >> output.txt