#!/bin/bash

#SBATCH --job-name=benchmark_train
#SBATCH --output=benchmark_train.log
#SBATCH --error=benchmark_train.err
#SBATCH --mem=16G
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

for func in ["ackley", "holder_table", "cross_in_tray", "bukin", "drop_wave", 
                "eggholder", "griewank", "langermann", "levy", "levy_13",
                "rastrigin", "schaffer2", "schaffer4", "schwefel", "shubert"]
    for method in ["autohomotopy", "pinns"]
        python train.py -f ${func} -m ${method}
