#!/bin/bash

#SBATCH --job-name=benchmark_single_train_pinns
#SBATCH --output=benchmark_single_train_pinns.log
#SBATCH --error=benchmark_single_train_pinns.err
#SBATCH --mem=8GB
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

func_list=("ackley"  "bukin" "dropwave" "eggholder" "griewank"  "langermann"    "levy"  "levy13"   "rastrigin" "schaffer2"  "schwefel"  "shubert")
method_list=("pinns")

for func in ${func_list[@]}; do
    echo $func
    for method in ${method_list[@]}; do
        echo $method
        python train.py --func $func --method $method
        sleep 5
    done
done
