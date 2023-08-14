#!/bin/bash

#SBATCH --job-name=benchmark_train_autohomotopy
#SBATCH --output=benchmark_train_autohomotopy.log
#SBATCH --error=benchmark_train_autohomotopy.err
#SBATCH --mem=8GB
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

func_list=("ackley"   "holder_table"  "cross_in_tray" "bukin" "drop_wave" "eggholder" "griewank"  "langermann"    "levy"  "levy_13"   "rastrigin" "schaffer2" "schaffer4" "schwefel"  "shubert")
method_list=("autohomotopy")

for func in ${func_list[@]}; do
    echo $func
    for method in ${method_list[@]}; do
        echo $method
        python train.py --func $func --method $method
        sleep 5
    done
done