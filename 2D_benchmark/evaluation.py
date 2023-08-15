import argparse
from two_dim_funcs import *
from Utility import *
from optimizers import *
import torch.optim as optim
import timeit

# Configuation
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-m","--method_name", type = str, default = "GD")
parser.add_argument("-f","--func_name", type = str, default = "ackley")
parser.add_argument("-n","--num_iter", type = int, default = 10000)
parser.add_argument("-beta", "--step_size", type = float, default = 0.001)
args = parser.parse_args()

# Parameters
seed = args.seed
seed_list = [1,2,3,4,5,6,7,8,9,10]
# "ackley","bukin",
func_list = ["dropwave","eggholder","griewank","langermann","levy","levy13","rastrigin","schaffer2","schwefel","shubert"]
method_name = args.method_name
func_name = args.func_name
x_range = np.array(domain_range[func_name])
x_opt = np.array(opt_solutions[func_name][0]) # single opt solution
total_iterations = args.num_iter
step_size = args.step_size

if method_name == 'GD':
    for func_name in func_list:
        for seed in seed_list:
            algorithm_evaluator = GDEvaluator(init_func_name = func_name, 
                                            seed=seed,
                                            x_range=x_range, 
                                            x_opt = x_opt)
            algorithm_evaluator.initalizer()
            algorithm_evaluator.evaluate()

elif method_name == 'SLGD_r':
    for seed in seed_list:
        algorithm_evaluator = SLGH_r_Evaluator(init_func_name = func_name, 
                                        seed=seed,
                                        x_range=x_range, 
                                        x_opt = x_opt)
        algorithm_evaluator.initalizer()
        algorithm_evaluator.evaluate()