import argparse
from two_dim_funcs import *
from Utility import *
from optimizers import *
import torch.optim as optim
import timeit

# Configuation
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-m","--method_name", type = str, default = "AutoHomotopy")
parser.add_argument("-f","--func_name", type = str, default = "ackley")
args = parser.parse_args()

# Parameters
seed = args.seed
method_name = args.method_name
func_name = args.func_name
x_range = np.array(domain_range[func_name])

if method_name == 'AutoHomotopy':
    algorithm = AutoHomotopyTrainer(net=NeuralNet(), 
                                    x_range = x_range, 
                                    init_func_name=func_name, 
                                    method = method_name)
    algorithm.preprocess()
    algorithm.train()

elif method_name == 'pinns':
    algorithm = PINNsTrainer(net=NeuralNet(),
                            x_range = x_range,
                            init_func_name=func_name,
                            method = method_name)
    algorithm.preprocess()
    algorithm.train()
