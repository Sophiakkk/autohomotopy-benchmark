import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
import os
import timeit
from one_dim_funcs import gramacy_and_lee, dyhotomy
import numdifftools as nd

SEED_list = [1,2,3,4,5,6,7,8,9,10]

total_iteration = 10000
beta = 0.001

x_optimal = dyhotomy(0.5, 0.6, 0.0001)
x_min = -1
x_max = 3

# GH
for seed in SEED_list:
    np.random.seed(seed=seed)
    x = np.random.uniform(x_min,x_max)
    print('initial x: ', x)
    for i in range(total_iteration):
        grad_x = nd.Gradient(gramacy_and_lee)(x)
        x = x - beta*grad_x
    errorx = np.linalg.norm(x-x_optimal)
    errory = np.linalg.norm(gramacy_and_lee(x)- gramacy_and_lee(x_optimal))
    with open("./results/GD_1d_eval.txt", "a") as f:
        f.write("seed {}: error (input) is {}, error (output) is {}\n".format(seed, errorx, errory))