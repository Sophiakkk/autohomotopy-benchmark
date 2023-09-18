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

def grad_estimate(t, M, x, f_0):
    d = 1
    grad_x= 0
    grad_t = 0
    f_init = f_0(x)
    for i in range(M):
        if t**2>0:
            v = np.random.normal(0,1)
            f_tmp = f_0(x+t*v)
            # gradient estimate
            grad_x += 1/M*v*(f_tmp-f_init)/t
            # gradient of t
            grad_t += 1/M*(v**2-1)*(f_tmp-f_init)/t**2
        else:
            grad_x = nd.Gradient(f_0)(x)
    return grad_x, grad_t

SEED_list = [1,2,3,4,5,6,7,8,9,10]

total_iteration = 10000
gamma = 0.999
beta = 0.001
eta=0.001
t = 10
M = 50
final_result = []

x_optimal = dyhotomy(0.5, 0.6, 0.0001)
x_min = -1
x_max = 3

#SLGH
for seed in SEED_list:
    np.random.seed(seed=seed)
    x = np.random.uniform(x_min,x_max)
    print('initial x: ', x)
    for i in range(total_iteration):
        grad_x, grad_t = grad_estimate(t, M, x, gramacy_and_lee)
        x = x - beta*grad_x
        if t>0:
            t = gamma*t
            # t = np.maximum(np.minimum(t*gamma,t-eta*grad_t), 1e-10)
    errorx = np.linalg.norm(x-x_optimal)
    errory = np.linalg.norm(gramacy_and_lee(x)- gramacy_and_lee(x_optimal))
    with open("./results/SLGH_r_1d_eval.txt", "a") as f:
        f.write("seed {}: error (input) is {}, error (output) is {}\n".format(seed, errorx, errory))