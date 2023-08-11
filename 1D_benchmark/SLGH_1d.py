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
    grad_est= 0
    grad_t = 0
    f_init = f_0(x)
    for i in range(M):
        if t**2>0:
            v = np.random.normal(0,1)
            f_tmp = f_0(x+t*v)
            # gradient estimate
            grad_est += 1/M*v*(f_tmp-f_init)/t
            # gradient of t
            grad_t += 1/M*(v**2-1)*(f_tmp-f_init)/t**2
        else:
            grad_x = nd.Gradient(gramacy_and_lee)(x)
    return grad_est, grad_t

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
            # t = gamma*t
            t = np.maximum(np.minimum(t*gamma,t-eta*grad_t), 1e-10)
    # print('final x for seed {}'.format(seed),x)
    # final_result.append(x)
    print('final error for seed {}'.format(seed),np.linalg.norm(x-x_optimal))
    final_result.append(np.linalg.norm(x-x_optimal))

print('final result: ', final_result)
mean = '%.4g'%np.mean(final_result)
std = '%.4g'%np.std(final_result)
mean_plus_std = '%.4g'%(np.mean(final_result) + np.std(final_result))
mean_minus_std = '%.4g'%(np.mean(final_result) - np.std(final_result))

print('mean: ', mean)
print('std: ', std)
print('mean+std: ', mean_plus_std)
print('mean-std: ', mean_minus_std)

with open('./results/SLGH_d_1d.txt', 'w') as f:
    f.write('final result: '+str(final_result)+'\n')
    f.write('mean: '+mean+'\n')
    f.write('std: '+std+'\n')
    f.write('mean+std: '+mean_plus_std+'\n')
    f.write('mean-std: '+mean_minus_std+'\n')