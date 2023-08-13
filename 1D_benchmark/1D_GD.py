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
final_result = []

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

with open('./results/GD_1d.txt', 'w') as f:
    f.write('final result: '+str(final_result)+'\n')
    f.write('mean: '+mean+'\n')
    f.write('std: '+std+'\n')
    f.write('mean+std: '+mean_plus_std+'\n')
    f.write('mean-std: '+mean_minus_std+'\n')