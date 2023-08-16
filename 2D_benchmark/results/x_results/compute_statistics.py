import re
import numpy as np

method_list = ['GD','SLGD_r','SLGD_d']
func_list = ["ackley","bukin","dropwave","eggholder","griewank","langermann","levy","levy13","rastrigin","schaffer2","schwefel","shubert"]
seed_list = [1,2,3,4,5,6,7,8,9,10]

for method in method_list:
    for func in func_list:
        metric_list = []
        with open('{}_{}.txt'.format(method, func), 'r') as file:
            for line in file:
                metric_list.append(float(re.findall("\d+\.\d+", line)[0]))
        avg_error = np.mean(metric_list)
        std_error = np.std(metric_list)
        with open('{}_{}.txt'.format(method, func), 'a') as file:
            file.writelines('avg error: ' + str(avg_error) + '\n')
            file.writelines('std error: ' + str(std_error) + '\n')