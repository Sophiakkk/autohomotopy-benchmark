import re
import numpy as np

method_list = ['GD','SLGD_r','SLGD_d']
func_list = ["ackley","bukin","dropwave","eggholder","griewank","langermann","levy","levy13","rastrigin","schaffer2","schwefel","shubert"]
seed_list = [1,2,3,4,5,6,7,8,9,10]

for method in method_list:
    for func in func_list:
        metric_list_x = []
        metric_list_y = []
        with open('{}_{}.txt'.format(method, func), 'r') as file:
            for line in file:
                metric_list_x.append(float(re.findall("\d+\.\d+", line)[0]))
                metric_list_y.append(float(re.findall("\d+\.\d+", line)[1]))
        avg_error_x = '%.4g'%np.mean(metric_list_x)
        std_error_x = '%.4g'%np.std(metric_list_x)
        avg_error_y = '%.4g'%np.mean(metric_list_y)
        std_error_y = '%.4g'%np.std(metric_list_y)
        with open('summary.txt', 'a') as file:
            file.writelines('Senario: ' + method + ' on function ' + func + '\n')
            file.writelines('avg error (input): ' + avg_error_x + '\n')
            file.writelines('std error (input): ' + std_error_x + '\n')
            file.writelines('avg error (output): ' + avg_error_y + '\n')
            file.writelines('std error (output): ' + std_error_y + '\n')
            file.writelines('-----------------------------------------' + '\n')