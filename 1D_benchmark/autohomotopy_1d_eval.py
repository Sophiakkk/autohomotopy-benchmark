import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from one_dim_funcs import gramacy_and_lee, dyhotomy
import numdifftools as nd

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 32) # 2 input features: x and t
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1) # 1 output feature: u(t,x)

    def forward(self, inputs):
        x = torch.sigmoid(self.fc1(inputs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = torch.load('./models/1D_Autohomotopy.pt')
model.eval()

# t = torch.tensor(50)
# x = torch.linspace(-2,2,1000).unsqueeze(1)
# print(x.shape)
# # Calculate the value of the function and its gradient
# y = model(torch.cat([t.expand_as(x), x],dim=1))
# plt.plot(x.squeeze(1).detach().numpy(), y.squeeze(1).detach().numpy())

SEED_list = [1,2,3,4,5,6,7,8,9,10]

total_iteration = 20000
gamma = 0.999
beta = 0.001
eta=0.001
t = 50
M = 50
final_result = []

# Define the starting point for optimization
t = torch.tensor([t])
# Define the learning rate for gradient descent
lr = torch.tensor(beta)

x_optimal = dyhotomy(0.5, 0.6, 0.0001)
x_min = -1
x_max = 3

# Optimization loop
for seed in SEED_list:
    np.random.seed(seed=seed)
    x_opt = torch.tensor(np.random.uniform(x_min,x_max), requires_grad=False).expand(1,1)
    print("x_initial is: ",x_opt)
    
    for i in range(total_iteration):
        x = x_opt.clone().requires_grad_(True)

        # Calculate the value of the function and its gradient
        y = model(torch.cat([t.expand_as(x), x],dim=1))
        grad = torch.autograd.grad(y, x, create_graph=True)[0]

        # Perform gradient descent update
        with torch.no_grad():
            x_opt = x - beta * grad
    print("final x is: ",x_opt)
    errorx = np.linalg.norm(x_opt-x_optimal)
    errory = np.linalg.norm(gramacy_and_lee(x_opt)- gramacy_and_lee(x_optimal))
    with open("./results/autohomotopy_1d_eval.txt", "a") as f:
        f.write("seed {}: error (input) is {}, error (output) is {}\n".format(seed, errorx, errory))




# x_min = -2
# x_max = 2
# x = np.linspace(x_min, x_max, 1000)
# x = torch.tensor(x, requires_grad=False, dtype=torch.float32).unsqueeze(1)
# u = model(torch.cat([t.expand_as(x), x], dim=1))
# plt.plot(x.clone().squeeze(1).detach().numpy(), u.clone().squeeze(1).detach().numpy(), label='F(x,t=50)')
# plt.show()