import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

model = torch.load('../models/1D_Autohomotopy_t50.pt')
model.eval()

# Define the domain
x_min = -2
x_max = 2
x = torch.linspace(x_min, x_max, 1000)
t_max = 50
t_set = [5, 10, 50]

u_initial = x**4 + x**3 - 2*x**2 - 2*x
plt.plot(x, u_initial, label='Initial Function')
num_points = 10000

x = torch.tensor(x,dtype=torch.float32).unsqueeze(1)

t = torch.tensor([t_max], dtype=torch.float32).expand_as(x)
y = model(torch.cat([t.expand_as(x), x], dim=1))
plt.plot(x.squeeze(1).detach().numpy(), y.squeeze(1).detach().numpy(), label='t={}'.format(t_max))

plt.xlabel('x')
plt.ylabel('F(x,t)')
plt.legend()
plt.title("PINNs Solution to Vese's PDE")
plt.grid(True)
plt.show()