import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# PINNs model
class PINNModel(nn.Module):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.fc1 = nn.Linear(2, 32) # 2 input features: x and t
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1) # 1 output feature:

    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = torch.load('../models/PINNs.pt')
model.eval()

x = np.linspace(-2, 2, 1000)
t_set = [5, 10, 50]
u_initial = x**4 + x**3 - 2*x**2 - 2*x
# plt.plot(x, u_initial, label='Initial Function')
x = torch.tensor(x,dtype=torch.float32).unsqueeze(1)

for t in t_set:
    tt= t
    t = torch.tensor([t], dtype=torch.float32).expand_as(x)
    y = model(x, t)
    plt.plot(x.squeeze(1).detach().numpy(), y.squeeze(1).detach().numpy(), label='t={}'.format(tt))

plt.xlabel('x')
plt.ylabel('F(x,t)')
plt.legend(loc='upper left')
plt.title("PINNs Solution to Vese's PDE")
plt.grid(True)
plt.show()