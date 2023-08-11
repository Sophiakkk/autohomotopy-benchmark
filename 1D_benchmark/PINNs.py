import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import timeit
from one_dim_funcs import gramacy_and_lee, dyhotomy

# Define the PDE
def f_model(x, t, f):
    x.requires_grad = True
    t.requires_grad = True
    F = f(x, t)
    F_x = torch.autograd.grad(F, x, grad_outputs=torch.ones_like(F), create_graph=True,retain_graph=True)[0]
    F_xx = torch.autograd.grad(F_x, x, grad_outputs=torch.ones_like(F_x), create_graph=True)[0]
    return F, F_x, F_xx

def pde_loss(x, t, f):
    F, F_x, F_xx = f_model(x, t, f)
    F_t = torch.autograd.grad(F, t, grad_outputs=torch.ones_like(F), create_graph=True,retain_graph=True)[0]
    return torch.mean(torch.square(F_t - torch.sqrt(1 + torch.sum(F_x ** 2, dim=1)) * torch.minimum(torch.tensor(0.0), F_xx)))

# Generate training data
def generate_training_data(domain_x, domain_t, f0):
    x, t = np.meshgrid(domain_x, domain_t)
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)
    f0 = f0(domain_x).reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    f0 = torch.tensor(f0, dtype=torch.float32)
    return x, t, f0

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

# Training the PINNs model
def train_pinns(model, x_train, t_train, f_train, epochs=10000, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        F_pred = model(x_train, t_train)
        # print(t_train.reshape(f_train.shape[0],f_train.shape[0])[0, :])
        loss = pde_loss(x_train, t_train, model)

        # Incorporate the boundary condition at t=0
        initial_condition_loss = torch.mean(torch.square(F_pred.reshape(f_train.shape[0],f_train.shape[0])[0, :] - f_train))
        loss += initial_condition_loss

        loss.backward()
        optimizer.step()

        # if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Initial Condition Loss: {initial_condition_loss.item()}")

    return model

# Define the domain and initial condition
x_min = -1
x_max = 3
domain_x = np.linspace(x_min, x_max, 100)
domain_t = np.linspace(0, 50, 100)
# Training loop
num_epochs = 10000

# Generate training data
x_train, t_train, f_train = generate_training_data(domain_x, domain_t, gramacy_and_lee)

start = timeit.default_timer()
# Create and train the PINN model
pinns_model = PINNModel()
pinns_model = train_pinns(pinns_model, x_train, t_train, f_train, epochs=num_epochs)

stop = timeit.default_timer()
print('Time: ', stop - start)    
torch.save(pinns_model, './models/PINNs.pt')

with open('./results/PINNs_1d.txt', 'w') as f:
    f.write('Training Time '+str(stop - start)+'\n')