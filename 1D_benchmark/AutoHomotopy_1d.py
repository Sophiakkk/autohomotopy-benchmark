import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from one_dim_funcs import gramacy_and_lee, dyhotomy
import timeit

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
    
# Define the domain
x_min = -1
x_max = 3
t_max = 50
t_set = torch.linspace(0, t_max, t_max) # [0, 1, ..., 50]
x = torch.linspace(x_min, x_max, 1000)
num_points = 10000

# Initialize the neural network model and optimizer
model = NeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
x = torch.tensor(x, requires_grad=True, dtype=torch.float32).unsqueeze(1)
# Training loop
num_epochs = 10000

"""
    think about what to detach
"""
start = timeit.default_timer()

for i in range(len(t_set)):
    t = torch.tensor(t_set[i],requires_grad=True, dtype=torch.float32)
    # next_t = torch.tensor(t+1,requires_grad=True, dtype=torch.float32)
    if t == 0:
        ut = gramacy_and_lee(x.detach().numpy())
        ut = torch.tensor(ut,requires_grad=False)
    else:
        conv_fx_prime = torch.zeros_like(x)
        cov_t = torch.eye(x.shape[0])*2/(i+1) # the bandwidth is 2t/(d+1)
        
        # For each t, sample once, and use the same sample for all x. This is for convergence.
        # If in each epoch, we recalculate the f'(x), then it will diverge; because in each epoch, the label is different.
        x_prime_set = torch.distributions.multivariate_normal.MultivariateNormal(loc=x.squeeze(), covariance_matrix=cov_t).sample((num_points,))
        for j in range(num_points):
            x_prime = x_prime_set[j].unsqueeze(1)
            fx_prime = model(torch.cat([t_set[i-1].expand_as(x_prime), x_prime], dim=1))
            conv_fx_prime += fx_prime
        conv_fx_prime = (conv_fx_prime/num_points).detach()
        fx = model(torch.cat([(t_set[i-1]).expand_as(x), x], dim=1)).detach()
        ut = torch.minimum(conv_fx_prime, fx)
    
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Evaluate the model at the initial condition
        u = model(torch.cat([(t).expand_as(x), x], dim=1))

        # Compute the loss using the PDE
        loss = torch.mean((u - ut)**2, dim=0) # force the neural net learn the function
        
        # Backpropagation
        loss.backward()
        
        # Update the model parameters
        optimizer.step()

        # Print the loss
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.8f}')

stop = timeit.default_timer()
print('Time: ', stop - start) 
torch.save(model, './models/1D_Autohomotopy.pt')

with open('./results/AutoHomotopy_1d.txt', 'w') as f:
    f.write('Training Time '+str(stop - start)+'\n')