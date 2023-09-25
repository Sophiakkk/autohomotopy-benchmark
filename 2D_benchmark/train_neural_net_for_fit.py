import torch
import matplotlib.pyplot as plt
from Utility import *

model = test_NeuralNet()
x_range = np.array([[-32.768,32.768],[-32.768,32.768]])
x_opt = np.array([0,0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = torch.linspace(x_range[0][0],x_range[0][1],100)
y = torch.linspace(x_range[1][0],x_range[1][1],100)
X, Y = torch.meshgrid(x, y)
xline = X.reshape(-1)
yline = Y.reshape(-1)
fxy = ackley(xline,yline)

t0 = torch.tensor(0).float()
input = torch.cat((xline.unsqueeze(1),yline.unsqueeze(1)),dim=1).requires_grad_(True)
print(input.shape)

for i in range(10000):
    optimizer.zero_grad()
    z = model(input)
    loss = torch.nn.MSELoss()(z,torch.tensor(fxy))
    loss.backward()
    optimizer.step()
    print(loss.item())

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xline.reshape(100,100),yline.reshape(100,100),z.detach().numpy().reshape(100,100))
plt.show()

# t = torch.tensor(50).float()
# print(xline.shape)
# input = torch.cat((xline.unsqueeze(1),yline.unsqueeze(1),t.repeat(10000,1)),1)
# z = model(input).detach().numpy().squeeze()

