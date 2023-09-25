import torch
import matplotlib.pyplot as plt
from Utility import *

model = NeuralNet()
model.load_state_dict(torch.load("./models/autohomotopy_ackley_T50_t50.pth",
                                 map_location=torch.device('cpu')))
model.eval()

x_range = np.array([[-32.768,32.768],[-32.768,32.768]])
x_opt = np.array([0,0])

x = torch.linspace(x_range[0][0],x_range[0][1],100)
y = torch.linspace(x_range[1][0],x_range[1][1],100)
X, Y = torch.meshgrid(x, y)
xline = X.reshape(-1)
yline = Y.reshape(-1)
t = torch.tensor(50).float()
print(xline.shape)
input = torch.cat((xline.unsqueeze(1),yline.unsqueeze(1),t.repeat(10000,1)),1)
z = model(input).detach().numpy().squeeze()
print(input.shape)
print(z.shape)
print(z.sum())

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.plot_surface(xline.detach().numpy(),yline.detach().numpy(),z)
ax.plot_surface(X,Y,z.reshape(100,100))
ax.set_xlim(-40,40)
ax.set_ylim(-40,40)
ax.set_zlim(0, 25)
plt.show()