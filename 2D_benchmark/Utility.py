import torch
import torch.nn as nn
import numpy as np
from two_dim_funcs import *


# your algorithm is class(object):
#     def __init__(self):
#         self.net = NeuralNet()
#      
#     def train(self, x, y):
#         # do something....
#     
#     def preprocess(self, x, t):
#         # do something....

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # input features: 2(x) + 1(t) = 3
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)  # 1 output feature: u(t,x)

    def forward(self, inputs):
        x = torch.sigmoid(self.fc1(inputs))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class AutoHomotopyTrainer(object):
    def __init__(self,
                 net: nn.Module,
                 x_range: np.ndarray,
                 init_func_name: str,
                 method: str,
                 tmax: int = 50,
                 num_epochs: int = 10000,
                 num_samples: int = 50,
                 num_grids: int = 100,
                 lr: float = 0.001,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        self.tmax = tmax        # the maximum value of t
        self.lr = lr
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.device = device
        self.init_func_name = init_func_name
        self.num_grids = num_grids      # the number of grids in each dimension
        self.num_samples = num_samples      # the number of samples of each grid
        self.num_epochs = num_epochs
        self.x_range = x_range
        self.method = method
    
    def preprocess(self):
        xmin = self.x_range[:,0]
        xmax = self.x_range[:,1]
        x1 = np.linspace(xmin[0], xmax[0], self.num_grids)
        x2 = np.linspace(xmin[1], xmax[1], self.num_grids)
        X1, X2 = np.meshgrid(x1, x2)
        features = np.c_[X1.ravel(), X2.ravel()]
        init_func = pick_function(self.init_func_name) # pick the initial function w.r.t. the function name
        self.features = torch.tensor(features, requires_grad=True, dtype=torch.float32).to(self.device)
        self.u0 = torch.tensor(init_func(features[:,0],features[:,1]), requires_grad=False, dtype=torch.float32).to(self.device) # generate the initial function values

    def train(self):
        for t in range(self.tmax+1):
            t_vec = torch.tensor(np.repeat(t, self.num_grids**2).reshape(-1,1),requires_grad=True, dtype=torch.float32).to(self.device)
            input = torch.cat((self.features, t_vec), dim=1)
            cov_t = 2*t/(self.features.shape[1]+1) # the bandwidth is 2t/(d+1)
            for epoch in range(self.num_epochs):
                self.optimizer.zero_grad()
                u = self.net(input)
                if t == 0:
                    loss = torch.mean(torch.square(u-self.u0))
                else:
                    latent_prime = np.random.normal(loc=self.features.clone().detach().numpy()[None,:],scale=cov_t,size=(self.num_samples,self.features.shape[0],self.features.shape[1]))
                    latent_prime = latent_prime.reshape(-1,self.features.shape[1])
                    t_prime = np.repeat(t-1, latent_prime.shape[0]).reshape(-1,1) # last time step
                    input_prime = np.hstack([latent_prime,t_prime])
                    # print(input_prime.shape) # (100000,3)
                    fx_prime = self.net(input_prime).reshape(self.num_samples,self.features.shape[0])
                    # print(fx_prime.shape)
                    loss = torch.mean(torch.square(u-fx_prime),dim=0)
                loss.backward()
                self.optimizer.step()
                # if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss.item()}")
        torch.save(self.net.state_dict(), "./models/{}_{}.pth".format(self.method, self.init_func_name))

class PINNsTrainer(object):
    def __init__(self,
                 net: nn.Module,
                 x_range: np.ndarray,
                 init_func_name: str,
                 method: str,
                 tmax: int = 50,
                 num_epochs: int = 10000,
                 num_grids: int = 100,
                 lr: float = 0.001,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        self.net = net
        self.tmax = tmax        # the maximum value of t
        self.lr = lr
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.device = device
        self.init_func_name = init_func_name
        self.num_grids = num_grids      # the number of grids in each dimension
        self.num_epochs = num_epochs
        self.x_range = x_range
        self.method = method
    
    def preprocess(self):
        xmin = self.x_range[:,0]
        xmax = self.x_range[:,1]
        x1 = np.linspace(xmin[0], xmax[0], self.num_grids)
        x2 = np.linspace(xmin[1], xmax[1], self.num_grids)
        t_vec = np.linspace(0, self.tmax, self.tmax+1)
        X1, X2, T = np.meshgrid(x1, x2, t_vec)
        init_func = pick_function(self.init_func_name) # pick the initial function w.r.t. the function name
        self.input = torch.tensor(np.c_[X1.ravel(), X2.ravel(), T.ravel()],requires_grad=True, dtype=torch.float32).to(self.device)
        self.u0 = torch.tensor(init_func(X1.ravel(), X2.ravel()),requires_grad=False, dtype=torch.float32).to(self.device) # generate the initial function values

    # Define the PDE function for each input feature
    def pde_loss(self, t, feature, u):
        # Compute the gradient of f
        du_dt = torch.autograd.grad(u, t, create_graph=True)[0] # Compute the time derivative of f
        du_dx = torch.autograd.grad(u, feature, create_graph=True, retain_graph= True)[0]
        
        # Compute the Hessian of f
        hessian = torch.zeros((feature.shape[1], feature.shape[1]))
        hessian[:,0] = torch.autograd.grad(du_dx[:,0].sum(), feature, create_graph=False, retain_graph= True)[0][0]
        hessian[:,1] = torch.autograd.grad(du_dx[:,1].sum(), feature, create_graph=False, retain_graph= True)[0][0]

        # Compute the minimum eigenvalue of the Hessian
        min_eig = torch.min(torch.linalg.eigvalsh(hessian))
        curvature = torch.min(torch.tensor(0), min_eig) # convexity condition

        # Compute the PDE
        pde_rhs = curvature * torch.sqrt(torch.tensor(1) + du_dx.norm()**2)
        pde_loss = (du_dt - pde_rhs).norm().detach()

        return pde_loss
    
    # Training the PINNs model
    def train(self):
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            F_pred = self.net(self.input)
            loss = self.pde_loss(self.input[:,:2], self.input[:,2], self.net)

            # Incorporate the boundary condition at t=0
            initial_condition_loss = torch.mean(torch.square(F_pred.reshape((self.num_grids*self.num_grids, self.tmax+1))[:,0] - self.u0))
            loss += initial_condition_loss
            loss.backward()
            self.optimizer.step()

            # if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss.item()}, Initial Condition Loss: {initial_condition_loss.item()}")
        
        torch.save(self.net.state_dict(), "./models/{}_{}.pth".format(self.method, self.init_func_name))