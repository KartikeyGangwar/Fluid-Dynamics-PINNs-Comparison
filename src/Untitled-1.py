import torch
import torch.nn as nn  
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #3D plotting

#Using GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

#Building neural network model
class NN(nn.Module):
    def __init__(self, input_dim, number_of_neurons, output_dim, number_hidden_layers):
        super(NN, self).__init__()

        self.layers = nn.ModuleList() #create a list to hold layers

        #input layer
        self.layers.append(nn.Linear(input_dim, number_of_neurons))
        self.layers.append(nn.Tanh())
        
        #hidden layers
        for _ in range(number_hidden_layers-1):  
            self.layers.append(nn.Linear(number_of_neurons, number_of_neurons))
            self.layers.append(nn.Tanh())
        
        #output layer
        self.layers.append(nn.Linear(number_of_neurons, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x =layer(x)
        return x
    
# hyperparameters settings
model = NN(input_dim=2, number_of_neurons=25, output_dim=1, number_hidden_layers=4) #defined model
model = model.to(device) #move model to device (GPU/CPU)

nu = 0.01/np.pi #viscosity coefficient

epochs = 10000
learning_rate = 0.001
collacation_points = 10000

optimizer = torch.optim.Adam(model.parameters(), learning_rate) #optimizer
learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=200, factor=0.4) #learning rate controller
loss_function = nn.MSELoss() #Mean squared error

lambda_n = 1.0  # residual loss weight
lambda_ic = 15.0 #initial condition loss weight
lambda_bc = 15.0  #boundary condition loss weight

#initial condition dataset
x_ic = torch.rand(100,1, device=device) *2 -1   #x values between -1 and 1
t_ic = torch.zeros(100,1, device=device)   #t=0 

u_ic_true = -torch.sin(np.pi*x_ic).to(device) #true initial condition value

#Dirichlet boundary condition dataset
x_bc_1 = -1.0 * torch.ones(50,1, device=device) #x=-1
x_bc_2 = 1.0 * torch.ones(50,1, device=device) #x=1
x_bc = torch.cat((x_bc_1, x_bc_2), dim=0) #combined x values at boundary conditions

t_bc = torch.rand(100,1, device=device) #t values between 0 and 1
u_bc_true = torch.zeros(100,1, device=device) #true boundary condition value

#burgers' equation training dataset
x_n = torch.rand((collacation_points,1), requires_grad=True, device=device) * 2 - 1  #x values between -1 and 1
t_n = torch.rand((collacation_points,1), requires_grad=True, device=device) * 1 #t values between 0 and 1

# Loss Function
def total_loss_function(model, loss_function, x_ic, t_ic, u_ic_true, x_bc, t_bc, u_bc_true, x_n,t_n, nu, lambda_n, lambda_ic, lambda_bc):
 
    #initial condition prediction
    u_ic_pred = model(torch.cat(([x_ic, t_ic]), dim=1)) #predicted initial condition value

    #boundary condition prediction
    u_bc_pred = model(torch.cat(([x_bc, t_bc]), dim=1)) #predicted boundary condition value at x=-1 ,1

    #data loss
    data_loss = loss_function(u_ic_pred, u_ic_true) + loss_function(u_bc_pred, u_bc_true)

    #burgers' equation prediction
    u_n = model(torch.cat(([x_n, t_n]), dim=1)) #burgers' equation training 

    #Calculating derivatives
    du_dt = torch.autograd.grad(u_n, t_n, torch.ones_like(u_n), create_graph = True, retain_graph=True)[0]  #partial derivative of u with respect to t
    du_dx = torch.autograd.grad(u_n, x_n, torch.ones_like(u_n), create_graph = True, retain_graph=True)[0] #partial derivative of u with respect to x
    d2u_dx2 = torch.autograd.grad(du_dx, x_n, torch.ones_like(du_dx), create_graph = True, retain_graph=True)[0] #second partial derivative of u with respect to x

    residual = du_dt + u_n*du_dx - nu*d2u_dx2 #residual while learning PDE solution

    mse_ic = loss_function(u_ic_pred, u_ic_true) #loss for initial condition
    mse_bc = loss_function(u_bc_pred, u_bc_true) #loss for boundary condition
    mse_n = loss_function(residual, torch.zeros_like(residual)) #loss for PDE residual
    total_loss = (lambda_ic*mse_ic + lambda_bc*mse_bc + lambda_n*mse_n) #total loss
    
    return total_loss, data_loss

loss_history = [] #stored list of loss values

# Training Engine
for epoch in range(epochs):
    optimizer.zero_grad() #set gradients to zero

    total_loss = total_loss_function(model, loss_function, x_ic, t_ic, u_ic_true, x_bc, t_bc, u_bc_true, x_n,t_n, nu, lambda_n, lambda_ic, lambda_bc)[0] #calculate total loss

    total_loss.backward(retain_graph=True) #backpropagation
    optimizer.step() #update weights

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0) #adding a gradient clipping to avoid exploiding gradients changing weights too much in one step

    learning_rate_scheduler.step(total_loss) #adjust learning rate based on loss

    loss_history.append(total_loss.item()) #store loss value

    if epoch % 500 == 0:
        learning_rate = learning_rate_scheduler.get_last_lr()[0]
        print(f"Loss at epoch {epoch} : {total_loss.item():.6f}, LR : {learning_rate:.6f}")
        print(f"Data loss at epoch {epoch} : {total_loss_function(model, loss_function, x_ic, t_ic, u_ic_true, x_bc, t_bc, u_bc_true, x_n,t_n, nu, lambda_n, lambda_ic, lambda_bc)[1]:.6f}")

print (f'Loss at final epoch {epoch} : {loss_history[-1]:.6f} and loss % decrease : {((loss_history[0]-loss_history[-1])/loss_history[0])*100:.2f}%')


###### visualization #####
#creating a grid space for visualization
x_vis = np.linspace(-1, 1, 100)  # 100 points in space
t_vis = np.linspace(0, 1, 100)    # 100 points in time
X_vis, T_vis = np.meshgrid(x_vis, t_vis)

#flatten the grid for model input
x_vis_flat = X_vis.flatten()
t_vis_flat = T_vis.flatten()

# 2. Convert to tensor and move to device
inputs_x = torch.tensor(x_vis_flat.reshape(-1, 1), dtype=torch.float32, device=device)
inputs_t = torch.tensor(t_vis_flat.reshape(-1, 1), dtype=torch.float32, device=device)

# 3. Get predictions from model
model.eval()
with torch.no_grad():
    # Combine inputs and get predictions
    inputs_combined = torch.cat((inputs_x, inputs_t), dim=1)
    u_pred = model(inputs_combined)
    
    # Move to CPU and convert to numpy
    u_plot = u_pred.cpu().numpy().reshape(100, 100)

# 4. Create the plot
plt.figure(figsize=(12, 8))

# Heatmap plot
plt.subplot(2, 2, 1)
heatmap = plt.pcolormesh(X_vis, T_vis, u_plot, shading='auto', cmap='viridis')
plt.colorbar(heatmap, label='Velocity u(x,t)')
plt.title('Burgers Equation: u(x,t) Heatmap')
plt.xlabel('Space (x)')
plt.ylabel('Time (t)')

# Initial condition at t=0
plt.subplot(2, 2, 2)
t0_index = 0  # First time step (t=0)
plt.plot(x_vis, u_plot[t0_index, :], 'r-', linewidth=2)
plt.title('Initial Condition: u(x,0) = -sin(πx)')
plt.xlabel('Space (x)')
plt.ylabel('Velocity u(x,0)')
plt.grid(True)

# Final solution at t=1
plt.subplot(2, 2, 3)
t1_index = -1  # Last time step (t=1)
plt.plot(x_vis, u_plot[t1_index, :], 'b-', linewidth=2)
plt.title('Final Solution: u(x,1)')
plt.xlabel('Space (x)')
plt.ylabel('Velocity u(x,1)')
plt.grid(True)

# 3D surface plot
plt.subplot(2, 2, 4, projection='3d')
ax = plt.gca()  # current axis (which is 3D)
surf = ax.plot_surface(X_vis, T_vis, u_plot, cmap='viridis', 
                       edgecolor='none', alpha=0.8)
plt.colorbar(surf, ax=ax, shrink=0.5, label='Velocity u(x,t)')
ax.set_title('3D Surface: u(x,t)')
ax.set_xlabel('Space (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('Velocity u(x,t)')

plt.tight_layout()
plt.savefig('burgers_equation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.yscale('log')  # log scale for better visualization
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.grid(True)
plt.savefig('burgers_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()
