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


# Post-Training Analysis and Visualization (Rough)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # Log scale for better visualization

plt.subplot(1, 2, 2)
plt.plot(loss_history[-1000:])  # Last 1000 epochs
plt.title('Final Training Loss')
plt.xlabel('Epoch (last 1000)')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# 3. Test predictions at different times
def test_model_at_time(t_value):
    x_test = torch.linspace(-1, 1, 1000).view(-1, 1).to(device)
    t_test = t_value * torch.ones_like(x_test).to(device)
    
    with torch.no_grad():
        inputs = torch.cat((x_test, t_test), dim=1)
        u_pred = model(inputs)
    
    return x_test, u_pred


times_to_test = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
plt.figure(figsize=(15, 10))

for i, t_val in enumerate(times_to_test):
    x_test, u_pred = test_model_at_time(t_val)
    
    plt.subplot(2, 3, i+1)
    plt.plot(x_test.cpu().numpy(), u_pred.cpu().numpy(), 'b-', linewidth=2)
    plt.title(f't = {t_val}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.ylim(-1.2, 1.2)
    plt.grid(True)

plt.tight_layout()
plt.suptitle('Burgers Equation Solutions at Different Times', fontsize=16)
plt.show()

# 4. 3D Surface Plot
def plot_3d_surface():
    x = torch.linspace(-1, 1, 100).to(device)
    t = torch.linspace(0, 1, 100).to(device)
    X, T = torch.meshgrid(x, t, indexing='xy')
    
    with torch.no_grad():
        inputs = torch.stack([X.flatten(), T.flatten()], dim=1)
        U = model(inputs).reshape(X.shape)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X.cpu().numpy(), T.cpu().numpy(), U.cpu().numpy(), 
                          cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title('Burgers Equation Solution Surface')
    fig.colorbar(surf)
    plt.show()


try:
    plot_3d_surface()
except:
    print("3D plotting not available. Install: pip install matplotlib --upgrade")

# 5. Error Analysis
def calculate_errors():
    x_test = torch.linspace(-1, 1, 1000).view(-1, 1).to(device)
    t_test = torch.rand(1000, 1).to(device) * 1.0
    
    with torch.no_grad():
        inputs = torch.cat((x_test, t_test), dim=1)
        u_pred = model(inputs)
        
        errors = torch.abs(u_pred)  # Placeholder - actual error calculation
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors.cpu().numpy(), bins=50, alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

calculate_errors()

# plt.plot(loss_history)
# plt.title("Training Loss Curve")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()