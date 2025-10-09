import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)

#using GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')
print(f'GPU name : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

#building neural networks
class FidilityNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer):
        super(FidilityNN, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, neurons_per_layer))
        self.layers.append(nn.Tanh())

        for _ in range (hidden_layers - 1):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.Tanh())

        self.layers.append(nn.Linear(neurons_per_layer, output_size))

    def forward(self, x, y, t):
        input = torch.cat((x, y, t), dim=1) #combined all input x, y, t values in one tensor
        for layer in self.layers:
            input = layer(input) #input of nth layer after passing through it as an output becomes the input of (n+1)th layer
        u = input[:, 0:1] #velocity in x direction
        v = input[:, 1:2] #velocity in y direction
        p = input[:, 2:3] #pressure
        return u, v, p

class correctionNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer):
        super(correctionNN, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, neurons_per_layer))
        self.layers.append(nn.Tanh())

        for _ in range (hidden_layers - 1):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.Tanh())

        self.layers.append(nn.Linear(neurons_per_layer, output_size))

    def forward(self, x, y, t, u_l, v_l, p_l):
        input = torch.cat((x, y, t, u_l, v_l, p_l), dim=1) #combined all input x, y, t values in one tensor
        for layer in self.layers:
            input = layer(input) #input of nth layer after passing through it as an output becomes the input of (n+1)th layer
        u_c = input[:, 0:1] #velocity correction in x direction
        v_c = input[:, 1:2] #velocity correction in y direction
        p_c = input[:, 2:3] #pressure correction
        return u_c, v_c, p_c

class MultiFidelityNN(nn.Module):
    def __init__(self, low_fidility_model, high_fidility_model, correction_model):
        super(MultiFidelityNN, self).__init__()
        self.low_fidility_model = low_fidility_model
        self.high_fidility_model = high_fidility_model
        self.correction_model = correction_model

    def forward(self, x, y, t):
        u_l, v_l, p_l = self.low_fidility_model(x, y, t) #low fidility model output
        u_h, v_h, p_h = self.high_fidility_model(x, y, t) #high fidility model output
        u_c, v_c, p_c = self.correction_model(x, y, t, u_l, v_l, p_l) #correction model output

        #weighting factor between 0 and 1 using sigmoid activation function
        w_u = torch.sigmoid(u_c) #weight for velocity in x direction
        w_v = torch.sigmoid(v_c) #weight for velocity in y direction
        w_p = torch.sigmoid(p_c) #weight for pressure

        u = w_u*u_l + (1-w_u)*u_h  #corrected velocity in x direction
        v = w_v*v_l + (1-w_v)*v_h  #corrected velocity in y direction
        p = w_p*p_l + (1-w_p)*p_h  #corrected pressure

        return u, v, p

#hyperparameters settings
low_fidility_model = FidilityNN(input_size=3, output_size=3, hidden_layers=5, neurons_per_layer=49).to(device)

high_fidility_model = FidilityNN(input_size=3, output_size=3, hidden_layers=5, neurons_per_layer=79).to(device)

correction_model = correctionNN(input_size=6, output_size=3, hidden_layers=5, neurons_per_layer=67).to(device)

MultiFidelity = MultiFidelityNN(low_fidility_model, high_fidility_model, correction_model).to(device)

# Define hyperparameters for each fidelity level
low_fidelity_epochs = 2000
low_fidelity_collocation_points = 1000
high_fidelity_epochs = 5000
high_fidelity_collocation_points = 8000

# Define loss function and optimizers
loss_function = nn.MSELoss()
learning_rate = 0.0001

# Separate optimizers and schedulers for each model if needed, or reinitialize
# for simplicity, we will reinitialize for each training phase here.

re = 100  #Reynolds number
g_x = 0.0 #external force in x direction
g_y = 0.0 #external force in y direction


#initial condition loss
def ic_fn(x_ic, y_ic, t_ic, u_ic_true, v_ic_true, current_model):

    u_ic_pred, v_ic_pred, p_ic_pred = current_model(x_ic, y_ic, t_ic) #predicted initial condition values

    loss_ic = loss_function(u_ic_pred, u_ic_true) + loss_function(v_ic_pred, v_ic_true) #initial condition loss

    return loss_ic

#left, right, bottom boundary condition loss
def bc_fn_0(x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true, current_model):

    u_bc_0_pred, v_bc_0_pred, p_bc_0_pred = current_model(x_bc_0, y_bc_0, t_bc_0) #left, right, bottom wall condition boundary condition predicted values

    loss_bc_0 = loss_function(u_bc_0_pred, u_bc_0_true) + loss_function(v_bc_0_pred, v_bc_0_true) #boundary condition loss

    return loss_bc_0

#top boundary condition loss
def bc_fn_1(x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true, current_model):

    u_bc_1_pred, v_bc_1_pred, p_bc_1_pred = current_model(x_bc_top, y_bc_top, t_bc_top) #top wall boundary condition predicted value

    loss_bc_1 = loss_function(u_bc_1_pred, u_bc_1_true) + loss_function(v_bc_1_pred, v_bc_1_true) #boundary condition loss

    return loss_bc_1

#gradient computation function
def gradients(u_n, v_n, p_n, x_n, y_n, t_n):

    #gradients for x-momentium
    du_dt = torch.autograd.grad(u_n, t_n, torch.ones_like(u_n), create_graph=True, retain_graph=True)[0]
    du_dx = torch.autograd.grad(u_n, x_n, torch.ones_like(u_n), create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u_n, y_n, torch.ones_like(u_n), create_graph=True, retain_graph=True)[0]
    dp_dx = torch.autograd.grad(p_n, x_n, torch.ones_like(p_n), create_graph=True, retain_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_n, torch.ones_like(du_dx), create_graph=True, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y_n, torch.ones_like(du_dy), create_graph=True, retain_graph=True)[0]

    #gradients for y-momentium
    dv_dt = torch.autograd.grad(v_n, t_n, torch.ones_like(v_n), create_graph=True, retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v_n, x_n, torch.ones_like(v_n), create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v_n, y_n, torch.ones_like(v_n), create_graph=True, retain_graph=True)[0]
    dp_dy = torch.autograd.grad(p_n, y_n, torch.ones_like(p_n), create_graph=True, retain_graph=True)[0]
    d2v_dx2 = torch.autograd.grad(dv_dx, x_n, torch.ones_like(dv_dx), create_graph=True, retain_graph=True)[0]
    d2v_dy2 = torch.autograd.grad(dv_dy, y_n, torch.ones_like(dv_dy), create_graph=True, retain_graph=True)[0]

    return du_dt, du_dx, du_dy, dp_dx, d2u_dx2, d2u_dy2, dv_dt, dv_dx, dv_dy, dp_dy, d2v_dx2, d2v_dy2

##### Navier-Stokes equations #####
def navier_stokes_residuals(x_n, y_n, t_n, re, g_x, g_y, current_model):

    u_n, v_n, p_n = current_model(x_n , y_n, t_n) #predicted velocity, pressure values

    du_dt, du_dx, du_dy, dp_dx, d2u_dx2, d2u_dy2, dv_dt, dv_dx, dv_dy, dp_dy, d2v_dx2, d2v_dy2 = gradients(u_n, v_n, p_n, x_n, y_n, t_n)

    #x-momentum residual calculation
    residual_x = du_dt+u_n*du_dx+v_n*du_dy+dp_dx-(1/re)*(d2u_dx2+d2u_dy2)+g_x

    #y-momentum residual calculation
    residual_y = dv_dt+u_n*dv_dx+v_n*dv_dy+dp_dy-(1/re)*(d2v_dx2+d2v_dy2)+g_y

    #continuity equation residual calculation
    residual_c = du_dx+dv_dy

    return residual_x, residual_y, residual_c

# Total Loss Function
def total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc, x_n, y_n, t_n, x_ic, y_ic, t_ic, u_ic_true, v_ic_true, x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true, x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true, re, g_x, g_y, current_model):

    residual_x, residual_y, residual_c = navier_stokes_residuals(x_n, y_n, t_n, re, g_x, g_y, current_model)

    loss_ic = ic_fn(x_ic, y_ic, t_ic, u_ic_true, v_ic_true, current_model) #loss for initial condition

    loss_bc_0 = bc_fn_0(x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true, current_model) #loss for left, right, bottom wall boundary condition
    loss_bc_1 = bc_fn_1(x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true, current_model) #loss for top wall boundary condition
    loss_bc = loss_bc_1 + loss_bc_0

    mse_x = loss_function(residual_x, torch.zeros_like(residual_x)) #loss for x-momentum
    mse_y = loss_function(residual_y, torch.zeros_like(residual_y)) #loss for y-momentum
    mse_c = loss_function(residual_c, torch.zeros_like(residual_c)) #loss for continuity equation

    total_loss = (lambda_x*mse_x + lambda_y*mse_y + lambda_c*mse_c +lambda_ic*loss_ic + lambda_bc*loss_bc) #total loss

    return total_loss, loss_bc, loss_ic

# Define data for boundary and initial conditions (can be reused)
# initial condition dataset
x_ic = torch.rand(100,1, requires_grad = True, device=device)*2 - 1
y_ic = torch.rand(100,1, requires_grad = True, device=device)*2 - 1
t_ic = torch.zeros(100,1, requires_grad = True, device=device)

u_ic_true = torch.zeros(100,1, device=device) #true initial condition value for u
v_ic_true = torch.zeros(100,1, device=device) #true initial condition value for v

# left wall condition dataset
x_bc_left = -torch.ones(100,1, requires_grad = True, device=device) #x=-1
y_bc_left = torch.rand(100,1, requires_grad = True, device=device)*2-1 #y values between -1 and 1
t_bc_left = torch.rand(100, 1, requires_grad=True, device=device)

# right wall condition dataset
x_bc_right = torch.ones(100,1, requires_grad = True, device=device) #x=1
y_bc_right = torch.rand(100,1, requires_grad = True, device=device)*2-1 #y values between -1 and 1
t_bc_right = torch.rand(100, 1, requires_grad=True, device=device)

# bottom wall condition dataset
x_bc_bottom = torch.rand(100,1, requires_grad = True, device=device)*2-1 #x values between -1 and 1
y_bc_bottom = -torch.ones(100,1, requires_grad = True, device=device) #y=-1
t_bc_bottom = torch.rand(100, 1, requires_grad=True, device=device)

# left, right, bottom wall condition true values dataset
u_bc_0_true = torch.zeros(300,1, requires_grad = True, device=device) #true boundary condition value for u
v_bc_0_true = torch.zeros(300,1, requires_grad = True, device=device) #true boundary condition value for v

x_bc_0 = torch.cat((x_bc_left, x_bc_right, x_bc_bottom), dim=0) #combined x values at boundary conditions
y_bc_0 = torch.cat((y_bc_left, y_bc_right, y_bc_bottom), dim=0) #combined y values at boundary conditions
t_bc_0 = torch.cat((t_bc_left, t_bc_right, t_bc_bottom), dim=0) #t values between 0 and 1

# top wall condition dataset
x_bc_top = torch.rand(100,1, requires_grad = True, device=device)*2-1 #x values between -1 and 1
y_bc_top = torch.ones(100,1, requires_grad = True, device=device) #y=1
t_bc_top = torch.rand(100,1, requires_grad = True, device=device) #t values between 0 and 1

u_bc_1_true = torch.ones(100,1, requires_grad = True, device=device) #true boundary condition value
v_bc_1_true = torch.zeros(100,1, requires_grad = True, device=device) #true boundary condition value


# TRAINING LOW FIDELITY MODEL
print("** Training low fidility model **")
model = low_fidility_model
epochs = low_fidelity_epochs
collocation_points = low_fidelity_collocation_points

optimizer_adam = torch.optim.Adam(model.parameters(), learning_rate)
# Using CosineAnnealingLR for low fidelity training
learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=epochs)


# collocation points dataset for low fidelity
x_n = torch.rand(collocation_points, 1,requires_grad = True, device=device)*2 - 1 #x space values
y_n = torch.rand(collocation_points, 1,requires_grad = True, device=device)*2 - 1 #y space values
t_n = torch.rand(collocation_points, 1,requires_grad = True, device=device) #t values

loss_history_low = [] #stored list of loss values for low fidelity

for epoch in range (epochs):
    optimizer_adam.zero_grad()

    # Adjust lambdas based on curriculum learning
    if epoch < (epochs/5): #first 20% epochs only training with boundary and initial conditions for better model learning
        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 3.0, 3.0, 3.0, 50.0, 50.0
    elif epoch < ((0.3)*epochs): # less then 30% epochs training
        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 15.0, 15.0, 10.0, 25.0, 25.0
    elif epoch < ((0.95)*epochs): # less then 95% epochs
        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 60.0, 60.0, 45.0, 30.0, 25.0
    else:       #using lbfgs optimizer for last 95% epochs (Note: L-BFGS requires a closure function)
        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 70.0, 70.0, 40.0, 35.0, 25.0

        # L-BFGS optimizer setup for the end of training
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.1,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn='strong_wolfe'
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, _, _ = total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc, x_n, y_n, t_n, x_ic, y_ic, t_ic, u_ic_true, v_ic_true, x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true, x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true, re, g_x, g_y, model)
            total_loss.backward(retain_graph=True)
            return total_loss

        total_loss = optimizer_lbfgs.step(closure)
        loss_history_low.append(total_loss.item())

        if (epoch - ((0.95)*epochs)) % 50 == 0:
             print(f'LBFGS Epoch {epoch}: Loss: {total_loss.item():.6f}') # LR not directly available from L-BFGS step

        continue # Skip Adam step and scheduler update in L-BFGS phase


    total_loss, loss_bc, loss_ic = total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc, x_n, y_n, t_n, x_ic, y_ic, t_ic, u_ic_true, v_ic_true, x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true, x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true, re, g_x, g_y, model)
    total_loss.backward(retain_graph=True)
    optimizer_adam.step()

    learning_rate_scheduler.step() # CosineAnnealingLR does not need the loss in step()
    loss_history_low.append(total_loss.item())

    if epoch % 200 == 0:
        learning_rate = optimizer_adam.param_groups[0]['lr'] # Get current LR from Adam optimizer
        print(f'Loss at epoch {epoch} : {total_loss.item():.6f}, LR : {learning_rate:.6f}')
        print(f'Boundary condition loss at epoch {epoch} : {loss_bc.item():.6f} and Initial condition loss at epoch {epoch} : {loss_ic.item():.6f}')

print(f"Final Loss (Low Fidelity): {loss_history_low[-1]:.6f}")
if len(loss_history_low) > 1:
  print(f"Loss at epoch {epochs-1} : {loss_history_low[-1]} and % Loss decrease : {((loss_history_low[0]-loss_history_low[-1])/loss_history_low[0])*100:.2f}%")


# TRAINING HIGH FIDELITY MODEL
print("\n** Training high fidility model **")
model = high_fidility_model
epochs = high_fidelity_epochs
collocation_points = high_fidelity_collocation_points

optimizer_adam = torch.optim.Adam(model.parameters(), learning_rate) # Reinitialize optimizer for high fidelity
# Using CosineAnnealingLR for high fidelity training
learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=epochs)


# collocation points dataset for high fidelity
x_n = torch.rand(collocation_points, 1,requires_grad = True, device=device)*2 - 1 #x space values
y_n = torch.rand(collocation_points, 1,requires_grad = True, device=device)*2 - 1 #y space values
t_n = torch.rand(collocation_points, 1,requires_grad = True, device=device) #t values


loss_history_high = [] #stored list of loss values for high fidelity

for epoch in range (epochs):
    optimizer_adam.zero_grad()

    # Adjust lambdas based on curriculum learning
    if epoch < (epochs/5): #first 20% epochs only training with boundary and initial conditions for better model learning
        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 3.0, 3.0, 3.0, 50.0, 50.0
    elif epoch < ((0.3)*epochs): # less then 30% epochs training
        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 15.0, 15.0, 10.0, 25.0, 25.0
    elif epoch < ((0.95)*epochs): # less then 95% epochs
        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 60.0, 60.0, 45.0, 30.0, 25.0
    else:       #using lbfgs optimizer for last 95% epochs (Note: L-BFGS requires a closure function)
        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 70.0, 70.0, 40.0, 35.0, 25.0

        # L-BFGS optimizer setup for the end of training
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.1,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn='strong_wolfe'
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, _, _ = total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc, x_n, y_n, t_n, x_ic, y_ic, t_ic, u_ic_true, v_ic_true, x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true, x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true, re, g_x, g_y, model)
            total_loss.backward(retain_graph=True)
            return total_loss

        total_loss = optimizer_lbfgs.step(closure)
        loss_history_high.append(total_loss.item())

        if (epoch - ((0.95)*epochs)) % 50 == 0:
            print(f'LBFGS Epoch {epoch}: Loss: {total_loss.item():.6f}') # LR not directly available from L-BFGS step
        continue # Skip Adam step and scheduler update in L-BFGS phase

    total_loss, loss_bc, loss_ic = total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc, x_n, y_n, t_n, x_ic, y_ic, t_ic, u_ic_true, v_ic_true, x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true, x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true, re, g_x, g_y, model)
    total_loss.backward(retain_graph=True)
    optimizer_adam.step()

    learning_rate_scheduler.step() # CosineAnnealingLR does not need the loss in step()
    loss_history_high.append(total_loss.item())

    if epoch % 200 == 0:
        learning_rate = optimizer_adam.param_groups[0]['lr'] # Get current LR from Adam optimizer
        print(f'Loss at epoch {epoch} : {total_loss.item():.6f}, LR : {learning_rate:.6f}')
        print(f'Boundary condition loss at epoch {epoch} : {loss_bc.item():.6f} and Initial condition loss at epoch {epoch} : {loss_ic.item():.6f}')

print(f"Final Loss (High Fidelity): {loss_history_high[-1]:.6f}")
if len(loss_history_high) > 1:
  print(f"Loss at epoch {epochs-1} : {loss_history_high[-1]} and % Loss decrease : {((loss_history_high[0]-loss_history_high[-1])/loss_history_high[0])*100:.2f}%")


# Training correction model
print("\n** Training correction model **")
correction_epochs = 3000
correction_optimizer = torch.optim.Adam(correction_model.parameters(), lr=0.001)

loss_history_correction = [] # stored list of loss values for correction model

for corr_epoch in range(correction_epochs):
    correction_optimizer.zero_grad()

    #sample points
    x_corr = torch.rand(1000, 1, requires_grad=True, device=device)*2 - 1
    y_corr = torch.rand(1000, 1, requires_grad=True, device=device)*2 - 1
    t_corr = torch.rand(1000, 1, requires_grad=True, device=device)

    #get low fidelity predictions
    with torch.no_grad():
        u_l, v_l, p_l = low_fidility_model(x_corr, y_corr, t_corr)

    #calling correction to output small corrections
    u_c, v_c, p_c = correction_model(x_corr, y_corr, t_corr, u_l, v_l, p_l)

    #corrections loss function
    loss = torch.mean(u_c**2 + v_c**2 + p_c**2)
    loss.backward()
    correction_optimizer.step()
    loss_history_correction.append(loss.item())


    if corr_epoch % 500 == 0:
        print(f'Correction training epoch {corr_epoch}: Loss = {loss.item():.6f}')

print(f"Final Loss (Correction): {loss_history_correction[-1]:.6f}")
if len(loss_history_correction) > 1:
    print(f"Loss at epoch {correction_epochs-1} : {loss_history_correction[-1]} and % Loss decrease : {((loss_history_correction[0]-loss_history_correction[-1])/loss_history_correction[0])*100:.2f}%")


###### visualization #####
# Use the FINAL MultiFidelity model
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
x_flat = X.flatten()
y_flat = Y.flatten()
t_fixed = 0.8

# Transfer inputs to GPU
inputs_x = torch.tensor(x_flat.reshape(-1, 1), dtype=torch.float32, device=device)
inputs_y = torch.tensor(y_flat.reshape(-1, 1), dtype=torch.float32, device=device)
inputs_t = torch.full_like(inputs_x, t_fixed)

# Using MultiFidelity model for final predictions
MultiFidelity.eval()
with torch.no_grad():
    u_n_pred, v_n_pred, p_n_pred = MultiFidelity(inputs_x, inputs_y, inputs_t)

    u_n_plot = u_n_pred.cpu().numpy().reshape(50, 50)
    v_n_plot = v_n_pred.cpu().numpy().reshape(50, 50)
    p_n_plot = p_n_pred.cpu().numpy().reshape(50, 50)

#plotting
plt.figure(figsize=(15, 5))

# Velocity Plot
plt.subplot(1, 3, 1)
plt.quiver(X, Y, u_n_plot, v_n_plot, scale=30, color='blue')
plt.title('Velocity Field (u_n, v_n)')
plt.xlabel('X')
plt.ylabel('Y')

# Pressure Plot
plt.subplot(1, 3, 2)
contour = plt.contourf(X, Y, p_n_plot, levels=50, cmap='viridis')
plt.colorbar(contour, label='Pressure (p_n)')
plt.title('Pressure Contour')
plt.xlabel('X')
plt.ylabel('Y')

# Streamlines Plot
plt.subplot(1, 3, 3)
plt.streamplot(X, Y, u_n_plot, v_n_plot, color='black', linewidth=1, density=2)
plt.title('Streamlines')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.savefig('navier_stokes_results.png')
plt.show()

# Loss Curve plot
plt.figure()
plt.plot(loss_history_low, label='Low Fidelity')
plt.plot(loss_history_high, label='High Fidelity')
plt.plot(loss_history_correction, label='Correction')
plt.yscale('log')
plt.title('Training Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.savefig('loss_curves.png')
plt.show()
