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

        u_c, v_c, p_c = self.correction_model(x, y, t) #correction model output



        weight = torch.sigmoid(u_c, v_c, p_c) #weighting factor between 0 and 1 using sigmoid activation function



        w_u = weight[:, 0:1] #weight for velocity in x direction

        w_v = weight[:, 1:2] #weight for velocity in x direction

        w_p = weight[:, 2:3] #weight for velocity in x direction



        u = w_u*u_l + (1-w_u)*u_h  #corrected velocity in x direction

        v = w_v*v_l + (1-w_v)*v_h  #corrected velocity in y direction

        p = w_p*p_l + (1-w_p)*p_h  #corrected pressure



        return u, v, p



#hyperparameters settings

low_fidility_model = MultiFidelityNN.FidilityNN(input_size=3, output_size=3, hidden_layers=5, neurons_per_layer=49).to(device)



high_fidility_model = MultiFidelityNN.FidilityNN(input_size=3, output_size=3, hidden_layers=5, neurons_per_layer=79).to(device)



correction_model = MultiFidelityNN.correctionNN(input_size=6, output_size=3, hidden_layers=5, neurons_per_layer=67).to(device)



MultiFidelity = MultiFidelityNN(low_fidility_model, high_fidility_model, correction_model).to(device)



#model selection according to fidility

fidility = ["low", "high"]

for fidility in fidility:

    if fidility == "low":

        epochs = 2000

        collocation_points = 1000

        print("** Training low fidility model **")

        model = low_fidility_model

    else:

        epochs = 5000

        collocation_points = 8000

        print("** Training high fidility model **")

        model = high_fidility_model



learning_rate = 0.0001



re = 100  #Reynolds number

g_x = 0.0 #external force in x direction

g_y = 0.0 #external force in y direction



loss_function = nn.MSELoss()

optimizer_adam = torch.optim.Adam(model.parameters(), learning_rate) #using adam optimizer



learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, 

                                                                     mode='min', factor=0.1, 

                                                                     patience=10, threshold=0.0001, 

                                                                     threshold_mode='rel', cooldown=0, 

                                                                     min_lr=0, eps=1e-08) #using CosineAnnealingLR learning rate scheduler only for 9500 epochs



optimizer_lbfgs = torch.optim.LBFGS(  #using lbfgs optimizer

    model.parameters(), 

    lr=0.1, 

    max_iter=20, 

    max_eval=25,

    tolerance_grad=1e-7,

    tolerance_change=1e-9,

    history_size=50,

    line_search_fn='strong_wolfe'

)



lambda_x = 50.0 #x-momentum loss weight

lambda_y = 50.0 #y-momentum loss weight

lambda_c = 30.0 #continuity equation loss weight

lambda_ic = 2.0 #initial condition loss weight

lambda_bc = 2.0  #boundary condition loss weight





##### initial condition #####



#initial condition dataset

x_ic = torch.rand(100,1, requires_grad = True, device=device)*2 - 1

y_ic = torch.rand(100,1, requires_grad = True, device=device)*2 - 1

t_ic = torch.zeros(100,1, requires_grad = True, device=device)



u_ic_true = torch.zeros(100,1, device=device) #true initial condition value for u

v_ic_true = torch.zeros(100,1, device=device) #true initial condition value for v



#initial condition loss

def ic_fn(x_ic, y_ic, t_ic, u_ic_true, v_ic_true):

  

    u_ic_pred, v_ic_pred, p_ic_pred = model(x_ic, y_ic, t_ic) #predicted initial condition values



    loss_ic = loss_function(u_ic_pred, u_ic_true) + loss_function(v_ic_pred, v_ic_true) #initial condition loss



    return loss_ic



###### boundary condition ######



#left wall condition dataset

x_bc_left = -torch.ones(100,1, requires_grad = True, device=device) #x=-1

y_bc_left = torch.rand(100,1, requires_grad = True, device=device)*2-1 #y values between -1 and 1

t_bc_left = torch.rand(100, 1, requires_grad=True, device=device)



#right wall condition dataset

x_bc_right = torch.ones(100,1, requires_grad = True, device=device) #x=1

y_bc_right = torch.rand(100,1, requires_grad = True, device=device)*2-1 #y values between -1 and 1

t_bc_right = torch.rand(100, 1, requires_grad=True, device=device)



#bottom wall condition dataset

x_bc_bottom = torch.rand(100,1, requires_grad = True, device=device)*2-1 #x values between -1 and 1

y_bc_bottom = -torch.ones(100,1, requires_grad = True, device=device) #y=-1

t_bc_bottom = torch.rand(100, 1, requires_grad=True, device=device)



#left, right, bottom wall condition true values dataset

u_bc_0_true = torch.zeros(300,1, requires_grad = True, device=device) #true boundary condition value for u

v_bc_0_true = torch.zeros(300,1, requires_grad = True, device=device) #true boundary condition value for v



x_bc_0 = torch.cat((x_bc_left, x_bc_right, x_bc_bottom), dim=0) #combined x values at boundary conditions

y_bc_0 = torch.cat((y_bc_left, y_bc_right, y_bc_bottom), dim=0) #combined y values at boundary conditions

t_bc_0 = torch.cat((t_bc_left, t_bc_right, t_bc_bottom), dim=0) #t values between 0 and 1



#left, right, bottom boundary condition loss

def bc_fn_0(x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true):



    u_bc_0_pred, v_bc_0_pred, p_bc_0_pred = model(x_bc_0, y_bc_0, t_bc_0) #left, right, bottom wall condition boundary condition predicted values



    loss_bc_0 = loss_function(u_bc_0_pred, u_bc_0_true) + loss_function(v_bc_0_pred, v_bc_0_true) #boundary condition loss

    

    return loss_bc_0



#top wall condition dataset

x_bc_top = torch.rand(100,1, requires_grad = True, device=device)*2-1 #x values between -1 and 1

y_bc_top = torch.ones(100,1, requires_grad = True, device=device) #y=1

t_bc_top = torch.rand(100,1, requires_grad = True, device=device) #t values between 0 and 1



u_bc_1_true = torch.ones(100,1, requires_grad = True, device=device) #true boundary condition value

v_bc_1_true = torch.zeros(100,1, requires_grad = True, device=device) #true boundary condition value



#top boundary condition loss

def bc_fn_1(x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true):



    u_bc_1_pred, v_bc_1_pred, p_bc_1_pred = model(x_bc_top, y_bc_top, t_bc_top) #top wall boundary condition predicted value



    loss_bc_1 = loss_function(u_bc_1_pred, u_bc_1_true) + loss_function(v_bc_1_pred, v_bc_1_true) #boundary condition loss



    return loss_bc_1 





#collocation points dataset

x_n = torch.rand(collocation_points, 1,requires_grad = True, device=device)*2 - 1 #x space values

y_n = torch.rand(collocation_points, 1,requires_grad = True, device=device)*2 - 1 #y space values

t_n = torch.rand(collocation_points, 1,requires_grad = True, device=device) #t values



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

def navier_stokes_residuals(x_n, y_n, t_n, re, g_x, g_y):



    u_n, v_n, p_n = model(x_n , y_n, t_n) #predicted velocity, pressure values



    du_dt, du_dx, du_dy, dp_dx, d2u_dx2, d2u_dy2, dv_dt, dv_dx, dv_dy, dp_dy, d2v_dx2, d2v_dy2 = gradients(u_n, v_n, p_n, x_n, y_n, t_n)



    #x-momentum residual calculation

    residual_x = du_dt+u_n*du_dx+v_n*du_dy+dp_dx-(1/re)*(d2u_dx2+d2u_dy2)+g_x



    #y-momentum residual calculation

    residual_y = dv_dt+u_n*dv_dx+v_n*dv_dy+dp_dy-(1/re)*(d2v_dx2+d2v_dy2)+g_y



    #continuity equation residual calculation

    residual_c = du_dx+dv_dy



    return residual_x, residual_y, residual_c



# Total Loss Function

def total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc):



    residual_x, residual_y, residual_c = navier_stokes_residuals(x_n, y_n, t_n, re, g_x, g_y)

    

    loss_ic = ic_fn(x_ic, y_ic, t_ic, u_ic_true, v_ic_true) #loss for initial condition



    loss_bc_0 = bc_fn_0(x_bc_0, y_bc_0, t_bc_0, u_bc_0_true, v_bc_0_true) #loss for left, right, bottom wall boundary condition

    loss_bc_1 = bc_fn_1(x_bc_top, y_bc_top, t_bc_top, u_bc_1_true, v_bc_1_true) #loss for top wall boundary condition

    loss_bc = loss_bc_1 + loss_bc_0

    

    mse_x = loss_function(residual_x, torch.zeros_like(residual_x)) #loss for x-momentum

    mse_y = loss_function(residual_y, torch.zeros_like(residual_y)) #loss for y-momentum

    mse_c = loss_function(residual_c, torch.zeros_like(residual_c)) #loss for continuity equation



    total_loss = (lambda_x*mse_x + lambda_y*mse_y + lambda_c*mse_c +lambda_ic*loss_ic + lambda_bc*loss_bc) #total loss



    return total_loss, loss_bc, loss_ic



loss_history = [] #stored list of loss values



#using Curriculum training approach

for epoch in range (epochs):

    if epoch < (epochs/5): #first 20% epochs only training with boundary and initial conditions for better model learning

        optimizer_adam.zero_grad() #set gradients to zero



        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 3.0, 3.0, 3.0, 50.0, 50.0



        total_loss = total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[0] #calculate total loss

        total_loss.backward(retain_graph=True) #backpropagation

        optimizer_adam.step() #update weights



        learning_rate_scheduler.step()

        loss_history.append(total_loss.item()) #store loss value in list



        if epoch % 200 == 0:

            print(f'Boundary condition loss at epoch {epoch} : {total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[1]:.6f} and Initial condition loss at epoch {epoch} : {total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[2]:.6f}')

    elif epoch < ((3*epochs)/10): # less then 30% epochs training 

        optimizer_adam.zero_grad() #set gradients to zero



        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 15.0, 15.0, 10.0, 25.0, 25.0



        total_loss = total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[0] #calculate total loss

        total_loss.backward(retain_graph=True) #backpropagation

        optimizer_adam.step() #update weights



        learning_rate_scheduler.step()

        loss_history.append(total_loss.item()) #store loss value in list



        if epoch % 200 == 0:

            print(f'Boundary condition loss at epoch {epoch} : {total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[1]:.6f} and Initial condition loss at epoch {epoch} : {total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[2]:.6f}')

    elif epoch < ((19*epochs)/20): # less then 95% epochs 

        optimizer_adam.zero_grad() #set gradients to zero

        

        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 60.0, 60.0, 45.0, 30.0, 25.0



        total_loss = total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[0] #calculate total loss

        total_loss.backward(retain_graph=True) #backpropagation

        optimizer_adam.step() #update weights



        learning_rate_scheduler.step() #update learning rate based on loss

        loss_history.append(total_loss.item()) #store loss value in list

        

        if epoch % 200 == 0:

            learning_rate_scheduler.get_last_lr()[0]

            print(f'Loss at epoch {epoch} : {total_loss.item():.6f}, LR : {learning_rate:.6f}')

            print(f'Boundary condition loss at epoch {epoch} : {total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[1]:.6f} and Initial condition loss at epoch {epoch} : {total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[2]:.6f}')

    

    

    

    else:       #using lbfgs optimizer for last 95% epochs

        lambda_x, lambda_y, lambda_c, lambda_bc, lambda_ic = 70.0, 70.0, 40.0, 35.0, 25.0

        def closure():

            optimizer_lbfgs.zero_grad()

            total_loss = total_loss_function(lambda_x, lambda_y, lambda_c, lambda_ic, lambda_bc)[0]

            total_loss.backward(retain_graph=True)

            return total_loss

        

        total_loss = optimizer_lbfgs.step(closure)

        loss_history.append(total_loss.item())

        

        if (epoch - 9500) % 50 == 0:

            print(f'LBFGS Epoch {epoch}: Loss: {total_loss.item():.6f}')

    

print(f"Final Loss: {loss_history[-1]:.6f}")





print(f"Loss at epoch {epoch} : {loss_history[-1]} and % Loss decrease : {(loss_history[0]-loss_history[-1]/loss_history[0])*100:.2f}%")





###### visualization #####

#plotting grid according to domain [-1,1] for x,y

x = np.linspace(-1, 1, 50)

y = np.linspace(-1, 1, 50)

X, Y = np.meshgrid(x, y)

x_flat = X.flatten()

y_flat = Y.flatten()

t_fixed = 0.8  # Fixed time



#transfer inputs to GPU

inputs_x = torch.tensor(x_flat.reshape(-1, 1), dtype=torch.float32, device=device)

inputs_y = torch.tensor(y_flat.reshape(-1, 1), dtype=torch.float32, device=device) 

inputs_t = torch.full_like(inputs_x, t_fixed)  # Same time for all points



#output predictions

model.eval()

with torch.no_grad():

    u_n_pred, v_n_pred, p_n_pred = model(inputs_x, inputs_y, inputs_t)

    

    #from GPU to CPU predected data transfer, them transformed tensor to array using numpy

    u_n_plot = u_n_pred.cpu().numpy().reshape(50, 50)

    v_n_plot = v_n_pred.cpu().numpy().reshape(50, 50)

    p_n_plot = p_n_pred.cpu().numpy().reshape(50, 50)



# 4. Plotting

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

plt.savefig('navier_stokes_results.png') #image save

plt.show()



# Loss Curve plot

plt.figure()

plt.plot(loss_history)

plt.yscale('log')  #log scale for better visualization

plt.title('Training Loss Curve')

plt.xlabel('Epoch')

plt.ylabel('Loss (log scale)')

plt.savefig('loss_curve.png')

plt.show()