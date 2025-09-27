# Physics-Informed Neural Networks for Fluid Dynamics
A comparative study of Physics-Informed Neural Networks (PINNs) for solving 1-D Viscous Burgers' equation and incompressible Navier-Stokes 2-D equations. Analyzing performance, convergence, and limitations.

## Currently In Progress 
This repository is under active development.
*   Phase 1: Burger's Equation - **COMPLETED** 
*   Phase 2: Navier-Stokes Equation - **COMPLETED** 
*   Phase 3: Comparative Analysis - In Progress

The code and results for solving the Viscous Burger's and Navier-Stokes Equation is now available in the `src/` and `results/` directory respectively.

For burgers equation the convergence of loss and the shock wave was obtained after few tuning of hyperparameters for 6 times. Still the loss curve is having the fluctions between 0.1 to 0.01 but its still managable can can be easily refined due to the equation's simplicity, BUT,

For The Navier-Stokes Equation with lid cavity driven conditions struggle persists, loss converges to 0.02 after trying 10000 epochs of training each time for 14 times after tuning hyperparameters, using different optimization techniques and activation functions, curriculum training approach BUT still the model doesn't shows the good results because the streamlines shows that the model is not following boundary conditions.

Now focusing on adding the vanishing function for boundary condition in forward call for Navier-Stokes eqn. for model to predict values satisfying the BC's before training even happens.

For solving we took, initial condition as u(x,0) = f(x) where f(x) is initial state of system so we took f(x) as -sin(pi*x) & Dirichlet boundary condition for Burgers' equation AND lid cavity driven experiment conditions for Navier-Stokes Equation.

