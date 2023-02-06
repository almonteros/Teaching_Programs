# Teaching_Programs
Programs I have created related to teach Theoretical Physics or Computational Physics

SE_solutions
  Provides a visualization of some analytic solutions to Schrodinger's equations

Motion Simulator
  Displays 2D parametric motion. For example this can be used to visualize projectile motion or circular motion.
  Enter the equations for x(t) and y(t) using Python syntax. The only variable allowed is 't'. Press 'start' when ready.
  You can enter the limits as well and change the limits as the program runs by pressing 'Set limits'.
  Finish the simulation by pressing 'Stop'. To change the function you must hit "Stop' before pressing 'Start' again.
  For the interested reader, the equation is fed to Sympy's lambdify function which turns the equation into a usable format.

Romberg_Integration_Example
  A small program that I designed and wrote to aid in understanding and visualizing Romberg integration.
  The notation and the equations used within the code are taken from the book "Computational Physics" by Newman.
  Romberg integration is an adaptive approach to numerically calculating an integral. In this example the integration is done with trapezoidal slices. The adaptive
  part comes from the number of slices we use to calculate the integral.
  
  This involves five steps:
    Calculate the integral with a set number of slices
    Double the number of slices
    Calculate the error between the previous integrations
    Correct the integration with the calculated error
    Repeat steps 2-4 until the desired accuracy is reached.

Computational programming exercises
  Worked exercises based on Newman's book.
 
  Integration
    Integration techniques here include regular or adaptive forms of Simpson's rule, trapezoidal rule, and Romberg integration. Additionally there are a few
    numerical derivative taken.
  Non-linear
    There are a variety of programs here that solve non-linear equations. Techniques include the relaxation method, over-relaxation, binary search, Newton's method,
    and the golden ratio search. The problems include both single and multivariable situations.
  ODE
    This examined methods used in solving ordinary differential equations. This includes programs for the 4th order Runge-Kutta, adaptive Runge-Kutta, the leapfrog
    method, Verlot method, Modified midpoint method, and the Bulirsh_Stoer method. The types of system include circuits, the lorentz-equations, gravitational orbits,
    a 1D quantum eigenvalue problem solved by minimizing the energy, the double pendulum and more. There are a variety of plots including the 4 included in the folder.
    I find the double pendulum animation quite mesmerizing.
  PDE
    The programs here look at techniques in partial differential equations. I make use of the Jacobi method, Forward-Time Centered-Space method, Crank-Nicolson method,
    to look at the diffusion equation, the wave equation, and more. 
  Random_Simulations
    The programs in this file go through several key concepts in the use of random variables for computations. In particular I look at random walks,
    Monte-Carlo-integration (with and without importance sampling), Markov-chain-Monte-Carlo, and simulated annealing. Some of the exercises have animations which I 
    find fascinating to watch.
