# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:10:59 2023

@author: almon
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_e, hbar


def Jacobi_method_charge_density():
    """Consider  square box with walls held at 0 potential. The box has two
    small squres with charge density +1, -1 c/m^2. Solve phi such that
    grad^2 phi = -rho/eps_0."""
    # Parameters
    eps_0 = 1
    L = 1
    num_grid_squares = 100

    # Spacing
    a = L / num_grid_squares

    target_accuracy = 1e-6

    phi = np.zeros((num_grid_squares+1, num_grid_squares+1), float)

    # Charge density
    rho = np.zeros(phi.shape, float)

    # Add the negative box
    rho_neg = -1
    length = 0.2
    distance = 0.2
    # Coordinates of the corner of the negative box
    x_1 = int(distance/a)
    y_1 = int(distance/a)
    x_2 = int((length + distance)/a)
    y_2 = int((length + distance)/a)
    # Set the charge density for the negative box
    rho[y_1:y_2, x_1:x_2] = rho_neg

    # Add the positive box
    rho_plus = 1
    length = 0.2
    distance = 0.2
    # Coordinates of the corner of the positive box
    x_1 = int((L - distance - length)/a)
    y_1 = int((L - distance - length)/a)
    x_2 = int((L - distance)/a)
    y_2 = int((L - distance)/a)

    # Set the charge density for the positive box
    rho[y_1:y_2, x_1:x_2] = rho_plus

    # Start with phi, plug in to the differential equation to find phi and
    # repeat until the error is small
    error = 1
    while error > target_accuracy:
        # Discritize and rearrange differential equation
        # np.roll shift the value over np.roll(phi, 1, axis=0) = phi(x + a, y)
        phi_prime = (np.roll(phi, 1, axis=0)
                     + np.roll(phi, -1, axis=0)
                     + np.roll(phi, 1, axis=1)
                     + np.roll(phi, -1, axis=1))/4
        + a**2/(4*eps_0)*rho

        # Boundaries
        phi_prime[0, :] = phi[0, :]
        phi_prime[-1, :] = phi[-1, :]
        phi_prime[:, 0] = phi[:, 0]
        phi_prime[:, -1] = phi[:, -1]

        # Find the largest error
        error = np.max(abs(phi - phi_prime))
        phi = phi_prime

    plt.figure()
    plt.imshow(phi, cmap='gray', origin='lower')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()


def Jacobi_method_set_voltage():
    """Consider  square box with walls held at 0 potential. The box has two
    lines of voltage held at +1 and -1 Volts. Solve phi"""
    # Parameters
    num_grid_squares = 100
    V_1 = 1
    V_2 = -1
    L = 0.1
    target_accuracy = 1e-6

    # Spacing
    a = L/num_grid_squares
    # Position of the lines of voltage
    x_1 = 0.02
    x_2 = 0.08
    y_initial = 0.02
    y_final = 0.08
    # Indices of the lines of the positions
    i_1 = int(x_1/a)
    i_2 = int(x_2/a)
    j_initial = int(y_initial/a)
    j_final = int(y_final/a)
    # Walls are at 0 voltage
    phi = np.zeros((num_grid_squares+1, num_grid_squares+1), float)
    phi[j_initial:j_final, i_1] = V_1
    phi[j_initial:j_final, i_2] = V_2

    error = 1
    while error > target_accuracy:
        phi_prime = (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
                     + np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1))/4
        # Boundary conditions

        # Walls
        phi_prime[0, :] = phi[0, :]
        phi_prime[-1, :] = phi[-1, :]
        phi_prime[:, 0] = phi[:, 0]
        phi_prime[:, -1] = phi[:, -1]
        # Lines of voltage
        phi_prime[j_initial:j_final, i_1] = V_1
        phi_prime[j_initial:j_final, i_2] = V_2

        # Find the error
        error = np.max(abs(phi - phi_prime))
        phi = phi_prime

    plt.figure()
    plt.imshow(phi, cmap='gray', origin="lower")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()


def FTCS_1D_diffusion():
    """Use the forward-time centered-space method to find how the temperature
    inside the earth fluctates. Consider the temperate of the surface to be
    T(t) = A + Bsin(2 pi t/tau)"""
    h = 0.1  # Time step, days
    # All time parameters need to be scaled by the time step
    # Parameters
    tau = 365/h
    A = 10
    B = 12
    # Diffusion parameter
    D = 0.1*h
    # Temperature deep with the crust is roughly constant
    T_low = 11

    def T_high(t):
        return A + B*np.sin(2*np.pi*t/tau)

    # T_t = D T_xx
    year = int(365/h)
    # h mujst be less than or equal to a^2/(2D) for stability
    N = 100  # Number of steps for the depth
    L = 20  # Depth into the earth
    a = L/N

    depth = np.linspace(-L, 0, N + 1)
    time = np.arange(0, 10*year + 1, h)

    # Temperature
    T = np.full(N + 1, 10, float)
    T[0] = T_low
    T[-1] = T_high(0)

    plt.figure()
    # Plot every 3 months after 9 years
    day = 0
    month = 0

    c = h*D/a**2
    # Propagate forwards in time
    for t in time:
        T[-1] = T_high(t)
        # Diffusion equation
        T[1:-1] = T[1:-1] + c*(T[2:] + T[:-2] - 2*T[1:-1])
        # After year 9 plot the Temperature every 90 days
        if t/year > 9:
            day += h
            if day >= 90/h:
                day = 0
                month += 3
                plt.plot(depth, T, label=f"month {month}")
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Temperature")


def FTCS_wave_equation_unstable():
    """Solve the 1D wave equation after an initial impact"""

    def impact(x, L):
        # Parameters
        d = 0.1
        C = 1
        sigma = 0.3
        return C*x*(L - x)/L**2*np.exp(-(x - d)**2/(2*sigma**2))

    # Parameters
    L = 1  # Length of string
    v = 100  # Wave velocity
    h = 1e-4  # If you go above this the solution becomes unstable very quickly
    N = 100
    a = L/N
    x = np.linspace(0, L, N + 1)
    # Initial conditions
    phi = np.zeros(N + 1, float)
    phi_dot = impact(x, L)

    # Setting up the figures
    fig = plt.figure()
    fig_num = fig.number
    ax1 = fig.add_subplot(111)
    # Set initial data
    line1, = ax1.plot(x, np.zeros(N + 1), '-', lw=2)
    # Label axis
    ax1.set_xlim(0, L)
    ax1.set_ylim(-0.0005, 0.0005)
    ax1.set_ylabel(r"$y$")
    ax1.set_title(r"$\phi(x)$")
    ax1.set_xlabel(r"$x$")
    plt.show(block=False)

    fig.canvas.draw()
    while plt.fignum_exists(fig_num):
        # FTCS
        phi[1:-1] = phi[1:-1] + h*phi_dot[1:-1]
        phi_dot[1:-1] = phi_dot[1:-1]\
            + h*v**2/a**2*(phi[2:] + phi[:-2] - 2*phi[1:-1])

        line1.set_ydata(phi)

        # Redraw the graph.
        ax1.draw_artist(ax1.patch)
        ax1.draw_artist(line1)

        fig.canvas.update()
        fig.canvas.flush_events()


def Crank_Nicolson():
    """Use the Crank-Nicolson method to solve the 1D Schrodinger equation"""

    def initial_wave(x, L):
        # Parameters
        x0 = L/2
        sigma = 1e-10
        kappa = 5e10
        return np.exp(-(x - x0)**2/(2*sigma**2))*np.exp(1j*kappa*x)
    # Parameters
    L = 1e-8
    N = 1000
    a = L/N
    h = 1e-18
    # Spacing equal to a
    x = np.linspace(0, L, N + 1)
    psi = initial_wave(x, L)
    # Normalize
    C = sum(abs(psi)**2)**0.5
    psi = psi/C

    # Setting up the matricies A psi(t + h) = B psi(t)
    a1 = 1 + h*1j*hbar/(2*m_e*a**2)
    a2 = -h*1j*hbar/(4*m_e*a**2)

    b1 = 1 - h*1j*hbar/(2*m_e*a**2)
    b2 = h*1j*hbar/(4*m_e*a**2)

    # Diagonal elements
    A = a1*np.eye(N+1, dtype=float)
    B = b1*np.eye(N+1, dtype=float)
    # Off-diagonal elements
    for i in range(N):
        A[i, i+1] = a2
        A[i+1, i] = a2

        B[i, i+1] = b2
        B[i+1, i] = b2
    A[-1, -2] = a2
    B[-1, -2] = b2

    # A Tridiagonal solver would be faster
    A_inv = np.linalg.inv(A)

    # Setting up the figures
    fig = plt.figure()
    fig_num = fig.number
    ax1 = fig.add_subplot(111)

    # Set initial data
    line1, = ax1.plot(x, np.zeros(N + 1), '-', lw=2)

    # Label axis
    ax1.set_xlim(0, L)
    ax1.set_ylim(-1, 1)
    ax1.set_ylabel(r"$y$")
    ax1.set_title(r"$\phi(x)$")
    ax1.set_xlabel(r"$x$")
    plt.show(block=False)

    fig.canvas.draw()
    t = 0
    U = A_inv@B
    while plt.fignum_exists(fig_num):
        psi = U@psi  # tri_band(A, B@psi)
        line1.set_ydata(psi.real)

        # Redraw the graph.
        ax1.draw_artist(ax1.patch)
        ax1.draw_artist(line1)

        fig.canvas.update()
        fig.canvas.flush_events()

        t += h
