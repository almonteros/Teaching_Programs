# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 21:47:30 2023

@author: almon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random


def atomic_decay():
    """Model the atomic decay of Bi 213"""
    # Time step
    del_t = 1

    # Initial number of atoms of each kind
    N_Bi_213 = 10000
    N_Tl = 0
    N_Pb = 0
    N_Bi_209 = 0

    # Keep a list of how the atoms decay
    Bis_213 = [N_Bi_213]
    Tls = [N_Tl]
    Pbs = [N_Pb]
    Bis_209 = [N_Bi_209]

    # Half lives
    tau_Bi_213 = 46*60
    tau_Tl = 2.2*60
    tau_Pb = 3.3*60

    # Probability that a single atom has decayed
    p_Bi_213 = 1 - 2**(-del_t/tau_Bi_213)
    p_Tl = 1 - 2**(-del_t/tau_Tl)
    p_Pb = 1 - 2**(-del_t/tau_Pb)

    # Time to run the simulation
    t_final = 20001
    time = np.arange(0, t_final, del_t)
    for t in time[1:]:
        # Decay chain, moving from Pb upwards.
        for i in range(N_Pb):
            if random() < p_Pb:
                N_Bi_209 += 1
                N_Pb -= 1
        for i in range(N_Tl):
            if random() < p_Tl:
                N_Pb += 1
                N_Tl -= 1
        for i in range(N_Bi_213):
            if random() < p_Bi_213:
                if random() < 0.9791:
                    N_Pb += 1
                else:
                    N_Tl += 1
                N_Bi_213 -= 1

        Bis_213.append(N_Bi_213)
        Tls.append(N_Tl)
        Pbs.append(N_Pb)
        Bis_209.append(N_Bi_209)

    plt.figure()
    plt.plot(time, Bis_213, label="Bi 213")
    plt.plot(time, Tls, label="Tl")
    plt.plot(time, Pbs, label="Pb")
    plt.plot(time, Bis_209, label="Bi 209")
    plt.legend()


def random_walk():
    """Random walk"""
    # Size of grid
    L = 101
    # Starting point
    x, y = L//2 + 1, L//2 + 1

    # Probabilities to move
    p_x_up = 0.25
    p_x_down = 0.25
    p_y_up = 0.25
    p_y_down = 0.25

    assert np.isclose(1, p_x_up + p_x_down + p_y_up + p_y_down), \
        "Probabilities should sum to one"

    def move(x, y,):
        r = random()
        #
        if r < p_x_up:
            if x < L:
                x += 1
        elif r < p_x_down + p_x_up:
            if x > 0:
                x -= 1
        elif r < p_y_up + p_x_down + p_x_up:
            if y < L:
                y += 1
        else:
            if y > 0:
                y -= 1
        return x, y

    # Setting up the figures
    fig = plt.figure()
    fig_num = fig.number
    ax1 = fig.add_subplot(111)
    # Set initial data
    line1, = ax1.plot(x, y, marker="o")
    # Label axis
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)
    plt.xlabel("x")
    ax1.set_ylabel("y")

    fig.canvas.draw()

    while plt.fignum_exists(fig_num):
        x, y = move(x, y)
        line1.set_data(x, y)

        # Redraw the graph.
        ax1.draw_artist(ax1.patch)
        ax1.draw_artist(line1)

        fig.canvas.update()
        fig.canvas.flush_events()


def nonuniform_sampling():
    """Transform the uniform sampling into an exponential sampling to model
    atomic decay"""
    # Sampling method
    # Half life
    tau = 3.053*60
    # Number of samples
    N = 1000
    # Random numbers uniformly distributed
    z = np.random.rand(N)

    # Transform the probability distribution to be exponential
    # This gives us a random sampling at what times the decay occurs
    mu = np.log(2)/tau
    time_decay = -1/mu*np.log(1 - z)

    # Sort the times
    time_decay.sort()

    # The number of atoms, after each decay N_Tl goes down and N_Pb goes up
    N_Tl = [i for i in range(N, 0, -1)]
    N_Pb = [i for i in range(0, N, 1)]

    plt.figure()
    plt.plot(time_decay, N_Tl, label="Tl")
    plt.plot(time_decay, N_Pb, label="Pb")
    plt.legend()


def random_integration():
    """Integration using random sampling two ways, Monte-Carlos and mean
    value"""
    def f(x):
        """Function to integrate"""
        return np.sin(1/(x*(2 - x)))**2

    # Number of steps
    N = 10000
    # Integration limits
    a = 0
    b = 2

    # Monte-Carlo
    # Randomly sample across the x-axis
    x = (b - a) * np.random.rand(N)
    # Find values
    fun = f(x)
    # Randomly sample across the y-axis
    y_min = 0  # np.min(fun)
    y_max = 1  # np.max(fun)
    y = (y_max - y_min) * np.random.rand(N)

    # Find the total area of the integration window
    A = (b - a)*(y_max - y_min)
    # Number of points that are below the function
    count = np.sum(y < fun)
    # Integral is the probability that a random point is below the function
    # times the area of the integration window
    I = A*count/N
    # Standard deviation
    sigma = (I*(A - I)/N)**(0.5)
    print(I, sigma)

    # Mean Value
    I = (b - a)*np.mean(fun)
    sigma = (b - a)/N**0.5 * np.std(fun)
    print(I, sigma)


def high_dimensional_monte_carlo():
    """Find the volume of an N-dimensional sphere"""
    def f(r, R):
        """Return 1 if the length of r is less than the radius of the
        hypersphere otherwise returns 0"""
        length = np.sum(r**2, axis=0)**0.5
        return np.where(length <= R, 1, 0)

    # Parameters
    dim = 2
    R = 1
    # Number of samples
    N = 100000
    # Limits of integration for each dimension
    b = np.full((dim, 1), R)
    a = np.full((dim, 1), -R)
    # Volume of the integration window
    V = np.product((b - a))

    # Find N random dim-dimensional vectors between -R and R
    r = R*(2*np.random.rand(dim, N) - 1)

    I = V*np.mean(f(r, R))
    print(I)


def importance_sampling():
    """Integrate the function below from 0 to 1 using importance sampling.
    Weight is w(x) = x^(-1/2)"""

    def f(x):
        return x**(-0.5)/(np.exp(x) + 1)

    def w(x):
        return 1/np.sqrt(x)

    # Number of samples
    N = 1000000
    # Draw from the probability distribution defined by the weighted function
    z = np.random.rand(N)
    x = z**2
    # Integral of w(x) from 0 to 1
    I_w = 2
    I = 1/N*I_w*np.sum(f(x)/w(x))

    # Using the built in weighted average
    # p_z = 1/2*w(z)
    # I = np.average(f(z)/w(z), weights=p_z)

    print(I)


def Markov_chain_Monte_Carlo():
    """Simulate a 2D Ising model using MCMC using the Metropolis algorithm"""
    # Parameters
    J = 1
    # Temperature
    T = 1
    # Boltzmann's constant
    k = 1
    beta = 1/(k*T)
    # Number of spin sites in one dimension for a total of N^2 spins
    N = 20

    # Initial state of the spins, they can be +1 or -1
    s = 2*np.random.randint(0, 2, (N, N)) - 1

    def energy(J, s):
        """Find the current energy of the system assuming nearest neighbor
        interactions"""
        # Shift the array rows over and the multiply, remove the first row
        # assuming open boundary conditions
        spin_sum = np.sum((s*np.roll(s, 1, axis=0))[1:, :])
        # Shift the array columns over and the multiply, remove the first
        # column assuming open boundary conditions
        spin_sum += np.sum((s*np.roll(s, 1, axis=1))[:, 1:])
        return -J*spin_sum

    def magnetization(s):
        """Total magnetization of the system"""
        return np.sum(s)

    def energy_change(s, i, j):
        """Gives the change in energy if s[i, j] is flipped"""
        N = len(s)
        delta_E = 0
        # Sum the spins of all the neighbors
        if i != 0:
            delta_E += s[i-1, j]
        if i != N - 1:
            delta_E += s[i+1, j]
        if j != 0:
            delta_E += s[i, j-1]
        if j != N - 1:
            delta_E += s[i, j+1]
        # Multiply by the center spin. The factor of -2 is because we are
        # switch from -1 to 1 or 1 to -1
        delta_E = -2*(-J*delta_E*s[i, j])
        return delta_E

    E_current = energy(J, s)
    moves = 100000
    M = np.empty(moves)
    for m in range(moves):
        # Get random spin
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        # Get the change in energy (E_new - E_current)
        delta_E = energy_change(s, i, j)
        # Accept the move with probability given my Boltzmann. If the energy
        # change if lower or the same it will always accept the move
        if np.random.random() < np.exp(-beta*(delta_E)):
            # Flip the spin
            s[i, j] *= -1
            # Get the new energy
            E_current += delta_E
        # Add in the magnetization
        M[m] = magnetization(s)

    plt.figure()
    plt.plot(M)
    plt.xlabel("Moves")
    plt.ylabel("Magnetization")


def simulated_annealing():
    """Use simulated annealing to find the global minimum for the two functions
    below"""

    def f(x):
        return x**2 - np.cos(4*np.pi*x)

    def g(x):
        return np.cos(x) + np.cos(2**0.5*x) + np.cos(3**0.5*x)

    def annealing(f, x0, x_min=-np.inf, x_max=np.inf):
        """Annealing simulation, find the minimum of the function 'f' starting
        at x0 constrained to be between x_min and x_max"""
        # Boltzmann constant, set to 1
        k = 1

        # Temperature parameters
        T_max = 100
        T_min = 1e-4
        tau = 1e3

        # Initial conditions
        t = 0
        xs = [x0]
        x = x0
        T = T_max
        f_prev = f(x0)
        while T > T_min:
            # Update the time and the temperature
            t += 1
            T = T_max*np.exp(-t/tau)
            beta = 1/(k*T)
            delta = np.random.normal(0, 1)
            f_new = f(x + delta)
            df = f_new - f_prev
            if np.random.rand() < np.exp(-beta*df):
                if not(x + delta > x_max or x + delta < x_min):
                    x += delta
                    f_prev = f_new
            xs.append(x)

        plt.figure()
        plt.scatter(np.arange(t+1), xs)
        plt.xlabel("Time")
        plt.ylabel("x")
        print(f"The minimum is {f(x)} found at {x=}")

    # First function
    x_array = np.linspace(-4, 4, 1000)
    y = f(x_array)
    plt.figure()
    plt.plot(x_array, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    annealing(f, 2)

    # Second function
    x_array = np.linspace(0, 50, 1000)
    y = g(x_array)
    plt.figure()
    plt.plot(x_array, y)
    plt.xlabel("x")
    plt.ylabel("g(x)")
    annealing(g, 10, 0, 50)


def dimer_dozen():
    """Use simulated annealing to maxmize the number of dimers put down on a
    square grid"""
    # Size of grid
    L = 50
    # Boltzmann's constant
    k = 1

    # Temperature Parameters
    T_max = 10
    tau = 1e4
    T_min = 1e-4

    # Initial array, it is an LxLx2 array where the last dimension is the
    # coordinates for the second part of the dimer
    A = np.zeros((50, 50, 2), int)
    # Define the empty spot
    zero = np.zeros(2)
    # Initial conditions
    t = 0
    T = T_max
    while T > T_min:
        t += 1
        T = T_max*np.exp(-t/tau)
        beta = 1/(k*T)

        coord = np.random.randint(0, L, 2)
        i, j = coord
        site_1 = A[i, j]
        # Check if site 1 is empty
        if all(site_1 == zero):
            # pick site_2 at random
            # Build up the allowed neighbors
            neighbors = []
            if j > 0:
                neighbors.append(np.array([0, -1]))
            if i > 0:
                neighbors.append(np.array([-1, 0]))
            if j < L - 1:
                neighbors.append(np.array([0, 1]))
            if i < L - 2:
                neighbors.append(np.array([1, 0]))

            # Pick a neighbor at random
            n = np.random.choice(np.arange(len(neighbors)))
            coord2 = coord + neighbors[n]
            i_2, j_2 = coord2
            if all(A[i_2, j_2] == zero):
                # Place the dimer down
                A[i, j] = coord2
                A[i_2, j_2] = coord
        # Non empty, find the neighbor and remove with some probability
        i_2, j_2 = A[i, j]
        df = 1
        if np.random.rand() < np.exp(-beta*df):
            # If accepted remove the dimer
            A[i, j] = zero
            A[i_2, j_2] = zero

    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    # Convert the A array into an array of True and False
    B = np.sum(A, axis=-1)
    # The filled spots will show as white and the empty as black
    plt.imshow(B > 0, cmap="gray", origin="lower")
    ideal = L**2/2
    print("Ratio of covering vs ideal:", np.sum(B > 0)/2/ideal)


def random_ball():
    """Take random samples from the surface of a sphere and plot them on an
    Cartesian grid"""
    # Number of samples
    N = 500
    # Radius of sphere
    r = 1
    # Get random points
    rand = np.random.rand(N)
    phi = 2*np.pi*rand
    rand = np.random.rand(N)
    theta = np.arccos(1 - 2*rand)

    # Convert to x, y, z
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)


def DLA():
    """Simulate a sticky particle moving at random in a box"""
    # Grid Size
    L = 25
    # Initial points
    x0, y0 = (L - 1)//2, (L - 1)//2

    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots()
    point, = ax.plot(x0, y0, "bo")
    ax.set_xlim(0, L - 1)
    ax.set_ylim(0, L - 1)

    # takes data from frame
    def func(frame, xs, ys):
        """Frame gives the information about the moving particle, xs and ys
        gives the positions of the stuck particles"""
        x, y = frame
        point.set_data(xs + [x], ys + [y])
        return point

    def hit_something(x, y, stuck):
        # Check the wall boundaries
        if x == L - 1 or y == L - 1 or x == 0 or y == 0:
            return True
        # Get the x and y values of the stuck particles
        xs = stuck[0]
        ys = stuck[1]
        xys = zip(xs, ys)

        is_stuck = False
        empty = False
        while not is_stuck and not empty:
            # Set the values to be empty once the stuck list is out
            s_x, s_y = next(xys, ("empty", "empty"))

            if x == s_x:
                # The particle is aligned with a stuck particle in the x-axis
                # Check if the particle is next to it along the y-axis
                above = (y - 1 == s_y)
                below = (y + 1 == s_y)
                is_stuck = above or below
            if y == s_y:
                # The particle is aligned with a stuck particle in the y-axis
                # Check if the particle is next to it along the x-axis
                to_the_left = (x - 1 == s_x)
                to_the_right = (x + 1 == s_x)
                is_stuck = to_the_left or to_the_right
            if s_x == "empty":
                empty = True
        return is_stuck

    def move(x, y, stuck):
        if hit_something(x, y, stuck):
            stuck[0].append(x)
            stuck[1].append(y)
            return x0, y0
        r = np.random.choice(["up", "down", "left", "right"])
        if r == "up":
            return x, y + 1
        if r == "down":
            return x, y - 1
        if r == "right":
            return x + 1, y
        if r == "left":
            return x - 1, y

    def run():
        x, y = x0, y0
        while True:
            x, y = move(x, y, stuck)
            yield x, y

    stuck = [[], []]
    ani = FuncAnimation(fig, func, frames=run, fargs=stuck, interval=0)
    return ani
