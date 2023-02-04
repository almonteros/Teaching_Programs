# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:43:32 2023

@author: almon
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, e, m_e, hbar
from scipy.integrate import simpson
import matplotlib.animation as animation
from time import perf_counter


def RK(f, r0, ts):
    """
    Fourth order Runge-Katta method for a system with n equations of mth order
    Parameters
    ----------
    f : Function
        A function that takes an nXm array where n is the number of equations
        and m is the order of the differential equation. It should return
        d^m x/dt^m for each row
    r0 : nXm array
        The initial conditions where the rows are the varius equations and the
        columns are the different orders of the equation.
    ts : Array or list
         The values of time that will be used. This should be an evenly spaced
         array of times

    Returns
    -------
    rs : Array of NXnXm
        An array of the solution to the differential equation for each time
        step

    """
    N = len(ts)
    h = abs(ts[1] - ts[0])
    rs = np.empty([N] + list(r0.shape), float)
    rs[0] = r0

    for i in range(N - 1):
        r = rs[i]
        t = ts[i]
        k1 = h * f(r, t)
        k2 = h*f(r + 1/2*k1, t + 1/2*h)
        k3 = h*f(r + 1/2*k2, t + 1/2*h)
        k4 = h*f(r + k3, t + h)
        r = r + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        rs[i+1] = r
    return rs


def RK_assist(f, r, t, h):
    """
    Impliments the Runge-Kutta for a single step h
    ----------
    f : Function
        A function that takes an nXm array where n is the number of equations
        and m is the order of the differential equation.
    r : nXm array
        The initial conditions where the rows are the varius equations and the
        columns are the different orders of the equation.
    t : Real number
        Point in time
    h : Real number
        Time step

    Returns
    -------
    TYPE
        Value of the next time step

    """
    k1 = h * f(r, t)
    k2 = h*f(r + 1/2*k1, t + 1/2*h)
    k3 = h*f(r + 1/2*k2, t + 1/2*h)
    k4 = h*f(r + k3, t + h)
    return r + 1/6*(k1 + 2*k2 + 2*k3 + k4)


def RK_adapt(f, r0, ti, tf, h, delta):
    """
    Adaptive Runge-Kutta method. Changes the step size based on the calculated
    error. If it is small the step size is increased, if it is large the step
    size is decreased
    ----------
    f : Function
        A function that takes an nXm array where n is the number of equations
        and m is the order of the differential equation. It should return
        d^m x/dt^m for each row
    r0 : nXm array
        The initial conditions where the rows are the varius equations and the
        columns are the different orders of the equation
    ti : Real number
        Initial time
    tf : Real number
        Final time.
    h : Real number
        Step size.
    delta : Real number
        Target error per time step

    Returns
    -------
    Array of NXnXm
        An array of the solution with N time steps

    """
    max_rho = 16
    r = r0
    rs = [r0]
    t = ti
    ts = [ti]
    while t < tf:
        # Two single steps
        r1 = RK_assist(f, r, t, h)
        r2 = RK_assist(f, r1, t + h, h)
        # One souble step
        r3 = RK_assist(f, r, t, 2*h)
        # Error
        error = np.sum(((r3 - r2)/30)**2)**(0.5)
        # rho is the ratio of the target error to actual error we need a max
        # rho to avoid giant steps
        if error == 0:
            rho = max_rho
        else:
            rho = h*delta / error

        if rho > 1:
            # The  error is small so we can use the values and adjust the step
            # size later
            t += 2*h
            rs.append(r1)
            rs.append(r2)
            r = r2
            ts.append(t + h)
            ts.append(t + 2*h)

        if rho > max_rho:
            # We don't want to take giant steps
            h *= 2
        else:
            # Adjust the step size to be bigger, rho > 1, or smaller, rho < 1
            h *= rho**(1/4)
    return np.array(rs)


def leapfrog(f, r0, ts):
    """
    Impliments the leap frog method for solving differential equation. It is
    typically better when timer-reversal symmetry (energy conservation) is a
    concern and is a better leaping off point for Richardson extrapolation

    Parameters
    ----------
    f : Function
        A function, dr/dt = f(r, t), that takes an nXm array where n is the
        number of equations and m is the order of the differential equation.
    r0 : nXm array
        The initial conditions where the rows are the varius equations and the
        columns are the different orders of the equation.
    ts : Array or list
         The values of time that will be used. This should be an evenly spaced
         array of times

    Returns
    -------
    rs : Array of NXnXm
        An array of the solution to the differential equation for each time
        step

    """
    N = len(ts)
    h = abs(ts[1] - ts[0])
    rs = np.empty([N] + list(r0.shape), float)
    rs[0] = r0
    x_half = r0 + 1/2*h*f(r0, ts[0])
    for i in range(N - 1):
        r = rs[i]
        t = ts[i]
        rs[i + 1] = r + h*f(x_half, t + 1/2*h)
        x_half = x_half + h*f(rs[i + 1], ts[i + 1])
    return rs


def verlot(f, r0, t):
    """


    Parameters
    ----------
    f : Function
        A function, dr/dt = f(r, t), that takes an nXm array where n is the
        number of equations and m is the order of the differential equation.
        d^m x/dt^m for each row
    r0 : nX2 array
        The initial conditions where the rows are the varius equations and the
        columns are the initial position and velocity
    t : Array or list
         The values of time that will be used. This should be an evenly spaced
         array of times

    Returns
    -------
        2 X N X n array
        An array of the position and velocity values for all N values of time
        and n umber of equations

    """
    N = len(t)
    h = abs(t[1] - t[0])
    x0 = r0[:, 0]
    v0 = r0[:, 1]
    x = np.empty([N] + [n for n in x0.shape], float)
    v = np.empty([N] + [n for n in v0.shape], float)
    x[0] = x0
    v[0] = v0
    v_half = v0 + 1/2*h*f(x0, t[0])
    for i in range(N - 1):
        x[i + 1] = x[i] + h*v_half
        k = h*f(x[i + 1], t[i + 1])
        v[i + 1] = v_half + 1/2*k
        v_half = v_half + k
    return np.array([x, v])


def modified_midpoint(f, r0, t0, H, n):
    """
    Impliments the modified midpoint algorithm.

    Parameters
    ----------
    f : Function
        A function, dr/dt = f(r, t), that takes an nXm array where n is the
        number of equations and m is the order of the differential equation.
    r0 : nXm array
        The initial conditions where the rows are the varius equations and the
        columns are the different orders of the equation.
    t0 : Real number
        Initial time
    H : Real number
        Time step
    n : Integer
        Number of steps

    Returns
    -------
    r : nXm array
        Solution

    """
    h = H/n

    r_half_step = r0 + 1/2*h*f(r0, t0)
    r_full_step = r0 + h*f(r_half_step, t0)
    r = [r_full_step]
    for i in range(n - 1):
        r_half_step = r_half_step + h*f(r_full_step, t0 + i*h)
        r_full_step = r_full_step + h*f(r_half_step, t0 + (i + 1/2)*h)
        r.append(r_full_step)
    r = 1/2*(r_full_step + r_half_step + 1/2*h*f(r_full_step, t0 + H))

    # Personal preference
    # r_full_step = r0 + 0  # So r0 and r_full_step are not the same reference
    # r_half_step = r0 + 1/2*h*f(r0, t0)
    # r = [r_full_step]
    # for i in range(n - 1):
    #     r_half_step = r_full_step + h/2*f(r_full_step, t0 + i*h)
    #     r_full_step = r_full_step + h*f(r_half_step, t0 + (i + 1/2)*h)
    #     r.append(r_full_step)
    # r = 1/2*(r_full_step + r_half_step + 1/2*h*f(r_full_step, t0 + H))
    return r


def Bulirsh_Stoer_step(f, r0, H, t0, delta, n_max=50):
    """
    Impliements one large step of the Bulirish-Stoer solver for ordinary
    differential equations

    Parameters
    ----------
    f : Function
        A function, dr/dt = f(r, t), that takes an nXm array where n is the
        number of equations and m is the order of the differential equation.
    r0 :nXm array
        The initial conditions where the rows are the varius equations and the
        columns are the different orders of the equation.
    H : Real number
        Step for the Bulirish-Stoer method
    t0 : Real number
        Initial time
    delta : Real number
        Maximum allowed error per step
    n_max : Integer
        Maximum number of steps allowed. The default is 50

    Returns
    -------
    The solution at time t0 + H

    """
    max_error = H*delta
    n = 1

    R_11 = modified_midpoint(f, r0, t0, H, n)
    R_prev_nm = np.array([R_11])

    n = 2
    shape = [n] + list(r0.shape)
    R_nm = np.zeros(shape, float)

    R_21 = modified_midpoint(f, r0, t0, H, n)
    error = (R_21 - R_11)/3
    R_22 = R_21 + error

    R_nm[0] = R_21
    R_nm[1] = R_22
    while np.max(abs(error)) > max_error and n < n_max:
        n += 1
        R_prev_nm = R_nm
        shape = [n] + [ax for ax in r0.shape]
        R_nm = np.zeros(shape, float)
        R_n1 = modified_midpoint(f, r0, t0, H, n)
        R_nm[0] = R_n1

        # Find the error for R_nm
        for m in range(1, n):
            error = (R_nm[m-1] - R_prev_nm[m-1])/((n/(n - 1))**(2*m) - 1)
            R_nm[m] = R_nm[m-1] + error
    if n >= n_max:
        print(f"The max number of steps ({n_max}) has been reached.")
    return R_nm[-1], n


def Bulirsh_Stoer(f, r0, t, delta=1e-3, n_max=50):
    """
    Impliements the Bulirish-Stoer solver for ordinary differential equations

    Parameters
    ----------
    f : Function
        A function, dr/dt = f(r, t), that takes an nXm array where n is the
        number of equations and m is the order of the differential equation.
    r0 :nXm array
        The initial conditions where the rows are the varius equations and the
        columns are the different orders of the equation.
    t : Array or list
         The values of time that will be used. This should be an evenly spaced
         array of times
    delta : Real number
        Maximum allowed error per step. The default is 1e-3
    n_max : Integer
        Maximum number of steps allowed. The default is 50

    Returns
    -------
    r : Array of NXnXm
        An array of the solution to the differential equation for each time
        step

    """
    H = t[1] - t[0]
    N = len(t)
    shape = [N] + list(r0.shape)
    r = np.empty(shape)
    r[0] = r0
    for i in range(1, N):
        r[i] = Bulirsh_Stoer_step(f, r[i - 1], H, t[i - 1], delta, n_max)[0]
    return r


def exercise_circuit_RK():
    """Given a low pass filter and an square-wave input voltgage find the
    voltage out"""
    RC = 1

    def V_in(t):
        """Voltage in"""
        t_floor = np.floor(2*t)
        if t_floor % 2 == 0:
            return 1
        return -1

    def f(V_out, y):
        """dV_dt = f"""
        return 1/RC * (V_in(t) - V_out)

    # Parameters for the time
    a = 0
    b = 10
    N = 1000
    h = (b - a)/N
    ts = np.linspace(a, b, N)
    V_out_array = np.empty(N, float)
    # Initial condition for the voltage
    V_out_array[0] = 0
    for i in range(N - 1):
        t = ts[i]
        V_out = V_out_array[i]
        k1 = h * f(V_out, t)
        k2 = h*f(V_out + 1/2*k1, t + 1/2*h)
        k3 = h*f(V_out + 1/2*k2, t + 1/2*h)
        k4 = h*f(V_out + k3, t + h)
        V_out += 1/6*(k1 + 2*k2 + 2*k3 + k4)
        V_out_array[i+1] = V_out
    plt.figure()
    plt.plot(ts, V_out_array)
    plt.xlabel("t")
    plt.ylabel("Voltage")


def exercise_3variable_RK():
    """Solve the Lorentz Equations"""

    # Parameters
    sigma = 10
    sigma_r = 28
    sigma_b = 8/3
    # Initial conditions
    x_initial = 0
    y_initial = 1
    z_initial = 0

    def f(r, t):
        x = r[0]
        y = r[1]
        z = r[2]
        fx = sigma * (y - x)
        fy = sigma_r*x - y - x*z
        fz = x*y - sigma_b*z
        return np.array([fx, fy, fz])

    # Set up the time steps
    a = 0
    b = 50
    N = 10000
    h = (b - a)/N
    ts = np.linspace(a, b, N)

    # Put all of the inital conditions into one structure
    rs = np.empty((N, 3), float)
    rs[0] = x_initial, y_initial, z_initial

    for i in range(N - 1):
        r = rs[i]
        t = ts[i]
        k1 = h * f(r, t)
        k2 = h*f(r + 1/2*k1, t + 1/2*h)
        k3 = h*f(r + 1/2*k2, t + 1/2*h)
        k4 = h*f(r + k3, t + h)
        r += 1/6*(k1 + 2*k2 + 2*k3 + k4)
        rs[i+1] = r

    plt.figure()
    plt.plot(ts, rs[:, 0])
    plt.xlabel("t")
    plt.ylabel("x")

    plt.figure()
    plt.plot(ts, rs[:, 1])
    plt.xlabel("t")
    plt.ylabel("y")

    plt.figure()
    plt.plot(ts, rs[:, 2])
    plt.xlabel("t")
    plt.ylabel("z")

    plt.figure()
    plt.plot(rs[:, 0], rs[:, 2])
    plt.xlabel("x")
    plt.ylabel("z")


def exersice_2nd_order_RK():
    """Model a small sphere orbiting a heavy, thin rod. Consider the rod
    to be stationary"""
    # Parameters
    G = 1
    # Mass of the rod (A.U.)
    M = 10
    # Length of the rod (A.U.)
    L = 2
    # Initial conditions
    x0 = 1
    y0 = 0
    vx0 = 0
    vy0 = 1
    # The columns of r0 have the different equations and the rows have the
    # different order
    r0 = np.array([[x0, vx0], [y0, vy0]])
    # Time
    t = np.linspace(0, 10, 1000)

    def f(r, t):
        x = r[0, 0]
        y = r[1, 0]
        vx = r[0, 1]
        vy = r[1, 1]
        r2 = x**2 + y**2
        f_x = vx
        f_y = vy
        denom = r2*(r2 + L**2/4)**0.5
        f_vx = -G*M*x/denom
        f_vy = -G*M*y/denom
        return np.array([[f_x, f_vx], [f_y, f_vy]])

    r = RK(f, r0, t)
    x = r[:, 0, 0]
    # vx = r[:, 0, 1]
    y = r[:, 1, 0]
    # vy = r[:, 1, 1]

    plt.figure()
    plt.plot(t, x)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.figure()
    plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel("y")

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")


def exercise_springs_RK():
    """Vibrations of a linear chain of N identical masses on springs driven
    with a cosine force"""
    # Parameters all in arbitrary units
    m = 1  # Mass
    k = 6  # Spring constant
    w = 2  # Driving frequency
    N = 5  # Number of masses
    # Everything starts at rest, no displacement
    r0 = np.zeros((N, 2))
    t = np.linspace(0, 10, 1000)

    def F(w, t):
        """Driving force"""
        force = np.zeros(N)
        force[0] = np.cos(w*t)
        return force

    def f(r, t):
        eta = r[:, 0]
        eta_dot = r[:, 1]
        f_eta = eta_dot
        f_eta_dot = k/m * (np.roll(eta, -1) +
                           np.roll(eta, 1) - 2*eta) + F(w, t)
        # Fix the end points
        f_eta_dot[0] -= k/m * (eta[-1] - eta[0])
        f_eta_dot[-1] -= k/m * (eta[0] - eta[-1])
        return np.column_stack((f_eta, f_eta_dot))

    r = RK(f, r0, t)
    eta = r[:, :, 0]
    v = r[:, :, 1]

    plt.figure()
    plt.plot(t, eta)
    plt.xlabel("t")
    plt.ylabel("Position of each mass")
    plt.figure()
    plt.plot(t, v)
    plt.xlabel("t")
    plt.ylabel("Velocity of each mass")


def exercise_adaptive_RK():
    """Model a comet going around the sun. Compare the regular RK approach to
    the adaptive approach"""
    # Parameters
    M = 1.989e30  # Mass of sun in kg
    x0 = 4e12
    y0 = 0
    vx0 = 0
    vy0 = 500
    r0 = np.array([[x0, vx0], [y0, vy0]])
    year = 360*24*60*60
    T = 50 * year
    t = np.linspace(0, 2*T, 200000)

    def f(r, t):
        x = r[0, 0]
        y = r[1, 0]
        vx = r[0, 1]
        vy = r[1, 1]
        r3 = (x**2 + y**2)**(3/2)
        fx = vx
        fy = vy
        fvx = -G*M*x/r3
        fvy = -G*M*y/r3
        return np.array([[fx, fvx], [fy, fvy]])

    t_i = perf_counter()
    r = RK(f, r0, t)
    t_f = perf_counter()
    x = r[:, 0, 0]
    y = r[:, 1, 0]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.xlabel("y")
    plt.title(f"Regular Runge-Kutta, time: {t_f - t_i :.4f}")

    # Adaptive approach. A lot faster!
    delta = 1e3/year
    h = 5000
    t_i = perf_counter()
    r = RK_adapt(f, r0, 0, 2*T, h, delta)
    t_f = perf_counter()
    x = r[:, 0, 0]
    y = r[:, 1, 0]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.xlabel("y")
    plt.title(f"Adaptive Runge-Kutta, time: {t_f - t_i :.4f}")


def exercise_leapfrog():
    """Solve the differential equation d^2x/dt^2 - (dx/dt)^2 + x + 5 = 0 using
    the leapfrog method"""
    # Initial conditions, x(0) = 1, dx(0)/dt = 0
    r0 = np.array([1., 0.])
    # Step size
    h = 1e-4
    # Time range
    t = np.arange(0, 50, h)

    def f(r, t):
        x = r[0]
        v = r[1]
        f_x = v
        f_v = v**2 - x - 5
        return np.array([f_x, f_v])

    r = leapfrog(f, r0, t)
    plt.figure()
    plt.plot(t, r[:, 0])
    plt.xlabel("t")
    plt.ylabel("x")


def exercise_verlet():
    """Using the Verlet method calculate the orbit of the earth"""
    # Parameters
    M = 1.989e30  # Mass of sun in kg
    m = 5.9722e24  # Mass of earth in kg
    h = 60*60  # One hour
    # Initial conditions, The top row is x0, vx0 and the bottom row is y0, vy0
    r0 = np.array([[1.471e11, 0], [0, 3.0287e4]])
    year = 365*24*60*60
    t = np.arange(0, 3*year, h)

    def f(r, t):
        x = r[0]
        y = r[1]
        r3 = (x**2 + y**2)**(3/2)
        fvx = -G*M*x/r3
        fvy = -G*M*y/r3
        return np.array([fvx, fvy])

    r = verlot(f, r0, t)
    x = r[0, :, 0]
    y = r[0, :, 1]

    vx = r[1, :, 0]
    vy = r[1, :, 1]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")

    PE = -G*M*m/(x**2 + y**2)**0.5
    KE = 1/2*m*(vx**2 + vy**2)

    plt.figure()
    plt.plot(t, PE, label="PE")
    plt.plot(t, KE, label="KE")
    plt.plot(t, KE + PE, label="Total Energy")
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.legend()

    # Notice that energy is conserved long term
    plt.figure()
    plt.plot(t, KE + PE)
    plt.xlabel("t")
    plt.ylabel("Total Energy")


def exercise_Bulrish_Stoer():
    """Using the Bulrish-Stoer method plot the orbits of Earth and Pluto"""
    # Parameters
    M = 1.989e30  # Mass of sun in kg
    # m = 5.9722e24  # Mass of earth in kg, unneed but nice to know
    H = 24*7*60*60  # Time steps
    # Initial conditions [[x0, vx0], [y0, vy0]] in meters and meters per second
    r0_earth = np.array([[1.4710e11, 0], [0, 3.0287e4]])
    year = 365*24*60*60
    # Accuracy per step
    delta = 1/year

    def f(r, t):
        x = r[0, 0]
        y = r[1, 0]
        vx = r[0, 1]
        vy = r[1, 1]
        r3 = (x**2 + y**2)**(3/2)
        fx = vx
        fy = vy
        fvx = -G*M*x/r3
        fvy = -G*M*y/r3
        return np.array([[fx, fvx], [fy, fvy]])

    t = np.arange(0, 250*year, H)
    r = Bulirsh_Stoer(f, r0_earth, t, delta)
    x = r[:, 0, 0]
    y = r[:, 1, 0]
    plt.figure()
    plt.plot(x, y, label="Earth")
    plt.xlabel("x")
    plt.ylabel("y")

    # Initial conditions [[x0, vx0], [y0, vy0]]
    r0_pluto = np.array([[4.4368e12, 0], [0, 6.1218e3]])
    r = Bulirsh_Stoer(f, r0_pluto, t, delta)
    x = r[:, 0, 0]
    y = r[:, 1, 0]
    plt.plot(x, y, label="Pluto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


def exercise_find_eigenvalues():
    """Consider the 1D, time independent Schrodinger equation with a quadratic
    potential. Find the energy states by solving the wavefunction for
    differing values of E. Use the secant method to minimize E"""

    def RK_E(f, r0, ts, E):
        """Runge-Kutta with an extra parameter"""
        N = len(ts)
        h = abs(ts[1] - ts[0])
        rs = np.empty([N] + [n for n in r0.shape], float)
        rs[0] = r0

        for i in range(N - 1):
            r = rs[i]
            t = ts[i]
            k1 = h * f(r, t, E)
            k2 = h*f(r + 1/2*k1, t + 1/2*h, E)
            k3 = h*f(r + 1/2*k2, t + 1/2*h, E)
            k4 = h*f(r + k3, t + h, E)
            r = r + 1/6*(k1 + 2*k2 + 2*k3 + k4)
            rs[i+1] = r
        return rs

    def V(x):
        return V0*(x/a)**2

    def f(r, x, E):
        phi = r[0]
        phi_dot = r[1]
        fphi = phi_dot
        fphi_dot = 2*m_e/hbar**2*(V(x) - E)*phi
        return np.array([fphi, fphi_dot])

    # Parameters
    V0 = 50*e  # Joules
    a = 1e-11
    # Boundary condition on the left side [psi, dpsi/dx]
    psi0 = np.array([0.0, 1.0])
    numPoints = 1000

    # Including the end point gives worse results. I think this might be
    # because the derivative at the end point DNE
    x = np.linspace(-10*a, 10*a, numPoints, endpoint=False)
    x_graph = np.linspace(-5*a, 5*a, numPoints, endpoint=False)

    # GROUND STATE
    # Pick window for E
    E1 = 100*e
    E2 = 200*e
    # Solve for the wavefunction, we only want the last value
    psi_new = RK_E(f, psi0, x, E1)[-1, 0]
    # Use the secant method to continually try to find the minimum E
    # We want the value E such that psi is 0 at the boundary
    max_error = e/10000
    while abs(E1 - E2) > max_error:
        psi_old, psi_new = psi_new, RK_E(f, psi0, x, E2)[-1, 0]
        E1, E2 = E2, E2 - psi_new * (E2 - E1)/(psi_new - psi_old)
    E_0 = E2
    print(f"The ground state energy is {E_0/e} eV")

    psi = RK_E(f, psi0, x_graph, E_0)[:, 0]
    A = simpson(abs(psi)**2, x_graph)

    plt.figure(figsize=(12, 8))
    plt.plot(x, psi/A**0.5)
    plt.xlabel("x")
    plt.ylabel(r"$|\psi_0|^2$")

    # FIRST EXCITED STATE
    # Pick window for E
    E1 = 200*e
    E2 = 400*e
    psi_new = RK_E(f, psi0, x, E1)[-1, 0]
    while abs(E1 - E2) > max_error:
        psi_old, psi_new = psi_new, RK_E(f, psi0, x, E2)[-1, 0]
        E1, E2 = E2, E2 - psi_new * (E2 - E1)/(psi_new - psi_old)
    E_1 = E2
    print(f"The first excited state energy is {E_1/e} eV")

    x = np.linspace(-5*a, 5*a, numPoints, endpoint=False)
    psi = RK_E(f, psi0, x_graph, E_1)[:, 0]
    A = simpson(abs(psi)**2, x_graph)

    plt.figure(figsize=(12, 8))
    plt.plot(x, psi/A**0.5)
    plt.xlabel("x")
    plt.ylabel(r"$|\psi_1|^2$")

    # SECOND EXCITED STATE
    # Pick window for E
    E1 = 500*e
    E2 = 700*e
    psi_new = RK_E(f, psi0, x, E1)[-1, 0]
    while abs(E1 - E2) > max_error:
        psi_old, psi_new = psi_new, RK_E(f, psi0, x, E2)[-1, 0]
        E1, E2 = E2, E2 - psi_new * (E2 - E1)/(psi_new - psi_old)
    E_2 = E2
    print(f"The second excited state energy is {E_2/e} eV")

    x = np.linspace(-5*a, 5*a, numPoints, endpoint=False)
    psi = RK_E(f, psi0, x_graph, E_2)[:, 0]
    A = simpson(abs(psi)**2, x_graph)

    plt.figure(figsize=(12, 8))
    plt.plot(x, psi/A**0.5)
    plt.xlabel("x")
    plt.ylabel(r"$|\psi_2|^2$")


def exercise_double_pendulum():
    """Given a double pendulum with equal mass and length graph the motion and
    create a simulation"""
    # Parameters
    num_points = 1000
    t = np.linspace(0, 100, num_points)
    g = 9.81  # m/s
    l = 0.4  # meters
    # Initial conditions
    theta_1_initial = np.pi/2
    theta_2_initial = np.pi/2
    r0 = np.array([[theta_1_initial, 0], [theta_2_initial, 0]])

    def f(r, t):
        th_1 = r[0, 0]
        w_1 = r[0, 1]
        th_2 = r[1, 0]
        w_2 = r[1, 1]

        f_th_1 = w_1
        f_th_2 = w_2

        f_w_1_num = w_1**2*np.sin(2*th_1 - 2*th_2)\
            + 2*w_2**2*np.sin(th_1 - th_2)\
            + g/l*(np.sin(th_1 - 2*th_2) + 3*np.sin(th_1))
        f_w_1_denom = 3 - np.cos(2*th_1 - 2*th_2)
        f_w_1 = -f_w_1_num/f_w_1_denom

        f_w_2_num = 4*w_1**2*np.sin(th_1 - th_2)\
            + w_2**2*np.sin(2*th_1 - 2*th_2)\
            + 2*g/l*(np.sin(2*th_1 - th_2) - np.sin(th_2))
        f_w_2_denom = 3 - np.cos(2*th_1 - 2*th_2)
        f_w_2 = f_w_2_num/f_w_2_denom
        return np.array([[f_th_1, f_w_1], [f_th_2, f_w_2]])

    r = RK(f, r0, t)

    th_1, w_1 = r[:, 0, 0], r[:, 0, 1]
    th_2, w_2 = r[:, 1, 0], r[:, 1, 1]

    # Put into x, y coordinates
    x1 = l*np.sin(th_1)
    y1 = -l*np.cos(th_1)
    x2 = x1 + l*np.sin(th_2)
    y2 = y1 - l*np.cos(th_2)

    plt.figure()
    plt.scatter(x1, y1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$y_1$")

    plt.figure()
    plt.scatter(x2, y2)
    plt.xlabel(r"$x_2$")
    plt.ylabel(r"$y_2$")

    E = l**2*(w_1**2 + 1/2*w_2**2 + w_1*w_2*np.cos(th_1 - th_2))\
        - g*l*(2*np.cos(th_1 + np.cos(th_2)))

    plt.figure()
    plt.plot(t, E)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E$")

    # Set up the animation

    # Coordinates of the pivot point
    x_pivot = 0
    y_pivot = 0
    # Set up plots
    fig, ax = plt.subplots()
    # ims will hold the artists to be animated
    ims = []
    for i in range(num_points):
        # Plot the pivot
        im = ax.plot(x_pivot, y_pivot, animated=True,
                     marker="o", color="black")
        # Add the time on to the graph
        im += [ax.text(-0.4, 0.05, f"t = {t[i]:.4f}", animated=True)]

        # Plot mass 1
        x = x1[i]
        y = y1[i]
        im += ax.plot(x, y, animated=True, marker="o", color="black")

        # Plot the line from pivot to mass 1
        xx0 = [x_pivot, x]
        yy0 = [y_pivot, y]
        im += ax.plot(xx0, yy0, animated=True, color="black")

        # Plot mass 2
        x = x2[i]
        y = y2[i]
        im += ax.plot(x, y, animated=True, marker="o", color="red")

        # Plot the line from mass 1 to mass 2
        xx = [x1[i], x]
        yy = [y1[i], y]
        im += ax.plot(xx, yy, animated=True, color="black")
        ims.append(im)

    # Create the animation and return it
    ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True)
    return ani
