# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:43:46 2023

@author: almon
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, epsilon_0
from scipy.integrate import dblquad


def exercise_simpsons_rule():
    """"Find the integral of x**4 - 2x + 1 from 0 to 2 with respect to x"""
    def f(x):
        """Function to integrate, takes one argument"""
        return x**4 - 2*x + 1

    def F(a, b):
        """Exact solution to the integral, takes the lower limit 'a' and upper
        limit 'b' as arguments"""
        return b**5/5 - b**2 + b - (a**5/5 - a**2 + a)
    # Number of slices
    N = 10
    # Integral limits
    a = 0
    b = 2
    # Spacing
    h = (b - a)/N

    # Solution using Simpson's rule
    s = f(a) + f(b)
    for k in range(1, N, 2):
        s += 4*f(a + k*h)
        s += 2*f(a + (k+1)*h)
    s -= 2*f(a + (k+1)*h)
    print(1/3*h*s)
    # Actual result
    actual = F(a, b)
    print(actual)

    # Redone but using numpy
    k_o = np.arange(1, N, 2)
    k_e = np.arange(2, N, 2)

    I = 1/3*h*(f(a) + f(b) + 4*sum(f(a + k_o*h)) + 2*sum(f(a + k_e*h)))
    print(I)

    print(f"Fractional error: {(I-actual)/actual}")


# Note the difference in limits
def exercise_error_function():
    """Calculate the function E(x) where E(x) is the integral from 0 to x of
    e to the -t^2"""

    def f(t):
        """Function to integrate, single argument"""
        return np.exp(-t**2)

    def E(x, N=1000):
        """ E(x) is a special function related to the error function. Here it
        is solved by using Simpson's rule. 'x' is an array of values for the
        upper limit of the integration. np.outer takes two 1D arrays and
        returns a 2D array where every value in one array is multiplied by
        every value in the other. See the documentation for more details"""
        a = 0
        b = x
        h = (b - a)/N
        k_o = np.arange(1, N, 2)
        k_e = np.arange(2, N, 2)

        # We only want to sum across the 0 axis which is the axis with
        # differing k values
        I = 1/3*h*(f(a) + f(b) + 4*np.sum(f(a + np.outer(k_o, h)), 0)
                   + 2*np.sum(f(a + np.outer(k_e, h)), 0))
        return I

    x = np.arange(-3, 3, 0.1)
    Es = E(x)
    plt.figure()
    plt.plot(x, Es)
    plt.xlabel("x")
    plt.ylabel("E(x)")


def exercise_trapazoidal_rule_adaptive():
    """Integrate from 0 to 1 sin^2(10x^(1/2)) with respect to x using an
    adaptive trapazoidal approach"""

    def f(x):
        """Integrating funciton"""
        return np.sin(np.sqrt(100*x))**2

    def integrate_f_trap(N):
        """Integration using trapazoidal rule"""
        a = 0
        b = 1
        h = (b - a)/N
        s = 1/2 * f(a) + 1/2 * f(b)
        k = np.arange(1, N)
        s += sum(f(a + k*h))
        return s*h
    # Start with 1 slice and double each time
    N = 1
    I_1 = integrate_f_trap(N)
    N = 2*N
    I_2 = integrate_f_trap(N)
    # The error on trapazoidal integration is like (1/N)^2 so the error
    # between the two integrals goes like 1/3 the difference.
    esp = 1/3*(I_2 - I_1)
    esp_min = 1e-6
    I_old = I_2
    while abs(esp) > esp_min:
        N = 2*N
        I = integrate_f_trap(N)

        esp = 1/3*(I - I_old)
        I_old = I
        print(f"N = {N}, I = {I}, esp = {esp}")


def exercise_romberg():
    """Integrate from 0 to 1 sin^2(10x^(1/2)) with respect to x using the
    Romberg integration technique"""
    def f(x):
        """Integrating funciton"""
        return np.sin(np.sqrt(100*x))**2

    def integrate_f_trap(N):
        """Integration using trapazoidal rule"""
        a = 0
        b = 1
        h = (b - a)/N
        s = 1/2*f(a) + 1/2*f(b)
        k = np.arange(1, N)
        s += sum(f(a + k*h))
        return s*h

    N = 1
    R_1_1 = integrate_f_trap(N)
    N = 2*N
    R_2_1 = integrate_f_trap(N)
    esp_2_1 = 1/(4**1 - 1) * (R_2_1 - R_1_1)
    R_2_2 = R_2_1 + esp_2_1
    esp = esp_2_1
    esp_min = 1e-6
    R_i_m_prev = np.array([R_2_1, R_2_2])
    print("Estimates of Integral, R_i_m")
    print(np.array([R_1_1]))
    print(R_i_m_prev)
    i = 3
    while abs(esp) > esp_min:
        N = 2*N
        R_i_m = np.empty(i)
        R_i_m[0] = integrate_f_trap(N)
        for m in range(1, i):
            R_i_m[m] = R_i_m[m-1] + 1/(4**m - 1) * \
                (R_i_m[m-1] - R_i_m_prev[m-1])
        print(R_i_m)
        esp = 1/(4**m - 1) * (R_i_m[m-1] - R_i_m_prev[m-1])
        R_i_m_prev = R_i_m
        i += 1
    I = R_i_m[i-2]

    print(f"N = {N}, I = {I}, esp = {esp}")


def exercise_simpsons_rule_adaptive():
    """Integrate from 0 to 1 sin^2(10x^(1/2)) with respect to x using the
    adaptive Simpson's approach"""
    def f(x):
        return np.sin(np.sqrt(100*x))**2

    N = 2
    a = 0
    b = 1
    h = (b - a)/N
    k_e = np.arange(2, N, 2)
    S = 1/3*(f(a) + f(b) + 2*sum(f(a+h*k_e)))
    k_o = np.arange(1, N, 2)
    T = 2/3 * sum(f(a + k_o*h))
    I_1 = h*(S + 2*T)

    def integrate_f_simp(N, S_old, T_old):
        h = (b - a)/N
        k_o = np.arange(1, N, 2)
        T = 2/3 * sum(f(a + k_o*h))
        S = S_old + T_old
        return h*(S + 2*T), S, T
    N = 2*N
    I_2, S, T = integrate_f_simp(N, S, T)
    I_old = I_2
    esp = 1/15*(I_2 - I_1)
    esp_min = 1e-6
    while abs(esp) > esp_min:
        N = 2*N
        I, S, T = integrate_f_simp(N, S, T)
        esp = 1/15*(I - I_old)
        I_old = I
        print(f"N = {N}, I = {I}, esp = {esp}")


def exercise_trapazoidal_rule_adaptive_2():
    """Integrate from 0 to 10 sin^2(x)/x^2 with respect to x using an
    adaptive trapazoidal approach. In this approach we split the range we are
    integrating over if the error is too large rather than increaseing the
    slices. Thus parts of the function that are easier to integrate have fewer
    slices than more difficult parts"""
    max_error = 1e-4

    def f(x):
        """Funtion to integral, the so-called sinc function"""
        # This returns a runtime warning.
        return np.where(x == 0, 1, np.sin(x)**2/x**2)

    def integrate(f, N, a, b):
        """Trapazoidal integration"""
        h = (b - a)/N
        s = 1/2*f(a) + 1/2*f(b)
        k = np.arange(1, N)
        s += sum(f(a + k*h))
        return s*h

    a = 0
    b = 10

    def adaptive_integrate(a, b, max_error):
        """Adapt the integration ranges until the results converge"""
        I_1 = integrate(f, 1, a, b)
        I_2 = integrate(f, 2, a, b)
        esp = 1/3*(I_2 - I_1)
        #delta = esp/(b - a)
        if abs(esp) > max_error:
            I_a_1 = adaptive_integrate(a, (a + b)/2, max_error)
            I_a_2 = adaptive_integrate((a + b)/2, b, max_error)
            return I_a_1 + I_a_2
        else:
            return I_2
    I = adaptive_integrate(a, b, max_error)
    print(I)


def exercise_find_electric_field_points():
    """Find the electric field from two point charges given the potential"""
    # add a find the gradiant of 2xy task
    def potential(q, r):
        k = 1/(4*pi*epsilon_0)
        return k*q/r

    q_1 = 1  # coloumb
    q_2 = -1  # coloumb
    sep = 10e-2  # meters
    n = 50  # number of points

    # Set up our grid
    x = np.linspace(-0.25, 0.25, n)
    y = np.linspace(-0.25, 0.25, n)

    X, Y = np.meshgrid(x, y)

    # Find the potential everywhere
    r_1 = ((sep/2 - X)**2 + Y**2)**(0.5)
    r_2 = ((-sep/2 - X)**2 + Y**2)**(0.5)
    p_1 = potential(q_1, r_1)
    p_2 = potential(q_2, r_2)
    p = p_1 + p_2

    # Plot the potential with grayscale
    plt.figure(figsize=(8, 8))
    plt.imshow(p, cmap='gray')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("V for points")
    plt.tight_layout()

    # Plot the potential with contours
    plt.figure(figsize=(8, 8))
    levels = np.linspace(p.min(), p.max(), n)
    plt.contour(X, Y, p, levels=levels)
    plt.xlabel("x")
    plt.ylabel("y")

    # When we differentiate to get the electric fields we lose a data point
    Ex = np.zeros((n - 1, n - 1), dtype=float)
    Ey = np.zeros((n - 1, n - 1), dtype=float)

    # Smallest possible value for h is the difference between two adjacent
    # points. It is the same for x as well as y.
    h = abs(x[1] - x[0])
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            Ey[i, j] = -(p[i + 1, j] - p[i, j])/h
            Ex[i, j] = -(p[i, j + 1] - p[i, j])/h

    E_mag = (Ex**2 + Ey**2)**(0.5)

    # Clip off the ends of X and Y to match the shape of Ex, Ey
    # Scale the fields for ease of visualization
    plt.quiver(X[:-1, :-1], Y[:-1, :-1], Ex/E_mag, Ey/E_mag)
    plt.title("E and V for points")
    plt.tight_layout()


def exercise_find_electric_field_plain():
    L = 10e-2  # m
    q0 = 100e-4  # coloumb/m^2

    def sigma(x, y):
        """Charge density at the location x,y"""
        return q0*np.sin(2*pi*x/L)*np.sin(2*pi*y/L)

    def dV(y, x, x0, y0):
        """One bit of potential at local x0, y0 due to charge at x, y"""
        k = 1/(4*pi*epsilon_0)
        return k * sigma(x, y) / ((x-x0)**2 + (y-y0)**2)**0.5

    # Set up our grid
    x = np.linspace(-L/2, L/2, 50)
    y = np.linspace(-L/2, L/2, 50)

    X, Y = np.meshgrid(x, y)
    a = -L/2
    b = L/2
    xs = np.arange(-L, L, 1e-2)
    ys = np.arange(-L, L, 1e-2)
    V = np.zeros((len(xs), len(ys)), dtype=float)

    for x_index, x in enumerate(xs):
        for y_index, y in enumerate(ys):
            V[x_index, y_index] = dblquad(dV, a, b, a, b, args=(x, y),
                                          epsabs=1e-4)[0]
    plt.figure(figsize=(8, 8))
    plt.imshow(V, cmap='gray')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("V for plain")
    plt.tight_layout()

    plt.figure(figsize=(8, 8))
    levels = np.linspace(V.min(), V.max(), 50)
    plt.contour(xs, ys, V, levels=levels)

    Ey = -np.gradient(V)[0]
    Ex = -np.gradient(V)[1]
    E_mag = (Ex**2 + Ey**2)**(0.5)
    plt.quiver(xs, ys, Ex/E_mag, Ey/E_mag)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("E and V for plain")
    plt.tight_layout()
