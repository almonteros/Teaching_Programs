# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:57:51 2023

@author: almon
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, hbar, k
from scipy.integrate import quad


def exercise_relxation():
    """x = 1 - e^(-cx) where c is a constant"""
    c = 2
    max_error = 1e-6

    def f(x):
        return 1 - np.exp(-c*x)

    def f_p(x):
        """Derivative"""
        return c*np.exp(-c*x)

    def find_fixed(c, print_iteration=False):
        """Find the fixed point using the relaxation method"""
        error = 1
        x_new = 1
        iteration = 0
        while error > max_error:
            x_new, x_old = f(x_new), x_new
            error = abs((x_old - x_new)/(1-1/f_p(x_old)))
            iteration += 1
        if print_iteration:
            print(f"The number of itereations is {iteration}")
        return x_new

    print(f"Solution for {c=} is {find_fixed(c, True)}")

    # Now examine the phase transition
    cs = np.arange(0.01, 3, 0.01)
    xs = []
    for c in cs:
        x = find_fixed(c)
        xs.append(x)
    plt.figure(figsize=(12, 8))
    plt.plot(cs, xs)
    plt.xlabel("c")
    plt.ylabel("x_fixed")
    plt.tight_layout()


def exercise_overrelxation():
    c = 2
    w = 0.5  # Acts like a momentum term
    max_error = 1e-6

    def f(x):
        return 1 - np.exp(-c*x)

    def f_p(x):
        return c*np.exp(-c*x)

    def find_fixed(c, w=0.5, print_iteration=False):
        error = 1
        x_new = 1
        iteration = 0
        while error > max_error:
            x_new, x_old = (1 + w)*f(x_new) - w*x_new, x_new
            # Note the different error equation
            error = abs((x_old - x_new)/(1-1/((1 + w)*f_p(x_old) - w)))
            iteration += 1
        if print_iteration:
            print(f"The number of itereations is {iteration}")
        return x_new

    print(f"Solution for {c=} is {find_fixed(c, print_iteration=True)}")

    cs = np.arange(0.01, 3, 0.01)
    xs = []
    for c in cs:
        x = find_fixed(c)
        xs.append(x)
    plt.figure(figsize=(12, 8))
    plt.plot(cs, xs)
    plt.xlabel("c")
    plt.ylabel("x_fixed")


def exercise_two_variable_relaxation():
    """Given the glycolysis equations dx/dt = -x +ay +x^2y and 
    dy/dt = b -ay -x^2y find the steady state solution for x and y for a = 1
    b = 2"""
    a = 1
    b = 2
    max_error = 1e-6

    def fx(x, y):
        # This result won't converge, solve for x with the other equation
        # return a*y + x**2*y
        return (b/y - a)**0.5

    def fy(x, y):
        # This result won't converge, solve for x with the other equation
        # return y = b/(a + x**2)
        return x/(a+x**2)

    def find_fixed(print_iteration=False):
        error_x = 1
        error_y = 1
        x_new = 1
        y_new = 1.1
        iteration = 0
        while error_x > max_error or error_y > max_error:
            x_new, x_old, y_new, y_old = fx(
                x_new, y_new), x_new, fy(x_new, y_new), y_new

            error_x = abs(x_new - x_old)
            error_y = abs(y_new - y_old)
            iteration += 1
        if print_iteration:
            print(iteration)
        return x_new, y_new
    x, y = find_fixed()
    print(f"The realxation method for x, y gives {x}, {y}")
    # Analytic solution
    x_real = b
    y_real = b/(a + b**2)
    print(f"The analytic solution for x, y is {x_real}, {y_real}")


def exercise_binary_search():
    """Find the solution to the equation 5e^(-x) + x - 5 = 0"""
    max_error = 1e-6

    def f(x):
        return 5*np.exp(-x) + x - 5

    # Examine the plot to get an idea where the function is zero
    x = np.linspace(0, 10)
    plt.figure()
    plt.plot(x, f(x))

    # One solution is x = 0 which is easily verfied. The other is around 5
    # From this pick points around the solution
    x1 = 3
    x2 = 8
    # Find the values at those points
    f1 = f(x1)
    f2 = f(x2)
    # Find error
    error = abs(x2 - x1)
    while error > max_error:
        error = abs(x2 - x1)
        # Find the midpoint and value at the midpoint
        x_new = 1/2*(x1 + x2)
        f_new = f(x_new)
        # If the midpoint is the same sign as the value on the left
        # then we shift the value on the left to the midpoint and visa vera in
        # the other case
        if np.sign(f1) == np.sign(f_new):
            x1 = x_new
            f1 = f_new
        else:
            x2 = x_new
            f2 = f_new
    print(f" The solution is {x_new}")


def exersice_Newtons_method():
    """Give the complicated polynomial defined below, find the solution using
    Newtons method"""

    def P(x):
        y = 924*x**6 - 2772*x**5 + 3150*x**4 - 1680*x**3 + 420*x**2 - 42*x + 1
        return y

    def P_prime(x):
        y = 6*924*x**5 - 5*2772*x**4 + 4*3150*x**3 - 3*1680*x**2 + 2*420*x - 42
        return y

    # Plot the function to estimate the roots
    x = np.linspace(0, 1, 1000)
    plt.figure()
    plt.plot(x, P(x))
    # Place your estimates in this list
    x_estimates = [0.03, 0.165, 0.38, 0.61, 0.83, 0.96]

    def newton(x):
        """Newton's method for the function P(x)"""
        max_error = 1e-10
        delta = 1
        while abs(delta) > max_error:
            delta = P(x)/P_prime(x)
            x -= delta
        return x
    roots = []

    for x in x_estimates:
        roots.append(newton(x))

    print(f" The roots are found to be:")
    for i, root in enumerate(roots):
        print(f"root_{i+1} = {root}")


def exersice_multivariable_Newton():
    """Given a circuit with nonlinear elements find the voltages across the
    diode element. The circuit is a diamond shape with a diode through the
    middle each side of the diamond has a resistor and the top and bottom
    points of the diamond have a voltage difference V_p. V1, V2 correspond to
    the voltage at the left and right corners.
    """

    # Parameters from the problem
    V_p = 5
    # Resistors
    R1 = 1e3
    R2 = 4e3
    R3 = 3e3
    R4 = 2e3
    # Diode
    I0 = 3e-9
    V_T = 0.05

    def f1(V1, V2):
        """Equation from Kirchhoff's law, should equal 0"""
        return (V1 - V_p)/R1 + V1/R2 + I0*(np.exp((V1 - V2)/V_T) - 1)

    def f2(V1, V2):
        """Equation from Kirchhoff's law, should equal 0"""
        return (V2 - V_p)/R3 + V2/R4 - I0*(np.exp((V1 - V2)/V_T) - 1)

    def J(V1, V2):
        """Jacobian df_i/dV_j"""
        f11 = 1/R1 + 1/R2 + I0/V_T*np.exp((V1 - V2)/V_T)
        f12 = -I0/V_T*np.exp((V1 - V2)/V_T)
        f21 = I0/V_T*np.exp((V1 - V2)/V_T)
        f22 = 1/R3 + 1/R4 + I0/V_T*np.exp((V1 - V2)/V_T)
        return np.array([[f11, f12], [f21, f22]])

    delta = [1, 1]
    max_error = 1e-10
    # Guess the points around the answer
    V1 = 0.
    V2 = 1.
    while abs(delta[0]) > max_error or abs(delta[1]) > max_error:
        A = J(V1, V2)
        b = np.array([f1(V1, V2), f2(V1, V2)])
        # J delta_x = f(x). Find delta_x to update the values
        delta = np.linalg.solve(A, b)
        V1 -= delta[0]
        V2 -= delta[1]
    print(f"The voltage across the diode is {V1 - V2}")


def exercise_golden_ratio_search():
    """Find the temperature that gives the best efficiancy for a tungsten
    filiment lightbulb"""

    def integrand(x):
        """Integrand for the efficiency equation"""
        return x**3/(np.exp(x) - 1)

    def eta(T):
        """Efficiency for the lightbulb at temperature T, we integrate over the
        visible spectrum"""
        # Lower end of visible light
        wavelength_1 = 390e-9
        # Upper end of visible light
        wavelength_2 = 750e-9
        # Lower limit
        a = h*c/(wavelength_2*k*T)
        # Upper limit
        b = h*c/(wavelength_1*k*T)
        # Use premade Gaussian quadrature
        integral = quad(integrand, a, b)
        return integral[0]

    # Tungsten's melting point is 3695
    numPoints = 1000
    T = np.linspace(300, 20000, numPoints)
    efficiency = np.empty(numPoints)
    for i in range(numPoints):
        efficiency[i] = eta(T[i])
    plt.figure()
    plt.plot(T, efficiency)
    plt.xlabel('T(Kelvin)')
    plt.ylabel(r'$\eta$')
    plt.tight_layout()

    # We could find the derivative w.r.t. T but we will move along anyway
    max_error = 1
    # Golden ratio
    z = (1 + 5**(0.5))/2

    # Initial bracket
    T1 = 5000
    T4 = 10000
    T2 = T4 - (T4 - T1)/z
    T3 = T1 + (T4 - T1)/z

    # Initial function values
    e1 = -eta(T1)
    e2 = -eta(T2)
    e3 = -eta(T3)
    e4 = -eta(T4)

    while T4 - T1 > max_error:
        if e2 < e3:
            # Then the minimum is between T1 and T3, shift the bracket to the
            # left
            T4, e4 = T3, e3
            T3, e3 = T2, e2
            T2 = T4 - (T4 - T1)/z
            e2 = -eta(T2)
        else:
            # Otherwise shift the bracket to the right
            T1, e1 = T2, e2
            T2, e2 = T3, e3
            T3 = T1 + (T4 - T1)/z
            e3 = -eta(T3)
    T_min = 1/2*(T1 + T4)

    print(f"The most efficient temperature is {T_min}K")
    print(f"This gives an efficiency of {eta(T_min)}")
    print("The melting temperature of tungeston is 3695K")
    print(f"which would give an efficiency of {eta(3695)}")
