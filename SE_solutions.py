# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:43:56 2021

@author: almonter
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.special import hermite
plt.rcParams.update({'font.size': 32, 'axes.linewidth': 2,
                     'lines.linewidth': 2})
# Feel free to change the experiment to whatever you like by changing exp_num.
exp_num = 5
experiments = {0: 'Particle in a box',
               1: 'Free particle, no barrier',
               2: 'Free particle, low barrier',
               3: 'Free particle, high barrier',
               4: 'Step potential',
               5: 'Harmonic Oscillator'}
experiment = experiments[exp_num]

# These parameters are set so that the visuals come out looking nice.
pi = np.pi
h = 1  # 6.62607004*10**-34
m = 1  # 9.10938356*10**-31
hbar = h/(2*np.pi)
numPoints = 300
delta_t = 1/(numPoints)


def wavefunction(numPoints=numPoints):
    if experiment == "Particle in a box":
        # Feel free to change the two parameters below
        n = 2  # Excitation number
        L = 1  # Size of box

        xMin = 0
        xMax = L
        x = np.linspace(xMin, xMax, numPoints)

        A = np.sqrt(2/L)  # normilization constant
        k = n*np.pi/L  # wavenumber
        E = k**2*hbar**2/(2*m)  # Energy
        psi_x = A*np.sin(x*k)

    elif experiment == "Free particle, no barrier":
        # Feel free to change the two parameters below
        v = 2  # Velocity
        L = 1  # "Size" of free space

        xMin = 0
        xMax = L
        x = np.linspace(xMin, xMax, numPoints)

        A = 1/L**0.5
        k = m*v/hbar
        E = k**2*hbar**2/(2*m)
        psi_x = A*np.exp(-1j*k*x)

    elif experiment == "Free particle, low barrier":
        # Feel free to change the two parameters below
        E = 2  # Energy of particle
        U = E*(1 - 0.08)  # Energy of the barrier (U < E)

        k0 = np.sqrt(2*m*E)/hbar
        k1 = np.sqrt(2*m*(E - U))/hbar
        # esp = (k0 - k1)/(k0 + k1)
        # A = ((1 + esp**2)*L + 4*k0**2*L/(k0 + k1)**2
        #     + esp/k0*np.sin(2*k0*L))**(-1/2)

        L = 10
        xMin = -L
        xMax = L
        x = np.linspace(xMin, xMax, numPoints)

        A = 1
        psi_left = A*(np.exp(1j*k0*x) + (k0-k1)/(k0 + k1)*np.exp(-1j*k0*x))
        psi_right = 2*A*k0/(k0 + k1)*np.exp(1j*k1*x)
        psi_x = np.where(x < 0, psi_left, psi_right)

    elif experiment == "Free particle, high barrier":
        # Feel free to change the two parameters below
        E = 0.02  # Energy of the Particle
        U = E + 0.01*E  # Energy of the barrier ( U > E)

        L = 10
        xMin = -L
        xMax = L
        x = np.linspace(xMin, xMax, numPoints)

        A = 1
        k0 = np.sqrt(2*m*E)/hbar
        k1 = np.sqrt(2*m*(U - E))/hbar
        psi_left = A*(np.sin(k0*x)-k0/k1*np.cos(k0*x))
        psi_right = -A*k0/k1*np.exp(-k1*x)
        psi_x = np.where(x < 0, psi_left, psi_right)

    elif experiment == "Step potential":
        # Feel free to change the two parameters below
        E = 0.01  # Energy of particle
        U = E*(1 + 0.0000001)  # Energy of the step potential (U > E)
        L = 6

        xMin = -L
        xMax = L
        x = np.linspace(xMin, xMax, numPoints)

        A = 1
        k0 = np.sqrt(2*m*E)/hbar
        k1 = np.sqrt(2*m*(U - E))/hbar
        denom = 4j*k1*k0*np.cosh(k1*L) - 2*np.sinh(k1*L)*(k1**2 - k0**2)
        B = 2*(k1**2 + k0**2)*np.sinh(k1*L)/denom*A
        C = 2j*k0*(k1 + 1j*k0)*np.exp(-k1*L)/denom*A
        D = 2j*k0*(k1 - 1j*k0)*np.exp(k1*L)/denom*A
        F = 4j*k1*k0*np.exp(-1j*k0*L)/denom*A

        psi_x = np.zeros(numPoints, dtype=np.complex64)
        for i in range(numPoints):
            pos = x[i]
            if pos < 0:
                psi_x[i] = A*np.exp(1j*k0*pos) + B*np.exp(-1j*k0*pos)
            elif pos < L:
                psi_x[i] = C*np.exp(k1*pos) + D*np.exp(-k1*pos)
            else:
                psi_x[i] = F*np.exp(1j*k0*pos)

    elif experiment == "Harmonic Oscillator":
        # Feel free to change the three parameters below
        n = 0  # nth excitation
        w0 = 1  # Frequency of the simple harmonic oscillator

        H = hermite(n)
        A = (m*w0/(hbar*pi))**(1/4)*(1/np.sqrt(2**n)*factorial(n))
        E = (n + 1/2)*hbar*w0

        x_t = np.sqrt(2*E/(w0**2*m))

        xMin = -2*x_t
        xMax = 2*x_t
        x = np.linspace(xMin, xMax, numPoints)

        y_turn = np.abs(A*np.exp(-(1/2)*m*w0/hbar*x_t**2)
                        * H(np.sqrt(m*w0/hbar)*x_t))**2
        psi_x = A*np.exp(-(1/2)*m*w0/hbar*x**2)*H(np.sqrt(m*w0/hbar)*x)

        ax3.scatter(x_t, y_turn, color="black")
        ax3.scatter(-x_t, y_turn, color="black")
    return psi_x, E, x


# Setting up the figures.
fig = plt.figure()
fig_num = fig.number

ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

psi_x, E, x = wavefunction()
y1_max = np.max(psi_x.real)
y2_max = y1_max
y3_max = max(np.abs(psi_x)**2)
y3_min = min(np.abs(psi_x)**2)
xMin = min(x)
xMax = max(x)

# Set the initial data.
line1, = ax1.plot(x, np.zeros(numPoints), '-', lw=2)
line2, = ax2.plot(x, np.zeros(numPoints), '-', lw=2)
line3, = ax3.plot(x, np.zeros(numPoints), '-', lw=2)

# Label axis.
ax1.set_xlim(xMin, xMax)
ax1.set_ylim(-y1_max, y1_max)
ax1.set_ylabel(r"$Re(\Psi(x,t))$")
ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_xlim(xMin, xMax)
ax2.set_ylim(-y2_max, y2_max)
ax2.set_ylabel(r"$\mathcal{I}m(\Psi(x,t))$")
ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_xlim(xMin, xMax)
ax3.set_ylim(y3_min, y3_max)
ax3.set_xlabel("x")
ax3.set_ylabel("$|\Psi(x,t)|^2$")
ax3.set_xticks([])
ax3.set_yticks([])

plt.show(block=False)

fig.canvas.draw()

omega = E/hbar
# The time is taken over a single period, T

T = 2*np.pi/omega
t = np.linspace(0, T, numPoints+1)[:, None]
y = psi_x[None, :]*np.exp(-1j*omega*t)

y_R = np.real(y)
y_I = np.imag(y)

y_P = np.abs(psi_x)**2
line3.set_ydata(y_P)
i = 0
while plt.fignum_exists(fig_num):
    line1.set_ydata(y_R[i])
    line2.set_ydata(y_I[i])

    # Redraw the graph
    ax1.draw_artist(ax1.patch)
    ax1.draw_artist(line1)
    ax2.draw_artist(ax2.patch)
    ax2.draw_artist(line2)

    fig.canvas.update()
    fig.canvas.flush_events()
    # Increment the time and circle back every period
    i = (i + 1) % numPoints
