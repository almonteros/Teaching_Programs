# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:16:23 2022

@author: almon
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams.update({'font.size': 32, 'axes.linewidth': 2,
                     'lines.linewidth': 4})
# This is based on the book "Computational Physics" by Newman. The revised
# and expanded version.

# Visualization is based on the following stack exchange question.
# https://stackoverflow.com/questions/43204949/how-to-make-trapezoid-and-parallelogram-in-python-using-matplotlib

# Setting up starting values, number of slices and integration limits.
N_1 = 1
a = 0
b = 5
h_1 = (b-a)/N_1


# Here we define the function we are going to integrate, feel free to alter it.
def f(x):
    return np.sin(3*x)  # 3*x**12-6*x**7


# Here I am setting up my plotting environment to aid in visualization.
def plot_trap(ax, k, h):
    corners_x = [a + k*h, a + k*h, a + (k + 1)*h, a + (k + 1)*h]
    corners_y = [0, f(a + k*h), f(a + (k + 1)*h), 0]
    ax.add_patch(patches.Polygon(xy=list(zip(corners_x, corners_y)),
                                 fill=False))


def plot_fun(ax):
    xs = np.linspace(a, b, 1000)
    ys = f(xs)
    ax.plot(xs, ys)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(a, b)
    ax.set_ylim(min(ys), max(ys))


fig, ax = plt.subplots()
fig_number_1 = fig.number
plot_fun(ax)

plot_trap(ax, 0, h_1)
# Doesn't run the code below (range(1,1))
s = 1/2 * f(a) + 1/2 * f(b)
for k in range(1, N_1):
    s += f(a + k*h_1)
    plot_trap(ax, k, h_1)
# Calculate I_1 from the sum
I_1 = h_1 * s

fig, ax = plt.subplots()
fig_number_2 = fig.number
plot_fun(ax)

N_2 = 2 * N_1
h_2 = (b - a)/N_2
s = 1/2 * f(a) + 1/2 * f(b)
for k in range(1, N_2):
    s += f(a + k*h_2)
    plot_trap(ax, k, h_2)
# Calculate I_2 from the sum
I_2 = h_2 * s
esp_2_2 = 1/3*(I_2 - I_1)  # eqn 5.40

I = I_2 + esp_2_2  # eqn 5.41
# esp_i_m is the mth order error for the i-th iteration.

# Equantion 5.42
R_1_1 = I_1  # First iteration, no correction

R_2_1 = I_2  # Second iteration, no correction
R_2_2 = I_2 + esp_2_2  # Second iteration, one correction
# Same as R_2_2 = I_1 + 1/3*(I_2 - I_1)
# Same as R_2_2 = R_2_1 + 1/3 * (R_2_1 - R_1_1)

# To find the next order of error (4th order) We need to go through the process
# of doubling N and halving h again. R_i_m in general is the ith doubling with
# the mth correction. You can't have m>i because we calculate the error based
# on previous iterations.

# Now this has 4th order error, the book finds this in eqn 5.45 We could
# iterate again to get I_3 and find esp_4
# esp_4 = 1/15 * (R_3_2 - R_2_2)
# This is the third iteration corrected once minus the second iteration
# corrected once. We already have R_2_2 from before. R_3_2 = R_3_1 + esp_2
# esp_2 =
# I_3 = R_3_1, eqn 5.42
# I = R_3_2 + esp_4 = R_3_3
# This is the same as I = R_3_1 + esp_2 + esp_4 Or the ith doulbed integral
# with second order errors corrected and fourth order.

# The idea is to increase the number of slices while correcting errors. Once we
# get to our desired accuracy we can stop. This implies a while loop.

acceptable_error = 1e-6

# We need an I_1 and I_2 to start, luckily we already have that!
plt.close(fig_number_1)
plt.close(fig_number_2)

# We will need all the R_i_m from the previous i.
R_i_m = np.array([R_2_1, R_2_2])

# See equation 5.49 with m = 1 and i = 2.
esp = abs(1/3 * (R_2_1 - R_1_1))
i = 3
N_i = 2
while esp > acceptable_error:
    fig, ax = plt.subplots(figsize=(16, 8))
    fig_number = fig.number
    plot_fun(ax)

    # We need to keep the previous R values.
    R_i_prev_m = R_i_m

    # The number of corrections we need to do is the same as the number of
    # iterations.
    R_i_m = np.empty(i)

    # First we double our number of slices.
    N_i = 2 * N_i
    # This causes our widths to be cut in half.
    h_i = (b - a)/N_i

    # We plot the curve with the trapazoids to aid in visualization.
    plot_trap(ax, 0, h_i)

    # Then we go through our integration.
    s = 1/2 * f(a) + 1/2 * f(b)
    for k in range(1, N_i):
        s += f(a + k*h_i)
        plot_trap(ax, k, h_i)
    R_i_m[0] = h_i*s

    # We now make all of our corrections.
    for m in range(1, i):
        esp = 1/(4**m - 1) * (R_i_m[m - 1] - R_i_prev_m[m - 1])
        R_i_m[m] = R_i_m[m - 1] + esp
    # The last esp is our penultimate error, see the note on page 161.
    # We need our error to be positive to compare.
    esp = abs(esp)

    i += 1
    # Round the error and the value of the integral so they fit onto the title.
    power_of_error = int(np.log10(acceptable_error))
    round_e = esp.round(power_of_error + 3)
    round_I = R_i_m[-1].round(power_of_error)
    ax.set_title(f"N = {N_i}, error = {round_e}, I = {round_I}")
    fig.tight_layout()
    # We pause the program so you have time to look at the graphs, change as
    # desired.
    plt.pause(5)
    # We close the previous figure, if you want to keep them open to compare
    # then comment out the line below.
    plt.close(fig_number)

# The last calculated R is our most accurate version of the integral.
I = R_i_m[-1]
print(f"Our integral of f(x) from a = {a} to b = {b} is: {I}")
