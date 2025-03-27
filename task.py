#!/bin/python3
"""
Module containing set-up for calculations of given tasks.

MIT License

Copyright (c) 2025 Sean McGeoghegan

See LICENSE.txt for details

"""

import time as t
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from class_1 import MonteCarlo

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

###--------------------### Test of Monte Carlo sim ####----------------------###

def quad(x, a, b):
    """
    This function is used as a test to pass through the Monte Carlo simulation.

    Parameters
        a: Co-efficient of the x^2 term
        x: Variable
        b: Constant term

    """
    return a*x**2 + b

# Set-up of variables to test MC with quadrature function "quad"
N = int(100000)
x2 = np.array([2])
y2 = np.array([4])
v = np.array([1, 2])

Test = MonteCarlo(x2, y2, N, quad, *v)
I = MonteCarlo.integral(Test)

if rank == 0:
    print(f"The integral of x^2 between {x2} and {y2}: {Test}")

###--------------------### Task 1 ####----------------------###

def shape(x, r):
    """
    This function is used for any circular-shaped object. 
    Passed through if/else statement to check if some point in space
    is within a circular object, or lies upon its line/surface
 
    Parameters: 
        x: Describes a random point in space
        r: Describes the radius of a circle

    """

    if round(np.sum(np.square(x)), 2) <= (r**2):
        return True
    else:
        return False

# Variable set-up for spheres of dimensions (D): 2-5
D_2 = 2
D_3 = 3
D_4 = 4
D_5 = 5

# Variable set-up for the radius (R) used for each dimension
R1 = np.array([1])

# Variable set-up for the start (S) and end (E) points being repeated across an array dependent on dimension size
S1 = np.repeat(-R1, D_2)
E1 = np.repeat(R1, D_2)

S2 = np.repeat(-R1, D_3)
E2 = np.repeat(R1, D_3)

S3 = np.repeat(-R1, D_4)
E3 = np.repeat(R1, D_4)

S4 = np.repeat(-R1, D_5)
E4 = np.repeat(R1, D_5)

t1_a = t.time()

# Monte Carlo simulation for a circle, sphere, 4D & 5D hypersphere
circle = MonteCarlo(S1, E1, N, shape, *R1)
I_c = MonteCarlo.integral(circle)

sphere = MonteCarlo(S2, E2, N, shape, *R1)
I_s = MonteCarlo.integral(sphere)

hypersphere_4D = MonteCarlo(S3, E3, N, shape, *R1)
I_hs_4 = MonteCarlo.integral(hypersphere_4D)

hypersphere_5D = MonteCarlo(S4, E4, N, shape, *R1)
I_hs_5 = MonteCarlo.integral(hypersphere_5D)

pi = np.pi

# Variable set-up of the real area/volume for each dimension in use
area_2D = pi*R1[0]**2
area_3D = (4/3)*pi*R1[0]**3
area_4D = ((pi**2)/2)*R1[0]**4
area_5D = (8/15)*(pi**2)*R1[0]**5

if rank==0:
    print("")
    print(f"Calculated area for a 2D circle via MC, r = {R1[0]}: {I_c}")
    print(f"Expected area for a 2D circle: {area_2D}")
    print("")
    print(f"Calculated area for 3D sphere radius via MC, r = {R1[0]}: {I_s}")
    print(f"Expected volume for a 3D sphere: {area_3D}")
    print("")
    print(f"Calculated area for 4D hypersphere radius via MC, r = {R1[0]}: {I_hs_4}")
    print(f"Expected volume of a 4D hypersphere: {area_4D}")
    print("")
    print(f"Calculated area for 5D hypersphere radius via MC, r = {R1[0]}: {I_hs_5}")
    print(f"Expected volume of a 5D hypersphere:{area_5D}")
    print("")

# Final calculation timing and the time taken to run it
t1_b = t.time()
dt1 = t1_b - t1_a
print(f"Time taken for calculation on the area of a circle in each dimension = {dt1} secs")

# Set-up to find the speed-up (S), taken from the time taken (T) on each N core(s) to be plotted
T1 = np.array([28.4527, 28.8680, 28.4984, 28.4362, 26.2481])
N_c = np.array([1, 2, 4, 8, 16])

S1 = np.zeros(5)
for i in range(0,4):
    S1[i+1] =+ T1[0]/T1[i]

plt.figure(1)
plt.plot(N_c, S1)
plt.title("Speed up of N cores v N cores")
plt.ylabel("Speed up for N cores")
plt.xlabel("N cores")
plt.savefig("Graph1.jpg")

###--------------------### Task 2 ###----------------------###

e = np.exp

def gauss_function(x, sigma, x_0):
    """
    This function is used to describe the Gaussian function
    Parameters:
        sigma: Standard deviation of the distribution
        |x - x_0|^2: Mean value of the distribution

    """
    return (1/(sigma * np.sqrt(2 * pi))) * e((-(x - x_0)**2)/(2 * sigma**2))

# R sets the array used for start or end at [1, 1, 1, 1, 1]
R = np.repeat(1, 5)

# Variables used for the standard deviation, and the mean
vars_1 = np.array([0.4, 3])
vars_2 = np.array([0.2, 1])
vars_3 = np.array([0.5, 4])

# Initial measurement for calculation timing
t2_a = t.time()

gauss_f_1 = MonteCarlo(-R, R, N, gauss_function, *vars_1)
g_1 = MonteCarlo.infinity(gauss_f_1)

gauss_f_2 = MonteCarlo(-R, R, N, gauss_function, *vars_2)
g_2 = MonteCarlo.infinity(gauss_f_2)

gauss_f_3 = MonteCarlo(-R, R, N, gauss_function, *vars_3)
g_3 = MonteCarlo.infinity(gauss_f_3)

if rank==0:
    print("")
    print(f"Gaussian dist. σ = {vars_1[0]}, μ = {vars_1[1]}: Integral = {g_1[0]}, Variance = {g_1[1]}, Error = {g_1[2]}")
    print("")
    print(f"Gaussian dist. σ = {vars_2[0]}, μ = {vars_2[1]}: Integral = {g_2[0]}, Variance = {g_2[1]}, Error = {g_2[2]}")
    print("")
    print(f"Gaussian dist. σ = {vars_3[0]}, μ = {vars_3[1]}: Integral = {g_3[0]}, Variance = {g_3[1]}, Error = {g_3[2]}")

# Final calculation timing and the time taken to run it
t2_b = t.time()
dt2 = t2_b - t2_a
print(f"Time taken for Gaussian distribution calculations = {dt2} secs")

# Set-up to find the speed-up (S), taken from the time taken (T) on each N core(s) to be plotted
T2 = np.array([14.3461, 14.8378, 14.8083, 14.9860, 14.2158])
N_c = np.array([1, 2, 4, 8, 16])

S2 = np.zeros(5)
for i in range(0,4):
    S2[i+1] =+ T2[0]/T2[i]

plt.figure(2)
plt.plot(N_c, S2)
plt.title("Speed up of N cores v N cores")
plt.ylabel("Speed up for N cores")
plt.xlabel("N cores")
plt.savefig("Graph2.jpg")

# Finalise the MPI function
MPI.Finalize()
