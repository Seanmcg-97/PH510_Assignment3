#!/bin/python3

import time as t
import numpy as np
from mpi4py import MPI
from class_1 import Monte_Carlo


# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

###--------------------### Task 1 ####----------------------###

def shape(x, R):
    """
    This function is used for any circular-shaped object. 
    Passed through if/else statement to check if some point in space
    is within a circular object, or lies upon its line/surface
 
    Parameters: 
        x: Describes a random point in space
        R: Describes the radius of a circle

    """

    if round(np.sum(np.square(x)), 2) <= (R**2):
        return True
    else:
        return False

def quad(x, a, b):
    """
    This function is used as a test to pass through the Monte Carlo simulation.

    Parameters
        a: Co-efficient of the x^2 term
        x: Variable
        b: Constant term

    """    
    return a*x**2 + b


pi = np.pi

N = int(100000)
x2 = np.array([2])
y2 = np.array([4])
v = np.array([1, 2])


Test = Monte_Carlo(x2, y2, N, quad, *v)
I = Monte_Carlo.Integral(Test)

if rank == 0:
    print(f"The integral of x^2 between {x2} and {y2}: {Test}")


# Variable set-up for spheres of dimensions: 2-5
D_2 = 2
D_3 = 3
D_4 = 4
D_5 = 5

# Variable set-up for the radius used for each dimension
R1 = np.array([1])

# Variable set-up for the start and end points being repeated across an array dependent on dimension size 
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
circle = Monte_Carlo(S1, E1, N, shape, *R1)
I_c = Monte_Carlo.Integral(circle)

sphere = Monte_Carlo(S2, E2, N, shape, *R1)
I_s = Monte_Carlo.Integral(sphere)

hypersphere_4D = Monte_Carlo(S3, E3, N, shape, *R1)
I_hs_4 = Monte_Carlo.Integral(hypersphere_4D)

hypersphere_5D = Monte_Carlo(S4, E4, N, shape, *R1)
I_hs_5 = Monte_Carlo.Integral(hypersphere_5D)

# Variable set-up of the real area/volume for each dimension in use
area_2D = pi*R1[0]**2
area_3D = (4/3)*pi*R1[0]**3
area_4D = ((pi**2)/2)*R1[0]**4
area_5D = (8/15)*(pi**2)*R1[0]**5

if rank==0:
    print("")
    print(f"Expected area for a 2D ircle via MC, r = {R1[0]}: {I_c}")
    print(f"Calculated area for a 2D circle: {area_2D}")
    print("")
    print(f"Expected area for 3D Sphere radius via MC, r = {R1[0]}: {I_s}")
    print(f"Calculated volume for a 3D sphere: {area_3D}")
    print("")
    print(f"Expected area for 4D Hypersphere radius via MC, r = {R1[0]}: {I_hs_4}")
    print(f"Calculated volume of a 4D hypersphere: {area_4D}")
    print("")
    print(f"Expected area for 5D Hpersphere radius via MC, r = {R1[0]}: {I_hs_5}")
    print(f"Calculated volume of a 5D hypersphere:{area_5D}") 
    print("")

t1_b = t.time()
dt1 = t1_b - t1_a
print(f"Time taken for calculation on the area of a circle in each dimension = {dt1} secs")


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

t2_a = t.time()

gauss_f_1 = Monte_Carlo(-R, R, N, gauss_function, *vars_1)
g_1 = Monte_Carlo.infinity(gauss_f_1)

gauss_f_2 = Monte_Carlo(-R, R, N, gauss_function, *vars_2)
g_2 = Monte_Carlo.infinity(gauss_f_2)

gauss_f_3 = Monte_Carlo(-R, R, N, gauss_function, *vars_3)
g_3 = Monte_Carlo.infinity(gauss_f_3)


if rank==0:
    print("")
    print(f"Expected Gaussian distribution with sigma = {vars_1[0]}, mean = {vars_1[1]}: Integral = {g_1[0]}, Variance = {g_1[1]}, Error = {g_1[2]}")
    print("")
    print(f"Expected Gaussian distribution with sigma = {vars_2[0]}, mean = {vars_2[1]}: Integral = {g_2[0]}, Variance = {g_2[1]}, Error = {g_2[2]}")
    print("")
    print(f"Expected Gaussian distribution with sigma = {vars_3[0]}, mean = {vars_3[1]}: Integral = {g_3[0]}, Variance = {g_3[1]}, Error = {g_3[2]}")

t2_b = t.time()
dt2 = t2_b - t2_a
print(f"Time taken for Gaussian distribution calculations = {dt2} secs")

# Finalise the MPI function
MPI.Finalize()
