#!/bin/python3
"""

Class file containing framework for Monte Carlo simulations 
and assignment tasks

"""

from numpy.random import SeedSequence, default_rng
import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()


class Monte_Carlo:
    """
    This class is for running Monte Carlo simulations to find the integral, 
    variance and error for the area of a circle.

    """
   
    def __init__(self, start, end, N, f, *args):
        """
        Class initialisation for Monte Carlo
    
        Parameters:
        start = Starting coordinate in the square
        end = Final coordinate in the square
        f = The function being passed through
        vars = Any variables required for the function in use
        values = Class initialisation for f-string print result

        """
        self.start = start   # Starting point of square
        self.end = end   # End point 
        self.f = f   # Function
        self.N = N   # Number of iterations
        self.vars = args   # Argument set to be used as a variable within an array
        self.values = 0     # Used for final results

    
    def __str__(self):
        """
        F-string for final results

        """
        return f"(Integral: {self.values[0]}, Var: {self.values[1]}, Err: {self.values[2]})"

    
    def Integral(self):
        """
        This function is used to calculate the integral of a function, via random walks.

        """
        
        # Setting the dimension in use as the length of start
        D = len(self.start)   

        # Set-up for Random walks between rank 0 and nworkers, using random seed generator
        ss = SeedSequence(23456)  
        nworkers_seed = ss.spawn(nworkers)  
        
        # Ensuring each rank performs random walks
        Random_gen = [default_rng(s) for s in nworkers_seed]
        R_num = Random_gen[rank].random((self.N, D))

        # Setting up initial array's
        Sum_func = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)
        Sum_func_sq = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)
        Final_func = np.array(0, dtype=np.float64)
        Final_func_sq = np.array(0, dtype=np.float64)


        for n in R_num:
            for i in range(D):
                n[i] = n[i] * (self.end[i] - self.start[i]) + self.start[i]
            Sum_func += (self.f(n, *self.vars))
            Sum_func_sq += (self.f(n, *self.vars))**2

        comm.Allreduce(Sum_func, Final_func)
        comm.Allreduce(Sum_func_sq, Final_func_sq)

        
        ad_cb = np.prod(np.array(self.end) - np.array(self.start))
        inv_N = 1 / (self.N * nworkers)

        Integral = ad_cb * inv_N * Final_func  
        Variance = inv_N * (Final_func_sq * inv_N - (Final_func * inv_N) ** 2)  
        Error = ad_cb * np.sqrt(Variance)  
        self.values = np.array([Integral, Variance, Error])

        return self.values
    
    def infinity(self):
        '''
        This function is used for infinite or improper cases

        '''

        # Setting the dimension in use as the length of start
        D = len(self.start)

        # Set-up for Random walks between rank 0 and nworkers, using random seed generator
        ss = SeedSequence(23456)  
        nworkers_seed = ss.spawn(nworkers)  

        # Ensuring each rank performs random walks
        Random_gen = [default_rng(s) for s in nworkers_seed]
        R_num = Random_gen[rank].random((self.N, D))

        # Setting up initial array's
        Sum_f = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)
        Sum_func_sq = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)
        Final_func = np.empty(D, dtype=np.float64)
        Final_func_sq = np.empty(D, dtype=np.float64)


        a_inf, b_inf = -1, 1  

        for n in R_num:
            for i in range(D):
                n[i] = n[i] * (b_inf-a_inf)+a_inf
            x = n/(1-n**2) 
            y = (1+n**2)/((1-n**2)**2) 
            Sum_f += self.f(x, *self.vars) * y
            Sum_func_sq += (self.f(x, *self.vars) * y)**2
        

        comm.Allreduce(Sum_func, Final_func)
        comm.Allreduce(Sum_func_sq, Final_func_sq)

        ad_cb = 1
        for i in range(D):
            ad_cb += ad_cb * (b_inf - a_inf)
        inv_N = 1 / (self.N * nworkers)

        # Calculation for the integral, variance, and error
        Integral = ad_cb * inv_N * np.mean(Final_func)
        Variance = inv_N**2 * np.mean(Final_func_sq - Final_func**2)
        Error = ad_cb * np.sqrt(Variance)
        self.values = np.array([Integral, Variance, Error])

        return self.values

###--------------------### Task 1 ####----------------------###

def shape(x, R):

    if round(np.sum(np.square(x)), 2) <= (R**2):
        return True
    else:
        return False

def quad(x, a, b):
    return a*x**2 + b


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
    print(f"Expected area for a 2D Circle via MC, r = {R1[0]}: {I_c}")
    print(f"Actual area: {area_2D}")

    print(f"Expected area for 3D Sphere radius via MC, r = {R1[0]}: {I_s}")
    print(f"Actual area: {area_3D}")

    print(f"Expected area for 4D Hypersphere radius via MC, r = {R1[0]}: {I_hs_4}")
    print(f"Actual area: {area_4D}")

    print(f"Expected area for 5D Hpersphere radius via MC, r = {R1[0]}: {I_hs_5}")
    print(f"Actual area:{area_5D}") 

###--------------------### Task 2 ###----------------------###

pi = np.pi
e = np.exp

def gauss_function(x, x0, sigma):
    """
    This function is used to describe the Gaussian function
    Parameters:
        sigma: Standard deviation of the distribution
        |x - x_0|^2: Mean value of the distribution

    """
    return 1/(sigma * np.sqrt(2 * pi)) * e((-(x - x_0)**2)/(2 * sigma**2))


R = np.repeat(4, 5)
sigma, x, x_0 = 5, 0.65, 1

mc_gaussian = Monte_Carlo(-R, R, N, gauss_function, *variance)
g = Monte_Carlo.infinity(mc_gaussian)

if rank==0:
    print(f"Expected Gaussian distribution with {sigma}, {x}, and {x_0}: {g}")


MPI.Finalize()
