#!/bin/python3
"""

Module containing framework for Monte Carlo simulations 
and assignment tasks

MIT License

Copyright (c) 2025 Sean McGeoghegan

See LICENSE.txt for details

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
   
    def __init__(self, start, end, N, f, *varis):
        """
        Class initialisation for Monte Carlo
    
        Parameters:
        start = Starting coordinate in the square
        end = Final coordinate in the square
        f = The function being passed through
        vars = Any variables required for the function in use
        values = Class initialisation for f-string print result

        """
        self.start = start   # Starting point of integral
        self.end = end   # End point of integral
        self.f = f   # Function to be passed through MC
        self.N = N   # Number of iterations to be performed by MC
        self.vars = varis   # Variable(s) within an array for a function
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

        # Used to set up numerator/denominator of (a*d - c * b)/N
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
        Sum_func = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)
        Sum_func_sq = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)

        Final_func = np.empty(D, dtype=np.float64)
        Final_func_sq = np.empty(D, dtype=np.float64)

        # a_inf and b_inf are used as the variable names for the bottom and top values of the integral 
        # in infinite/improper cases.
        a_inf, b_inf = -1, 1  

        
        for n in R_num:
            for i in range(D):
                n[i] = n[i] * (b_inf - a_inf) + a_inf
            x = n/(1-n**2) 
            y = (1+n**2)/((1-n**2)**2) 
            Sum_func += self.f(x, *self.vars) * y
            Sum_func_sq += (self.f(x, *self.vars) * y)**2
        

        comm.Allreduce(Sum_func, Final_func)
        comm.Allreduce(Sum_func_sq, Final_func_sq)

        # Used to set up numerator/denominator of (a*d - c * b)/N
        ad_cb = np.prod(b_inf - a_inf)
        inv_N = 1 / (self.N * nworkers)

        # Calculation for the integral, variance, and error
        Integral = np.mean(ad_cb * inv_N * Final_func)
        Variance = inv_N * np.mean((Final_func_sq * inv_N - (Final_func * inv_N) ** 2)) ### FIX THIS 
        Error = ad_cb * np.sqrt(Variance)

        self.values = np.array([Integral, Variance, Error])
        return self.values

