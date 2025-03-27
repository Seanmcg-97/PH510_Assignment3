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


class MonteCarlo:
    """
    This class is for running Monte Carlo simulations to find the integral, 
    variance and error for the area of a circle.

    """

    def __init__(self, start, end, num, f, *varis):
        """
        Class initialisation for Monte Carlo
    
        Parameters:
        start = Starting coordinate in the square
        end = Final coordinate in the square
        f = The function being passed through
        varis = Any variables required for the function in use
        values = Class initialisation for f-string print result

        """
        self.start = start   # Starting point of integral
        self.end = end   # End point of integral
        self.f = f   # Function to be passed through MC
        self.num = num   # number of iterations to be performed by MC
        self.vars = varis   # Variable(s) within an array for a function
        self.values = 0     # Used for final results

    def __str__(self):
        """
        F-string for final results

        """
        return f"(integral: {self.values[0]}, Var: {self.values[1]}, Err: {self.values[2]})"


    def integral(self):
        """
        This function is used to calculate the integral of a function, via random walks.

        """

        # Setting the dimension in use as the length of start
        d = len(self.start)

        # Set-up for Random walks between rank 0 and nworkers, using random seed generator
        ss = SeedSequence(23456)
        nworkers_seed = ss.spawn(nworkers)

        # Ensuring each rank performs random walks
        random_gen = [default_rng(s) for s in nworkers_seed]
        r_num = random_gen[rank].random((self.num, d))

        # Setting up initial array's
        s_func = np.zeros_like(self.f(np.zeros(d), *self.vars), dtype=np.float64)
        s_func_sq = np.zeros_like(self.f(np.zeros(d), *self.vars), dtype=np.float64)
        final_func = np.array(0, dtype=np.float64)
        final_func_sq = np.array(0, dtype=np.float64)

        # double for loop to approximately calculate the integral over a select domain
        for n in r_num:
            for i in range(d):
                n[i] = n[i] * (self.end[i] - self.start[i]) + self.start[i]
            s_func += (self.f(n, *self.vars))
            s_func_sq += (self.f(n, *self.vars))**2

        # Setting up a reduction of for loop values before broadcasting to workers
        comm.Allreduce(s_func, final_func)
        comm.Allreduce(s_func_sq, final_func_sq)

        # Used to set up numerator/denominator of (a*d - c * b)/n
        ad_cb = np.prod(np.array(self.end) - np.array(self.start))
        inv_n = 1 / (self.num * nworkers)

        # Variable set-ups for calculations for the integral, variance and error values
        integral = ad_cb * inv_n * final_func
        variance = inv_n * (final_func_sq * inv_n - (final_func * inv_n) ** 2)  
        error = ad_cb * np.sqrt(variance)
        self.values = np.array([round(integral, 5), round(variance, 10), round(error, 5)])

        return self.values

    def infinity(self):
        '''
        This function is used for infinite or improper cases

        '''

        # Setting the dimension in use as the length of start
        d = len(self.start)

        # Set-up for Random walks between rank 0 and nworkers, using random seed generator
        ss = SeedSequence(23456)
        nworkers_seed = ss.spawn(nworkers)

        # Ensuring each rank performs random walks
        random_gen = [default_rng(s) for s in nworkers_seed]
        r_num = random_gen[rank].random((self.num, d))

        # Setting up initial array's
        s_func = np.zeros_like(self.f(np.zeros(d), *self.vars), dtype=np.float64)
        s_func_sq = np.zeros_like(self.f(np.zeros(d), *self.vars), dtype=np.float64)
        final_func = np.empty(d, dtype=np.float64)
        final_func_sq = np.empty(d, dtype=np.float64)

        # a_inf and b_inf are used as the variable names for the bottom 
        # and top values of the integral in infinite/improper cases.
        a_inf, b_inf = -1, 1

        # Double for loop to approximately calculate the integral over a select domain. Including
        # the added factors within the integral calculation
        for n in r_num:
            for i in range(d):
                n[i] = n[i] * (b_inf - a_inf) + a_inf
            x = n/(1-n**2)
            y = (1+n**2)/((1-n**2)**2)
            s_func += self.f(x, *self.vars) * y
            s_func_sq += (self.f(x, *self.vars) * y)**2

        # Setting up a reduction of for loop values before broadcasting to workers
        comm.Allreduce(s_func, final_func)
        comm.Allreduce(s_func_sq, final_func_sq)

        # Used to set up numerator/denominator of (a*d - c * b)/n
        ad_cb = np.prod(b_inf - a_inf)
        inv_n = 1 / (self.num * nworkers)

        # Calculation for the integral, variance, and error
        integral = np.mean(ad_cb * inv_n * final_func)
        variance = inv_n * np.mean((final_func_sq * inv_n - (final_func * inv_n) ** 2))
        error = ad_cb * np.sqrt(variance)
        self.values = np.array([round(integral, 5), round(variance, 9), round(error, 5)])
        return self.values
