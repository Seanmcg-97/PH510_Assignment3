#!/bin/python3
from numpy.random import SeedSequence, default_rng
import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()


class Monte_Carlo:
    def __init__(self, start, end, N, f, *args):
        self.start = start
        self.end = end
        self.f = f
        self.N = N
        self.vars = args # if args are passed, assumed to be a variable array for function
        self.values = 0   # data to be returned for any method

    def __str__(self):
        return f"(Integral: {self.values[0]}, Var: {self.values[1]}, Err: {self.values[2]})"

    def Integral(self):
        D = len(self.start)
        ss = SeedSequence(23456)  
        nworkers_seed = ss.spawn(nworkers)  
        Random_gen = [default_rng(s) for s in nworkers_seed]
        R_num = Random_gen[rank].random((self.N, D))

        Sum_final = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)
        E_f_sq = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)

        for n in R_num:
            for i in range(D):
                n[i] = n[i] * (self.end[i] - self.start[i]) + self.start[i]
            Sum_final += (self.f(n, *self.vars))
            E_f_sq += (self.f(n, *self.vars))**2

        Final_SF = np.array(0, dtype=np.float64)
        Final_F_sq = np.array(0, dtype=np.float64)

        comm.Allreduce(Sum_final, Final_SF)
        comm.Allreduce(E_f_sq, Final_F_sq)

        I_numerator = np.prod(np.array(self.end) - np.array(self.start))
        I_denominator = 1 / (self.N * nworkers)

        Final_integral = I_numerator * I_denominator * Final_SF  
        Final_Variance = I_denominator * (Final_F_sq * I_denominator - (Final_SF *I_denominator) ** 2)  
        Final_Error = I_numerator * np.sqrt(Final_Variance)  
        self.values = np.array([Final_integral, Final_Variance, Final_Error])

        return self.values
    
    def infinity(self):
        '''
        This is used for improper/infinite cases

        '''
        D = len(self.start)
        ss = SeedSequence(23456)  
        nworkers_seed = ss.spawn(nworkers)  
        Random_gen = [default_rng(s) for s in nworkers_seed]
        R_num = Random_gen[rank].random((self.N, D))

        Sum_f = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)
        E_f_sq = np.zeros_like(self.f(np.zeros(D), *self.vars), dtype=np.float64)

        inf_starts, inf_ends = -1, 1  

        for n in R_num:
            for i in range(D):
                n[i] = n[i] * (inf_ends-inf_starts)+inf_starts
            x = n/(1-n**2) 
            factor = (1+n**2)/((1-n**2)**2) 
            Sum_f += self.f(x, *self.vars) * factor
            E_f_sq += (self.f(x, *self.vars) * factor)**2
        
        Final_SF = np.empty(dim, dtype=np.float64)
        Final_F_sq = np.empty(dim, dtype=np.float64)

        comm.Allreduce(Sum_f, Final_SF)
        comm.Allreduce(E_f_sq, Final_F_sq)

        I_numerator = 2
        I_denominator = 1 / (self.N * nworkers)

        # Calculate integral, variance, and error
        Final_integral = I_numerator * I_denominator * np.mean(Final_SF)
        Final_Variance = I_denominator**2 * np.mean(Final_F_sq - Final_SF**2)
        Final_Error = I_numerator * np.sqrt(Final_Variance)
        self.values = np.array([Final_integral, Final_Variance, Final_Error])

        return self.values
        

def shape(x, R):  # FUNCTION FOR ANY ROUND SHAPE
    return round(np.sum(np.square(x)), 5) <= (R**2)

def test(x, a, b):
    return a*x**2 + b

pi = np.pi

def gauss_function(x, x0, sigma):
    return 1/(sigma*np.sqrt(2*pi)) * np.exp((-(x-x0)**2)/(2*sigma**2))


N = int(100000)
start = np.array([2])
end = np.array([4])
v = np.array([1, 2])


Test = Monte_Carlo(start, end, N, test, *v)
I = Monte_Carlo.Integral(Test)

if rank == 0:
    print(f"Evaluating integral of x^2 between {start} and {end}: {Test}")


# Variable set-up for spheres of dimensions: 2-5
D_2 = 2
D_3 = 3
D_4 = 4
D_5 = 5

# Variable set-up for the radius used for each dimension
R1 = np.array([1])
R2 = np.array([2])

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
c = Monte_Carlo.Integral(circle)

sphere = Monte_Carlo(S2, E2, N, shape, *R1)
s = Monte_Carlo.Integral(sphere)

hypersphere_4D = Monte_Carlo(S3, E3, N, shape, *R1)
hs_4 = Monte_Carlo.Integral(hypersphere_4D)

hypersphere_5D = Monte_Carlo(S4, E4, N, shape, *R1)
hs_5 = Monte_Carlo.Integral(hypersphere_5D)

# Variable set-up of the real area/volume for each dimension in use
D_2_f = pi*R1[0]**2
D_3_f = (4/3)*pi*R1[0]**3
D_4_f = ((pi**2)/2)*R1[0]**4
D_5_f = (8/15)*(pi**2)*R1[0]**5

if rank==0:
    print(f"Expected value for 2D Circle radius of {R1[0]}: {c}")
    print(f"Actual value: {D_2_f}")
    print(f"Expected value for 3D Sphere radius of {R1[0]}: {s}")
    print(f"Actual value: {D_3_f}")
    print(f"Expected value for 4D Hypersphere radius of {R1[0]}: {hs_4}")
    print(f"Actual value: {D_4_f}")
    print(f"Expected value for 5D Hpersphere radius of {R1[0]}: {hs_5}")
    print(f"Actual value:{D_5_f}") 

R = np.repeat(4, 5)
variance = np.array([3, 0.6])
gaussian = Monte_Carlo(-R, R, N, gauss_function, *variance)
g = Monte_Carlo.infinity(gaussian)


if rank==0:
    print(g)
MPI.Finalize()
