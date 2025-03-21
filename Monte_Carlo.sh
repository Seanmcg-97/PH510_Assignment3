#!/bin/bash

#======================================================
#
# Job script for running GROMACS on a single node
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required (max. of 40)
#SBATCH --ntasks=4
#SBATCH --distribution=block:block
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=02:00:00
#
# Job name
#SBATCH --job-name=Monte_Carlo_test
#
# Output file
#SBATCH --output=Monte_Carlo-%j.out
#======================================================

module purge
module add miniconda/3.12.8
module load gromacs/intel-2020.4/2020.3-single
module load openmpi/gcc-8.5.0/4.1.1



#======================================================
# Prologue script to record job details
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

#export OMP_NUM_THREADS=1

mpirun -np 4 python3 ./class_1.py



#======================================================
# Epilogue script to record job endtime and runtime
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
