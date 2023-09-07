#!/bin/bash

#SBATCH --job-name=log-reg-hw1p1
#SBATCH --partition=shared
#SBATCH -n 1
#SBATCH --time=00:10:00

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current directory
cd /home/ers131/SmithEvanEE4331HW1Submit/Part2

python hw1-log-reg.py

echo "Program finished with exit code $? at: `date`"
