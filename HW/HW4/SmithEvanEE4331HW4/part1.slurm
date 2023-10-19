#!/bin/bash

#SBATCH --job-name=hw4p1
#SBATCH --partition=parallel
#SBATCH -n 20
#SBATCH --time=00:30:00

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current WORKING directory
cd /home/ers131/Machine-Learning-Projects/HW/HW4/SmithEvanEE4331HW4

python Part1/hw4-part1.py

echo "Program finished with exit code $? at: `date`"
