#!/bin/bash

#SBATCH --job-name=hw3p2
#SBATCH --partition=shared
#SBATCH -n 14
#SBATCH --time=00:10:00

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current WORKING directory
cd /home/ers131/Machine-Learning-Projects/HW/HW3/SmithEvanEE4331HW3

python Part2/hw3-perceptron.py

echo "Program finished with exit code $? at: `date`"
