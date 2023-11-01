#!/bin/bash

#SBATCH --job-name=hw5
#SBATCH --partition=shared
#SBATCH -n 24
#SBATCH --time=00:30:00

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current WORKING directory
cd /home/ers131/Machine-Learning-Projects/HW/HW5/SmithEvanEE4331HW5

python hw5-dcnn.py

echo "Program finished with exit code $? at: `date`"
