#!/bin/bash

#SBATCH --job-name=knnhw1p1
#SBATCH --partition=shared
#SBATCH -n 14
#SBATCH --time=00:10:00

# Specify the full path to your Python script
PYTHON_SCRIPT="hw2-knn.py"

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current WORKING directory
cd /home/ers131/SmithEvanEE4331HW2Submit

python hw2-knn.py

echo "Program finished with exit code $? at: `date`"
