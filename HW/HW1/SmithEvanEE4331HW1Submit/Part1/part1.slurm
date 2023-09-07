#!/bin/bash

#SBATCH --job-name=perceptronhw1p1
#SBATCH --partition=shared
#SBATCH -n 1
#SBATCH --time=00:10:00

# Specify the full path to your Python script
PYTHON_SCRIPT="/home/ers131/SmithEvanEE4331HW1Submit/Part1/hw1-perceptron.py"

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current WORKING directory
cd /home/ers131/SmithEvanEE4331HW1Submit/Part1

python hw1-perceptron.py

echo "Program finished with exit code $? at: `date`"
