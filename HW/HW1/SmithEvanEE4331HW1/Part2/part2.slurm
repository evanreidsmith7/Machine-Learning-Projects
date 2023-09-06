#!/bin/bash

#SBATCH --job-name=log-reg-hw1p1
#SBATCH --partition=shared
#SBATCH -n 1
#SBATCH --time=00:10:00

# Specify the full path to your Python script
PYTHON_SCRIPT="/home/ers131/Machine-Learning-Projects/HW/HW1/SmithEvanEE4331HW1/Part2/hw1-log-reg.py"

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current directory
cd /home/ers131/Machine-Learning-Projects

python $PYTHON_SCRIPT

echo "Program finished with exit code $? at: `date`"
