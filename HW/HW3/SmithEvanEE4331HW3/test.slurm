#!/bin/bash

#SBATCH --job-name=hw3p1plot
#SBATCH --partition=himem
#SBATCH -n 20
#SBATCH --time=05:00:00

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current WORKING directory
cd /home/ers131/Machine-Learning-Projects/HW/HW3/SmithEvanEE4331HW3

python Part1/TestScripts/hw3-part1-plottest.py

echo "Program finished with exit code $? at: `date`"