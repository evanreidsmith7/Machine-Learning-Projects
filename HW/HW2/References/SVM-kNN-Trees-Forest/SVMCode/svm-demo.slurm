#!/bin/bash

#SBATCH --job-name=svm-demo
#SBATCH --partition=shared
#SBATCH -n 1
#SBATCH --time=00:10:00

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# go to my current directory
cd /home/ers131/MLdemo/L5/CodeExamples

python svm-demo.py

echo "Program finished with exit code $? at: `date`"