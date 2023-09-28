#!/bin/bash

#SBATCH --job-name=dfgh
#SBATCH --partition=shared
#SBATCH -n 1
#SBATCH --time=00:10:00


echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo "hello"

# go to my current WORKING directory
cd /home/ers131/hwsubtest

python --version

python part1/pt2.py

#echo "Program finished with exit code $? at: `date`"
