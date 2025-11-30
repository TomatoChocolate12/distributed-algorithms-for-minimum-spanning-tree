#!/bin/bash
#SBATCH --job-name=mpc
#SBATCH --output=mpc%j.out
#SBATCH --error=mpc%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16

module load python/3.12.5

mpiexec -n 4 graph_sketching.py
