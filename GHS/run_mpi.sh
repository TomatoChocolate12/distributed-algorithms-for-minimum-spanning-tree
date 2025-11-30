#!/bin/bash
#SBATCH --job-name=mpi_ghs
#SBATCH --partition=debug
#SBATCH --time=00:20:00 
#SBATCH --ntasks=50

set -euo pipefail

NUM_PROCS=50

# 1. Load the OpenMPI module
module load openmpi/4.1.5

# 2. Check for nlohmann/json library
if [ ! -f "json.hpp" ]; then
    echo "Downloading json.hpp..."
    wget -q https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
fi

# 4. Compile
echo "Compiling..."
mpicxx -std=c++17 -O3 ghs_mpi.cpp -o ghs_algo

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Running GHS with $NUM_PROCS processes..."
mpirun -np $NUM_PROCS ./ghs_algo inp/${NUM_PROCS}.json

echo "Done."