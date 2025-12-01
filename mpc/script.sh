#!/bin/bash
#SBATCH --job-name=mst_mpc_linear
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=mst_output_%j.txt
#SBATCH --partition=debug   # Check your cluster documentation for partition names

# Load MPI Module (Adjust based on your cluster, e.g., 'module load openmpi/4.1.1')
module load openmpi 
# Or sometimes: module load intel-mpi

# Define file names
SRC_FILE="graph_sketching.cpp"
EXE_FILE="graph_sketching"
GRAPH_FILE="edges_rand_dense.txt"
NODES=1000
EDGES=15000

echo "=========================================="
echo "Running on hosts: $(hostname)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total Ranks: $SLURM_NTASKS"
echo "=========================================="

# 1. Compile
echo "[Step 1] Compiling C++ code..."
mpic++ -O3 -std=c++11 $SRC_FILE -o $EXE_FILE

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 2. Generate Synthetic Graph (Python one-liner)
# Format: u v w
# echo "[Step 2] Generating random graph with $NODES nodes and $EDGES edges..."
# python3 -c "
# import random
# nodes = $NODES
# edges = $EDGES
# with open('$GRAPH_FILE', 'w') as f:
#     for _ in range(edges):
#         u = random.randint(0, nodes-1)
#         v = random.randint(0, nodes-1)
#         while u == v: v = random.randint(0, nodes-1)
#         w = random.randint(1, 1000)
#         f.write(f'{u} {v} {w}\n')
# "

# 3. Run MPI Job
echo "[Step 3] Executing MST Algorithm..."
# mpirun knows to check SLURM environment variables for node lists
mpirun ./$EXE_FILE $GRAPH_FILE $NODES

echo "=========================================="
echo "Job Complete."