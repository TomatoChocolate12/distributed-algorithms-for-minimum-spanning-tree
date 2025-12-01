#!/bin/bash
#SBATCH --job-name=mst_mpc_linear
#SBATCH --nodes=1
#SBATCH -c 40
#SBATCH --time=00:10:00
#SBATCH --output=mst_output_%j.txt
#SBATCH --partition=debug   # Check your cluster documentation for partition names

# Load MPI Module (Adjust based on your cluster, e.g., 'module load openmpi/4.1.1')
module load openmpi/4.1.0
# Or sometimes: module load intel-mpi

# Define file names
# Define files
SRC="graph_sketching.cpp"
EXE="graph_sketching"
GRAPH="graph.txt"

# 1. Compile
mpic++ -O3 -std=c++11 $SRC -o $EXE

# Write CSV Header to output file
echo "Experiment_Type,Processors,Nodes,Edges,SortTime,SketchTime,TotalTime,TotalBytes"

# --- EXPERIMENT 1: Time vs Number of Nodes (Scaling N) ---
# Fixed Processors (P=20), varying Nodes
P_FIXED=40
echo "Running Node Scaling Experiments (P=$P_FIXED)..." >&2

for N in 1000 5000 10000 20000 50000; do
    # Generate graph with M = 10 * N
    M=$((N * 10))
    python3 -c "import random; n=$N; m=$M; print('\n'.join(f'{random.randint(0,n-1)} {random.randint(0,n-1)} {random.randint(1,1000)}' for _ in range(m)))" > $GRAPH
    
    # Run and prepend tag "NodeScaling"
    mpirun -np $P_FIXED ./$EXE $GRAPH $N | grep "RESULTS" | sed "s/RESULTS,/NodeScaling,/"
done

# --- EXPERIMENT 2: Time vs Number of Processors (Scaling P) ---
# Fixed Nodes (N=20000, M=200000), varying Processors
N_FIXED=20000
M_FIXED=200000
python3 -c "import random; n=$N_FIXED; m=$M_FIXED; print('\n'.join(f'{random.randint(0,n-1)} {random.randint(0,n-1)} {random.randint(1,1000)}' for _ in range(m)))" > $GRAPH

echo "Running Processor Scaling Experiments (N=$N_FIXED)..." >&2

for P in 4 8 16 32 40; do
    # Run and prepend tag "ProcScaling"
    mpirun -np $P ./$EXE $GRAPH $N_FIXED | grep "RESULTS" | sed "s/RESULTS,/ProcScaling,/"
done

# Cleanup
rm $GRAPH
rm $EXE