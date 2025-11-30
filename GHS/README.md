# GHS - Distributed Minimum Spanning Tree Algorithm

A distributed implementation of the **Gallager-Humblet-Spira (GHS) algorithm** for computing Minimum Spanning Trees (MST) using MPI (Message Passing Interface).

## Overview

This project implements the GHS algorithm, a classic distributed algorithm for finding the minimum spanning tree of a weighted undirected graph. The implementation uses MPI for inter-process communication, where each node in the graph is represented by a separate MPI process.

## Project Structure

```
GHS/
├── src/                    # Source code
│   ├── ghs_mpi.cpp        # Main GHS algorithm implementation using MPI
│   ├── gen.py             # Graph generator for creating test inputs
│   ├── kruskals.cpp       # Kruskal's algorithm (sequential MST for verification)
│   ├── verifier.cpp       # Verifies correctness of GHS output
│   └── json.hpp           # JSON parsing library (nlohmann/json)
├── inp/                    # Input graph files (JSON format)
│   ├── 5.json             # Graph with 5 nodes
│   ├── 10.json            # Graph with 10 nodes
│   └── ...                # Graphs up to 150 nodes
├── logs/                   # Execution logs and results
│   ├── *.txt              # Log files for each graph size
│   └── extract_info.py    # Script to extract metrics from logs
├── run_mpi.sh             # Shell script to run MPI experiments
├── plot.py                # Plotting script for visualization
├── results.csv            # Collected experimental results
├── plot_*.png             # Generated performance plots
└── README.md
```

## Dependencies

- **MPI Implementation** (OpenMPI or MPICH)
- **C++11** or later
- **Python 3** with the following packages:
  - `numpy`
  - `networkx`
  - `matplotlib`
- **nlohmann/json** (included as `json.hpp`)

### Installing Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install mpich libmpich-dev g++

# Python dependencies
pip install numpy networkx matplotlib
```

## Building

Compile the GHS MPI implementation:

```bash
mpicxx -o ghs_mpi src/ghs_mpi.cpp -std=c++11
```

Compile the verification tools:

```bash
g++ -o kruskals src/kruskals.cpp -std=c++11
g++ -o verifier src/verifier.cpp -std=c++11
```

## Usage

### 1. Generating Input Graphs

Use `gen.py` to generate random connected weighted graphs with unique edge weights:

```bash
python src/gen.py <num_nodes> [output_file]

# Examples:
python src/gen.py 10 inp/10.json
python src/gen.py 50 inp/50.json
```

The generator creates:
- A connected Erdős-Rényi random graph
- **Unique edge weights** (required for GHS algorithm correctness)
- MST reference file: `<num_nodes>_GHS.txt`

### 2. Running the GHS Algorithm

Execute the distributed algorithm using MPI:

```bash
mpirun -np <num_processes> ./ghs_mpi <input_file.json>

# Example:
mpirun -np 10 ./ghs_mpi inp/10.json
```

**Note:** The number of MPI processes should match the number of nodes in the graph.

Or use the provided script to run batch experiments:

```bash
chmod +x run_mpi.sh
./run_mpi.sh
```

### 3. Verifying Results

Compare GHS output against Kruskal's sequential algorithm:

```bash
# Run Kruskal's algorithm
./kruskals inp/10.json

# Verify GHS output
./verifier <ghs_output> <kruskals_output>
```

## Input Format

Input graphs are specified in JSON format. Each node maps to its neighbors with edge weights:

```json
{
  "0": {"1": 10, "2": 25, "4": 15},
  "1": {"0": 10, "2": 5, "3": 20},
  "2": {"0": 25, "1": 5, "3": 8},
  "3": {"1": 20, "2": 8, "4": 12},
  "4": {"0": 15, "3": 12}
}
```

**Important:** All edge weights must be unique for the GHS algorithm to work correctly.

## Output Format

The MST output file (`<num_nodes>_GHS.txt`) contains:

```
Minimum Spanning Tree for 10 nodes
Total MST weight: 145
MST edges (9 edges):
0 -- 1 : 10
1 -- 2 : 5
...
```

## Algorithm Details

### How GHS Works

The GHS algorithm is a distributed algorithm where each node only knows about its local edges and communicates with neighbors via message passing.

1. **Initialization:** Each node starts as its own fragment (a subtree of the MST)
2. **Find MOE:** Each fragment finds its Minimum Outgoing Edge (MOE) - the minimum weight edge connecting it to another fragment
3. **Merge:** Fragments merge along their MOEs
4. **Repeat:** The process continues until only one fragment remains
5. **Result:** The edges used for merging form the MST

### Message Types

| Message | Description |
|---------|-------------|
| `CONNECT` | Request to connect fragments |
| `INITIATE` | Start a new phase in a fragment |
| `TEST` | Test if an edge is outgoing |
| `ACCEPT` | Edge is outgoing (connects different fragments) |
| `REJECT` | Edge is internal (same fragment) |
| `REPORT` | Report minimum outgoing edge weight |
| `CHANGE_ROOT` | Change the root of the fragment |

### Complexity Analysis

| Metric | Complexity |
|--------|------------|
| **Message Complexity** | O(E + N log N) |
| **Time Complexity** | O(N log N) |
| **Space per Node** | O(degree) |

Where:
- N = number of nodes
- E = number of edges

## Performance Analysis

The project includes tools for analyzing algorithm performance:

### Extracting Metrics

```bash
cd logs
python extract_info.py
```

### Generating Plots

```bash
python plot.py
```

### Generated Visualizations

| File | Description |
|------|-------------|
| `plot_messages.png` | Message complexity vs graph size |
| `plot_time_taken.png` | Execution time vs graph size |
| `plot_data_sent.png` | Communication overhead analysis |

## Experimental Results

Results are stored in `results.csv` with the following columns:
- `nodes` - Number of nodes in the graph
- `edges` - Number of edges in the graph
- `messages` - Total messages exchanged
- `time_ms` - Execution time in milliseconds
- `data_bytes` - Total data transferred

## Troubleshooting

### Common Issues

1. **Graph not connected**
   ```
   Error: Graph is not connected
   ```
   Solution: Regenerate the graph or increase density parameter in `gen.py`

2. **Duplicate edge weights**
   ```
   Error: Non-unique edge weights detected
   ```
   Solution: Use the updated `gen.py` which ensures unique weights

3. **MPI process count mismatch**
   ```
   Error: Number of processes doesn't match nodes
   ```
   Solution: Ensure `-np` matches the number of nodes in your graph

4. **Deadlock**
   - Check that all nodes are participating
   - Verify the input graph is correctly formatted

### Debugging

Enable verbose logging by modifying `ghs_mpi.cpp`:
```cpp
#define DEBUG 1
```

## References

1. Gallager, R. G., Humblet, P. A., & Spira, P. M. (1983). *"A Distributed Algorithm for Minimum-Weight Spanning Trees"*. ACM Transactions on Programming Languages and Systems (TOPLAS), 5(1), 66-77.

2. Nancy A. Lynch (1996). *"Distributed Algorithms"*. Morgan Kaufmann Publishers.

## License

This project is for educational purposes. See the root [`LICENSE`](../LICENSE) file for details.

## Authors

Distributed Algorithms Course Project

---

## Quick Start

```bash
# 1. Build
mpicxx -o ghs_mpi src/ghs_mpi.cpp -std=c++11

# 2. Generate a graph with 10 nodes
python src/gen.py 10 inp/10.json

# 3. Run GHS algorithm
mpirun -np 10 ./ghs_mpi inp/10.json

# 4. Check the output
cat 10_GHS.txt
```