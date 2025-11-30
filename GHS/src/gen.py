import numpy as np
import networkx as nx
import json  # Changed from pickle
import sys

def generate_distributed_graph(num_nodes, density=0.3, output_file="graph_dist.json"):
    print(f"Generating graph with {num_nodes} nodes...")
    
    # 1. Generate Graph
    while True:
        G = nx.erdos_renyi_graph(num_nodes, density)
        if nx.is_connected(G):
            break
            
    # 2. Assign Weights
    weights = np.random.choice(range(1, num_nodes*100), size=G.number_of_edges(), replace=False)
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['weight'] = int(weights[i]) # Ensure python int for JSON serialization

    # 3. Partition for MPI
    distributed_data = {}
    
    for node in G.nodes():
        neighbors = {}
        for nbr in G.neighbors(node):
            neighbors[int(nbr)] = G[node][nbr]['weight']
        distributed_data[int(node)] = neighbors

    # 4. Save as JSON
    with open(output_file, 'w') as f:
        # indent=2 makes it readable for you in text editors
        json.dump(distributed_data, f, indent=2) 
        
    print(f"Graph saved to {output_file}. Edges: {G.number_of_edges()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen.py <num_nodes> [output_file]")
        sys.exit(1)
    
    N = int(sys.argv[1])
    fname = sys.argv[2] if len(sys.argv) > 2 else "graph_dist.json"
    generate_distributed_graph(N, output_file=fname)
