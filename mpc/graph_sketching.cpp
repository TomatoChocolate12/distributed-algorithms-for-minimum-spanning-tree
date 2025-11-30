#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>

// --- Structures ---

struct Edge {
    int u, v;
    int weight;
    int id; // Unique ID for sketching/logic

    // Overload < for sorting
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

// --- DSU with Sketching Logic ---
// This handles the "Component Merging" and "XOR Logic"
struct ComponentTracker {
    std::vector<int> parent;
    std::vector<uint64_t> sketch_signature; // The "Binary Logic" part

    ComponentTracker(int n) {
        parent.resize(n);
        sketch_signature.resize(n);
        
        // Random number generator for component signatures
        std::mt19937_64 rng(12345); 
        std::uniform_int_distribution<uint64_t> dist;

        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            // Every node starts as its own component with a unique random binary signature
            sketch_signature[i] = dist(rng);
        }
    }

    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }

    // Returns true if merged, false if already connected
    bool union_sets(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);

        if (root_i != root_j) {
            // --- THE SKETCHING LOGIC ---
            // When two components merge, we XOR their signatures.
            // In full AGM sketching, this XORing cancels out internal edges.
            // Here, it merges the "identity" of the two components.
            uint64_t new_sig = sketch_signature[root_i] ^ sketch_signature[root_j];
            
            // Standard DSU merge
            parent[root_i] = root_j;
            
            // Update the sketch of the new root
            sketch_signature[root_j] = new_sig;
            return true;
        }
        return false;
    }
    
    // Check using the sketching logic (Demonstration purpose)
    // If sketches are different, they MIGHT be different components. 
    // (In reality we rely on find() for absolute correctness in this deterministic code)
    bool have_different_sketches(int i, int j) {
        return sketch_signature[find(i)] != sketch_signature[find(j)];
    }
};

// --- Main Program ---

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Graph Parameters
    const int NUM_NODES = 20;
    const int NUM_EDGES = 60; // Usually m >> n
    
    std::vector<Edge> all_edges;
    std::vector<Edge> mst_edges;
    long long mst_weight = 0;

    // --- STEP 1: SORTING (Simulated on Rank 0) ---
    // In a real MPC cluster, this is a distributed sort (e.g., TeraSort).
    if (world_rank == 0) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> node_dist(0, NUM_NODES - 1);
        std::uniform_int_distribution<int> weight_dist(1, 100);

        std::cout << "[Rank 0] Generating and Sorting " << NUM_EDGES << " edges..." << std::endl;
        
        for (int i = 0; i < NUM_EDGES; ++i) {
            Edge e;
            e.u = node_dist(rng);
            e.v = node_dist(rng);
            // Avoid self-loops for simplicity
            while(e.u == e.v) e.v = node_dist(rng);
            e.weight = weight_dist(rng);
            e.id = i;
            all_edges.push_back(e);
        }

        // Sort edges by weight (The fundamental step of Kruskal's / Filter-Kruskal)
        std::sort(all_edges.begin(), all_edges.end());
    }

    // Initialize Component Tracker (Graph Sketch / DSU)
    // In strict linear memory MPC, this state is distributed. 
    // Here we replicate the state table for simplicity of the MPI code.
    ComponentTracker tracker(NUM_NODES);

    // --- STEP 2: CHUNKING & PROCESSING ---
    // We break edges into chunks of size 'NUM_NODES' (or any batch size)
    int chunk_size = NUM_NODES; 
    int num_chunks = (NUM_EDGES + chunk_size - 1) / chunk_size;

    // Broadcast number of chunks to all workers
    MPI_Bcast(&num_chunks, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < num_chunks; ++i) {
        std::vector<Edge> current_chunk;
        int current_chunk_size = 0;

        // Rank 0 prepares the chunk
        if (world_rank == 0) {
            int start_idx = i * chunk_size;
            int end_idx = std::min((int)all_edges.size(), start_idx + chunk_size);
            
            for(int k=start_idx; k<end_idx; ++k) {
                current_chunk.push_back(all_edges[k]);
            }
            current_chunk_size = current_chunk.size();
            
            std::cout << "[Rank 0] Processing Chunk " << i+1 << "/" << num_chunks 
                      << " (Size: " << current_chunk_size << ")" << std::endl;
        }

        // Broadcast chunk size
        MPI_Bcast(&current_chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Resize buffer on workers
        if (world_rank != 0) current_chunk.resize(current_chunk_size);

        // Broadcast the actual edge data (simplified as struct of ints)
        // Note: In production MPI, you'd create a derived MPI_Datatype for 'Edge'
        // Here we just broadcast the raw bytes since it's POD (Plain Old Data)
        MPI_Bcast(current_chunk.data(), current_chunk_size * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);

        // --- STEP 3: "GRAPH SKETCHING" / CONNECTIVITY CHECK ---
        // Every rank processes the chunk. 
        // In this algorithm, we iterate through the chunk and decide connectivity.
        
        // Note: In a true massive parallel setup, different ranks would own different 
        // parts of the component vector. Here, we simulate the logic.
        
        for (const auto& edge : current_chunk) {
            // The "Observation": Check connectivity based on previous chunks.
            // We verify if u and v are in different components using our tracker.
            
            // Log the "Binary Logic" check
            // logic: If Sk(u) XOR Sk(v) != 0, they *might* be connected, 
            // but if find(u) != find(v), they are definitely separate.
            
            if (tracker.find(edge.u) != tracker.find(edge.v)) {
                // If Rank 0, actually record the MST edge
                if (world_rank == 0) {
                    mst_edges.push_back(edge);
                    mst_weight += edge.weight;
                    /* 
                       Visualization of the "Sketching" Step:
                       We are effectively XORing the component signatures of U and V.
                       This mimics mathematically cancelling out the internal path constraints.
                    */
                }
                
                // All ranks must update their local tracker to stay in sync
                // This corresponds to "Contracting" the graph nodes.
                tracker.union_sets(edge.u, edge.v);
            }
        }
        
        // Synchronization barrier to ensure all ranks finished the chunk
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- Output Results ---
    if (world_rank == 0) {
        std::cout << "--------------------------------" << std::endl;
        std::cout << "MST Computation Complete." << std::endl;
        std::cout << "Total MST Weight: " << mst_weight << std::endl;
        std::cout << "Edges in MST: " << mst_edges.size() << std::endl;
        std::cout << "Sample MST Edges:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)5, mst_edges.size()); ++i) {
            std::cout << "  (" << mst_edges[i].u << ", " << mst_edges[i].v 
                      << ") W:" << mst_edges[i].weight << std::endl;
        }
        std::cout << "--------------------------------" << std::endl;
    }

    MPI_Finalize();
    return 0;
}