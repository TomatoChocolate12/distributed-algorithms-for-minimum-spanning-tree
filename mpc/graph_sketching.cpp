#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <cstddef>

// --- Configuration ---
// Levels for guess k (cut size). 32 covers up to 4 billion edges.
const int SKETCH_LEVELS = 32; 

// "Amplified Repetitions": The number of independent trials/blocks.
// The paper suggests O(log n), but 5-10 is robust for practical sizes.
const int NUM_ROWS = 5; 

// --- Hashing & Utils ---

// A robust hash function that takes a seed (row index) to ensure independence
inline uint64_t hash_fn(uint64_t x, uint64_t seed) {
    // Mix the input with the seed
    x ^= seed * 0x517cc1b727220a95; 
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccd;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53;
    x ^= x >> 33;
    return x;
}

// Determines level based on trailing zeros of the hash
inline int get_sampling_level(int edge_id, int row_seed) {
    uint64_t h = hash_fn((uint64_t)edge_id, (uint64_t)row_seed);
    int level = 0;
    while (((h >> level) & 1) == 0 && level < SKETCH_LEVELS - 1) {
        level++;
    }
    return level;
}

// --- Data Structures ---
struct Edge {
    int u, v;
    int weight;
    int id; 

    bool operator<(const Edge& other) const {
        if (weight != other.weight) return weight < other.weight;
        return id < other.id;
    }
};

// --- AGM Sketch (L0-Sampler with Repetitions) ---
struct AgmSketch {
    // [ROW][LEVEL]
    // We maintain 'NUM_ROWS' independent sketches for probability amplification
    uint64_t id_sum[NUM_ROWS][SKETCH_LEVELS];
    uint64_t fp_sum[NUM_ROWS][SKETCH_LEVELS];

    AgmSketch() {
        std::memset(id_sum, 0, sizeof(id_sum));
        std::memset(fp_sum, 0, sizeof(fp_sum));
    }

    // Update all independent rows
    void update(int edge_id) {
        for(int r = 0; r < NUM_ROWS; ++r) {
            // Each row uses 'r' as a seed to sample edges differently
            int lvl = get_sampling_level(edge_id, r);
            uint64_t fp = hash_fn((uint64_t)edge_id, (uint64_t)r);
            
            // Standard L0 update: Add to all levels <= sampled level
            for (int i = 0; i <= lvl; ++i) {
                id_sum[r][i] ^= edge_id;
                fp_sum[r][i] ^= fp;
            }
        }
    }

    // Merge super-nodes by XORing their matrices
    void merge(const AgmSketch& other) {
        for (int r = 0; r < NUM_ROWS; ++r) {
            for (int i = 0; i < SKETCH_LEVELS; ++i) {
                id_sum[r][i] ^= other.id_sum[r][i];
                fp_sum[r][i] ^= other.fp_sum[r][i];
            }
        }
    }

    // Query: Try to find a valid edge. Check Row 0, then Row 1...
    int query() const {
        for (int r = 0; r < NUM_ROWS; ++r) {
            for (int i = SKETCH_LEVELS - 1; i >= 0; --i) {
                uint64_t id = id_sum[r][i];
                
                if (id == 0) continue; // Empty level
                
                // Verification: Does Hash(ID) match the Fingerprint Sum?
                // If yes, we successfully isolated exactly one edge in this bucket.
                if (hash_fn(id, (uint64_t)r) == fp_sum[r][i]) {
                    return (int)id;
                }
                // If no, a collision occurred (multiple edges). 
                // We simply move to the next level or the next independent row.
            }
        }
        return -1; // Failed to isolate an edge in any row
    }
};

struct DSU {
    std::vector<int> parent;
    DSU(int n) {
        parent.resize(n);
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    void unite(int i, int j) {
        parent[find(i)] = find(j);
    }
};

// --- MPI Helper ---
void register_mpi_type(MPI_Datatype* t) {
    const int n = 4;
    int blocks[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[4] = {
        offsetof(Edge, u), offsetof(Edge, v), 
        offsetof(Edge, weight), offsetof(Edge, id)
    };
    MPI_Type_create_struct(n, blocks, offsets, types, t);
    MPI_Type_commit(t);
}

// --- MPC Distributed Sort (Sample Sort) ---
void mpc_sort(std::vector<Edge>& local_data, int rank, int size, MPI_Datatype type) {
    std::sort(local_data.begin(), local_data.end());

    int num_samples = size; 
    std::vector<Edge> local_samples;
    local_samples.reserve(num_samples);
    
    if (local_data.empty()) {
        for(int i=0; i<num_samples; ++i) local_samples.push_back({0,0,INT32_MAX,0});
    } else {
        for(int i=0; i<num_samples; ++i) {
            local_samples.push_back(local_data[(i * local_data.size()) / num_samples]);
        }
    }

    std::vector<Edge> all_samples;
    if (rank == 0) all_samples.resize(num_samples * size);
    
    MPI_Gather(local_samples.data(), num_samples, type, 
               all_samples.data(), num_samples, type, 0, MPI_COMM_WORLD);

    std::vector<int> splitters(size - 1);
    if (rank == 0) {
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 0; i < size - 1; ++i) {
            splitters[i] = all_samples[(i + 1) * num_samples].weight;
        }
    }
    MPI_Bcast(splitters.data(), size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::vector<Edge>> buckets(size);
    for (const auto& e : local_data) {
        int target = 0;
        while (target < size - 1 && e.weight > splitters[target]) target++;
        buckets[target].push_back(e);
    }

    std::vector<Edge> send_buf;
    std::vector<int> s_counts(size), s_displs(size), r_counts(size), r_displs(size);
    int offset = 0;
    
    for (int i = 0; i < size; ++i) {
        s_counts[i] = buckets[i].size();
        s_displs[i] = offset;
        send_buf.insert(send_buf.end(), buckets[i].begin(), buckets[i].end());
        offset += buckets[i].size();
    }

    MPI_Alltoall(s_counts.data(), 1, MPI_INT, r_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = 0;
    for (int i = 0; i < size; ++i) {
        r_displs[i] = total_recv;
        total_recv += r_counts[i];
    }

    std::vector<Edge> recv_buf(total_recv);
    MPI_Alltoallv(send_buf.data(), s_counts.data(), s_displs.data(), type,
                  recv_buf.data(), r_counts.data(), r_displs.data(), type, MPI_COMM_WORLD);

    local_data = std::move(recv_buf);
    std::sort(local_data.begin(), local_data.end());
}

// --- Main Execution ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " <graph_file> <num_nodes>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string filename = argv[1];
    int num_nodes = std::atoi(argv[2]);

    MPI_Datatype mpi_edge_type;
    register_mpi_type(&mpi_edge_type);

    // --- Phase 1: Load and Scatter ---
    // (In production, use MPI_File_read_at for parallel IO)
    std::vector<Edge> local_edges;
    
    // Simple scatter from Root (OK for < 100GB graphs)
    std::vector<int> send_counts(size);
    std::vector<Edge> input_buffer;

    if (rank == 0) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int u, v, w, id=1;
        while(file >> u >> v >> w) input_buffer.push_back({u,v,w,id++});
        
        int total = input_buffer.size();
        int rem = total % size;
        for(int i=0; i<size; ++i) send_counts[i] = total/size + (i < rem ? 1:0);
        
        std::cout << "[Rank 0] Loaded " << total << " edges. Distributing..." << std::endl;
    }
    
    int my_count;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &my_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    local_edges.resize(my_count);
    
    std::vector<int> displs(size);
    if(rank==0) {
        int sum=0; 
        for(int i=0; i<size; ++i) { displs[i]=sum; sum+=send_counts[i]; }
    }

    MPI_Scatterv(input_buffer.data(), send_counts.data(), displs.data(), mpi_edge_type,
                 local_edges.data(), my_count, mpi_edge_type, 0, MPI_COMM_WORLD);

    // --- Phase 2: Distributed Sort ---
    if(rank == 0) std::cout << "[Cluster] Sorting edges globally..." << std::endl;
    mpc_sort(local_edges, rank, size, mpi_edge_type);

    // --- Phase 3: AGM Sketching w/ Repetitions ---
    // State size: Nodes * Rows * Levels * 16 bytes.
    // For 1M nodes, 5 rows, 32 levels ~ 2.5 GB RAM. Fits easily on server nodes.
    std::vector<AgmSketch> sketches(num_nodes);
    DSU dsu(num_nodes);
    std::unordered_map<int, Edge> edge_lookup; 
    
    long long mst_weight = 0;
    int mst_edge_count = 0;

    // Process Chunks
    for (int r = 0; r < size; ++r) {
        int chunk_size = 0;
        if (rank == r) chunk_size = local_edges.size();
        MPI_Bcast(&chunk_size, 1, MPI_INT, r, MPI_COMM_WORLD);

        std::vector<Edge> chunk(chunk_size);
        if (rank == r) chunk = local_edges;
        MPI_Bcast(chunk.data(), chunk_size, mpi_edge_type, r, MPI_COMM_WORLD);

        // Update 
        for (const auto& e : chunk) {
            edge_lookup[e.id] = e; // Cache for recovery
            int root_u = dsu.find(e.u);
            int root_v = dsu.find(e.v);
            
            if (root_u != root_v) {
                sketches[root_u].update(e.id);
                sketches[root_v].update(e.id);
            }
        }

        // Boruvka Step
        bool merged = true;
        while (merged) {
            merged = false;
            for (int i = 0; i < num_nodes; ++i) {
                int root = dsu.find(i);
                if (root != i) continue;

                // Query checks all 'NUM_ROWS' independent sketches
                int id = sketches[root].query();
                
                if (id != -1) {
                    auto it = edge_lookup.find(id);
                    if (it != edge_lookup.end()) {
                        Edge e = it->second;
                        int r_u = dsu.find(e.u);
                        int r_v = dsu.find(e.v);

                        if (r_u != r_v) {
                            if (rank == 0) {
                                mst_weight += e.weight;
                                mst_edge_count++;
                            }
                            
                            sketches[r_u].merge(sketches[r_v]);
                            dsu.unite(r_u, r_v); 
                            sketches[r_v] = AgmSketch(); // Clear
                            merged = true;
                        }
                    }
                }
            }
        }
    }

    if (rank == 0) {
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "MST Weight: " << mst_weight << std::endl;
        std::cout << "Edges in MST: " << mst_edge_count << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }

    MPI_Type_free(&mpi_edge_type);
    MPI_Finalize();
    return 0;
}