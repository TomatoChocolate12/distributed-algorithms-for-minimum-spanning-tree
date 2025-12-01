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
#include <iomanip>

// --- Configuration ---
const int SKETCH_LEVELS = 32; 
const int NUM_ROWS = 5; 

// --- Metrics ---
long long local_bytes_sent = 0;

// --- Hashing ---
inline uint64_t hash_fn(uint64_t x, uint64_t seed) {
    x ^= seed * 0x517cc1b727220a95; 
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccd;
    x ^= x >> 33;
    return x;
}

inline int get_sampling_level(int edge_id, int row_seed) {
    uint64_t h = hash_fn((uint64_t)edge_id, (uint64_t)row_seed);
    int level = 0;
    while (((h >> level) & 1) == 0 && level < SKETCH_LEVELS - 1) {
        level++;
    }
    return level;
}

// --- Structures ---
struct Edge {
    int u, v;
    int weight;
    int id; 
    bool operator<(const Edge& other) const {
        if (weight != other.weight) return weight < other.weight;
        return id < other.id;
    }
};

struct AgmSketch {
    uint64_t id_sum[NUM_ROWS][SKETCH_LEVELS];
    uint64_t fp_sum[NUM_ROWS][SKETCH_LEVELS];

    AgmSketch() {
        std::memset(id_sum, 0, sizeof(id_sum));
        std::memset(fp_sum, 0, sizeof(fp_sum));
    }

    void update(int edge_id) {
        for(int r = 0; r < NUM_ROWS; ++r) {
            int lvl = get_sampling_level(edge_id, r);
            uint64_t fp = hash_fn((uint64_t)edge_id, (uint64_t)r);
            for (int i = 0; i <= lvl; ++i) {
                id_sum[r][i] ^= edge_id;
                fp_sum[r][i] ^= fp;
            }
        }
    }

    void merge(const AgmSketch& other) {
        for(int r=0; r<NUM_ROWS; ++r) {
            for(int l=0; l<SKETCH_LEVELS; ++l) {
                id_sum[r][l] ^= other.id_sum[r][l];
                fp_sum[r][l] ^= other.fp_sum[r][l];
            }
        }
    }

    int query() const {
        for (int r = 0; r < NUM_ROWS; ++r) {
            for (int i = SKETCH_LEVELS - 1; i >= 0; --i) {
                uint64_t id = id_sum[r][i];
                if (id == 0) continue; 
                if (hash_fn(id, (uint64_t)r) == fp_sum[r][i]) return (int)id;
            }
        }
        return -1; 
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

// --- MPC Distributed Sort (Instrumented) ---
void mpc_sort(std::vector<Edge>& local_data, int rank, int size, MPI_Datatype type) {
    std::sort(local_data.begin(), local_data.end());

    int num_samples = size; 
    std::vector<Edge> local_samples;
    if(!local_data.empty()) {
        for(int i=0; i<num_samples; ++i) local_samples.push_back(local_data[(i * local_data.size()) / num_samples]);
    } else {
        local_samples.resize(num_samples, {0,0,INT32_MAX,0});
    }

    // Measure Gather
    local_bytes_sent += num_samples * sizeof(Edge);
    std::vector<Edge> all_samples;
    if (rank == 0) all_samples.resize(num_samples * size);
    MPI_Gather(local_samples.data(), num_samples, type, all_samples.data(), num_samples, type, 0, MPI_COMM_WORLD);

    std::vector<int> splitters(size - 1);
    if (rank == 0) {
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 0; i < size - 1; ++i) splitters[i] = all_samples[(i + 1) * num_samples].weight;
    }
    
    // Measure Bcast
    if(rank == 0) local_bytes_sent += (size - 1) * sizeof(int); 
    MPI_Bcast(splitters.data(), size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::vector<Edge>> buckets(size);
    for (const auto& e : local_data) {
        int target = 0;
        while (target < size - 1 && e.weight > splitters[target]) target++;
        buckets[target].push_back(e);
    }

    std::vector<Edge> send_buf; 
    std::vector<int> sc(size), sd(size), rc(size), rd(size);
    int off = 0;
    for(int i=0; i<size; ++i) { 
        sc[i]=buckets[i].size(); 
        sd[i]=off; off+=sc[i]; 
        send_buf.insert(send_buf.end(), buckets[i].begin(), buckets[i].end()); 
    }
    
    // Measure Alltoall (Counts)
    local_bytes_sent += size * sizeof(int);
    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    int total_r=0; for(int i=0; i<size; ++i) { rd[i]=total_r; total_r+=rc[i]; }
    std::vector<Edge> recv_buf(total_r);

    // Measure Alltoallv (Data)
    local_bytes_sent += send_buf.size() * sizeof(Edge);
    MPI_Alltoallv(send_buf.data(), sc.data(), sd.data(), type, recv_buf.data(), rc.data(), rd.data(), type, MPI_COMM_WORLD);
    
    local_data = std::move(recv_buf);
    std::sort(local_data.begin(), local_data.end());
}

// --- Main ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    if (argc < 3) {
        if(rank==0) std::cerr << "Usage: " << argv[0] << " <graph> <nodes>" << std::endl;
        MPI_Finalize(); return 1;
    }

    std::string filename = argv[1];
    int num_nodes = std::atoi(argv[2]);
    MPI_Datatype mpi_edge_type; register_mpi_type(&mpi_edge_type);

    // 1. Load and Scatter
    std::vector<Edge> local_edges;
    int total_m = 0;
    {
        std::vector<int> sc(size);
        std::vector<Edge> buf;
        if(rank==0) {
            std::ifstream f(filename); int u,v,w,id=1;
            while(f>>u>>v>>w) buf.push_back({u,v,w,id++});
            total_m = buf.size();
            int tot=buf.size(), rem=tot%size;
            for(int i=0;i<size;++i) sc[i]=tot/size+(i<rem);
            
            // Measure Scatter (Root sends to all)
            local_bytes_sent += tot * sizeof(Edge); 
        }
        int mc; 
        MPI_Scatter(sc.data(),1,MPI_INT,&mc,1,MPI_INT,0,MPI_COMM_WORLD);
        local_edges.resize(mc);
        std::vector<int> disp(size);
        if(rank==0) {int s=0; for(int i=0;i<size;++i){disp[i]=s;s+=sc[i];}}
        
        MPI_Scatterv(buf.data(), sc.data(), disp.data(), mpi_edge_type, local_edges.data(), mc, mpi_edge_type, 0, MPI_COMM_WORLD);
    }

    double load_end = MPI_Wtime();

    // 2. Distributed Sort
    mpc_sort(local_edges, rank, size, mpi_edge_type);

    double sort_end = MPI_Wtime();

    // 3. Streaming Sketch Logic
    std::vector<AgmSketch> sketches(num_nodes);
    DSU dsu(num_nodes);
    std::unordered_map<int, Edge> edge_lookup;
    long long mst_weight = 0;
    int mst_edges_count = 0;

    for (int r = 0; r < size; ++r) {
        int chunk_size = (rank == r) ? local_edges.size() : 0;
        
        // Measure Bcast size
        if(rank == r) local_bytes_sent += sizeof(int);
        MPI_Bcast(&chunk_size, 1, MPI_INT, r, MPI_COMM_WORLD);

        std::vector<Edge> chunk(chunk_size);
        if (rank == r) chunk = local_edges;
        
        // Measure Bcast data
        if(rank == r) local_bytes_sent += chunk_size * sizeof(Edge);
        MPI_Bcast(chunk.data(), chunk_size, mpi_edge_type, r, MPI_COMM_WORLD);

        // AGM Logic
        for (const auto& e : chunk) {
            edge_lookup[e.id] = e;
            int r_u = dsu.find(e.u);
            int r_v = dsu.find(e.v);
            if (r_u != r_v) {
                sketches[r_u].update(e.id);
                sketches[r_v].update(e.id);
            }
        }

        bool merged = true;
        while (merged) {
            merged = false;
            for (int i = 0; i < num_nodes; ++i) {
                int root = dsu.find(i);
                if (root != i) continue;

                int id = sketches[root].query();
                if (id != -1) {
                    if (edge_lookup.count(id)) {
                        Edge e = edge_lookup[id];
                        int r_u = dsu.find(e.u);
                        int r_v = dsu.find(e.v);
                        if (r_u != r_v) {
                            if (rank == 0) {
                                mst_weight += e.weight;
                                mst_edges_count++;
                            }
                            sketches[r_u].merge(sketches[r_v]);
                            dsu.unite(r_u, r_v);
                            sketches[r_v] = AgmSketch();
                            merged = true;
                        }
                    }
                }
            }
        }
    }

    double total_end = MPI_Wtime();

    // Reduce Bytes Transferred
    long long global_bytes_transferred = 0;
    MPI_Reduce(&local_bytes_sent, &global_bytes_transferred, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double sort_time = sort_end - load_end;
        double sketch_time = total_end - sort_end;
        double total_time = total_end - start_time;

        // CSV Header: Processors, Nodes, Edges, SortTime, SketchTime, TotalTime, TotalBytes
        std::cout << "RESULTS," 
                  << size << "," 
                  << num_nodes << "," 
                  << total_m << ","
                  << std::fixed << std::setprecision(5) << sort_time << "," 
                  << std::fixed << std::setprecision(5) << sketch_time << "," 
                  << std::fixed << std::setprecision(5) << total_time << "," 
                  << global_bytes_transferred 
                  << std::endl;
    }

    MPI_Type_free(&mpi_edge_type);
    MPI_Finalize();
    return 0;
}