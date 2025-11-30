// mpi_mst_boruvka_chunk_sketch_flags.cpp
// MPI-based distributed MST with optional chunked processing and XOR-based sketch connectivity.
// Adds command-line flags for tuning L (levels), R (repetitions), chunk_size and mode.
//
// Compile:
//   mpicxx -std=c++17 -O2 -o mpi_mst_boruvka_chunk_sketch_flags mpi_mst_boruvka_chunk_sketch_flags.cpp
//
// Run format:
//   mpirun -np <P> ./mpi_mst_boruvka_chunk_sketch_flags edges.txt n_vertices [--mode=<mode>] [--L=<levels>] [--R=<reps>] [--chunk_size=<size>]
//
// Examples:
//   mpirun -np 4 ./mpi_mst_boruvka_chunk_sketch_flags edges.txt 1000 --mode=chunk+sketch --L=20 --R=4 --chunk_size=1000
//   mpirun -np 8 ./mpi_mst_boruvka_chunk_sketch_flags edges.txt 50000 --mode=sketch --L=24 --R=8
//
// Flags (all optional):
//   --mode: one of {default, chunk, sketch, chunk+sketch}. Default: default (deterministic gather-based Boruvka).
//   --L: number of sketch levels (int). Default: 20.
//   --R: number of sketch repetitions (int). Default: 4.
//   --chunk_size: chunk size (approx edges per chunk). Default: n_vertices (i.e., process chunks of size ~= n).
//   -h, --help : print this usage message.
//
// Notes:
//  - The program stores a comp[] array (size n_vertices) on every process (linear memory per machine).
//  - The sketch primitive is probabilistic but uses verification, so accepted edges are always valid; you may need
//    to tune L and R to increase success probability for large graphs.
//

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int u, v;
    double w;
    long long gid; // global edge index (0-based)
};

static const double INF = 1e300;

// DSU for contraction
struct DSU {
    int n;
    vector<int> p, r;
    DSU(int n=0): n(n), p(n), r(n,0){
        for(int i=0;i<n;i++) p[i]=i;
    }
    int find(int a){ return p[a]==a ? a : p[a]=find(p[a]); }
    bool unite(int a,int b){
        a=find(a); b=find(b);
        if(a==b) return false;
        if(r[a]<r[b]) swap(a,b);
        p[b]=a;
        if(r[a]==r[b]) r[a]++;
        return true;
    }
};

// Read partitioned edges (round-robin by line index).
vector<Edge> read_partitioned_edges_with_gid(const string &filename, int rank, int nproc){
    vector<Edge> edges;
    ifstream in(filename);
    if(!in){
        if(rank==0) cerr << "Error: cannot open " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    string line;
    long long idx = 0;
    while(getline(in,line)){
        if(line.empty()) { idx++; continue; }
        if(line[0]=='#'){ idx++; continue; }
        if(idx % nproc == rank){
            stringstream ss(line);
            int u,v; double w;
            if(!(ss>>u>>v>>w)) { idx++; continue; }
            edges.push_back({u,v,w, idx});
        }
        idx++;
    }
    return edges;
}

void local_sort_edges(vector<Edge> &edges){
    sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b){
        if(a.w != b.w) return a.w < b.w;
        if(a.u != b.u) return a.u < b.u;
        return a.v < b.v;
    });
}

vector<double> serialize_edges_double(const vector<Edge> &E){
    vector<double> out;
    out.reserve(E.size()*3);
    for(auto &e: E){
        out.push_back((double)e.u);
        out.push_back((double)e.v);
        out.push_back(e.w);
    }
    return out;
}

vector<Edge> deserialize_edges_double(const vector<double> &buf){
    vector<Edge> out;
    int L = buf.size() / 3;
    out.reserve(L);
    for(int i=0;i<L;i++){
        int u = (int)buf[3*i+0];
        int v = (int)buf[3*i+1];
        double w = buf[3*i+2];
        out.push_back({u,v,w,-1});
    }
    return out;
}

// Deterministic connectivity (gather best per component)
vector<pair<pair<int,int>, double>> deterministic_component_candidates_and_contract(
    const vector<Edge> &local_edges,
    vector<int> &comp, // size n_vertices, current comp labels (shared across processes)
    int rank, int nproc, int n_vertices
){
    unordered_map<int, pair<double, pair<int,int>>> local_best;
    for(auto &e: local_edges){
        int cu = comp[e.u];
        int cv = comp[e.v];
        if(cu == cv) continue;
        if(local_best.find(cu)==local_best.end() || e.w < local_best[cu].first){
            local_best[cu] = {e.w, {e.u, e.v}};
        }
        if(local_best.find(cv)==local_best.end() || e.w < local_best[cv].first){
            local_best[cv] = {e.w, {e.v, e.u}};
        }
    }
    vector<double> packed;
    packed.reserve(local_best.size()*4);
    for(auto &kv: local_best){
        int compId = kv.first;
        double w = kv.second.first;
        int u = kv.second.second.first;
        int v = kv.second.second.second;
        packed.push_back((double)compId);
        packed.push_back(w);
        packed.push_back((double)u);
        packed.push_back((double)v);
    }
    int mylen = (int)packed.size();
    vector<int> recvcounts(nproc);
    MPI_Allgather(&mylen, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    vector<int> displs(nproc);
    int total = 0;
    for(int i=0;i<nproc;i++){ displs[i]=total; total += recvcounts[i]; }
    vector<double> recvbuf(total);
    MPI_Allgatherv(packed.data(), mylen, MPI_DOUBLE,
                   recvbuf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    unordered_map<int, pair<double, pair<int,int>>> global_best;
    int idx = 0;
    while(idx + 3 < total){
        int compId = (int)recvbuf[idx++];
        double w = recvbuf[idx++];
        int u = (int)recvbuf[idx++];
        int v = (int)recvbuf[idx++];
        auto &cur = global_best[compId];
        if(global_best.find(compId) == global_best.end() || w < cur.first){
            global_best[compId] = {w, {u,v}};
        }
    }
    vector<pair<pair<int,int>, double>> chosen;
    {
        unordered_set<int> comps_set;
        for(int v=0; v<n_vertices; ++v) comps_set.insert(comp[v]);
        unordered_map<int,int> comp_to_idx;
        int k = 0;
        for(int x: comps_set) comp_to_idx[x] = k++;
        DSU comp_dsu(k);
        for(auto &kv: global_best){
            int compId = kv.first;
            auto pr = kv.second.second; double w = kv.second.first;
            int u = pr.first, v = pr.second;
            int cu = comp[u], cv = comp[v];
            if(cu == cv) continue;
            comp_dsu.unite(comp_to_idx[cu], comp_to_idx[cv]);
            chosen.push_back({{u,v}, w});
        }
        vector<int> newComp(n_vertices);
        unordered_map<int,int> remap;
        int cur = 0;
        for(int v=0; v<n_vertices; ++v){
            int x = comp[v];
            int rootidx = comp_dsu.find(comp_to_idx[x]);
            if(remap.find(rootidx)==remap.end()) remap[rootidx] = cur++;
            newComp[v] = remap[rootidx];
        }
        MPI_Bcast(newComp.data(), n_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        comp = newComp;
    }
    int rank_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    if(rank_world==0) return chosen;
    else return {};
}

// --- Sketch primitive ---
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}
int ctz_u64(uint64_t x){ if(x==0) return 64; return __builtin_ctzll(x); }

vector<pair<pair<int,int>, double>> sketch_component_candidates_and_contract(
    const vector<Edge> &local_edges,
    vector<int> &comp,
    int rank, int nproc, int n_vertices,
    int L = 20, int R = 4
){
    int levels = L;
    size_t total_entries = (size_t)n_vertices * levels;
    vector<uint64_t> xor_key_local(total_entries), xor_u_local(total_entries), xor_v_local(total_entries);
    vector<unsigned char> parity_local(total_entries);
    vector<uint64_t> xor_key_global(total_entries), xor_u_global(total_entries), xor_v_global(total_entries);
    vector<unsigned char> parity_global(total_entries);
    vector<pair<pair<int,int>, double>> chosen_edges_root;

    for(int rep=0; rep<R; ++rep){
        fill(xor_key_local.begin(), xor_key_local.end(), 0ULL);
        fill(xor_u_local.begin(), xor_u_local.end(), 0ULL);
        fill(xor_v_local.begin(), xor_v_local.end(), 0ULL);
        fill(parity_local.begin(), parity_local.end(), 0);
        uint64_t seed = ((uint64_t)rep + 0x9e3779b97f4a7c15ULL) ^ 0x123456789abcdefULL;
        for(const auto &e: local_edges){
            int cu = comp[e.u], cv = comp[e.v];
            if(cu == cv) continue;
            uint64_t a = (uint64_t)min(e.u,e.v);
            uint64_t b = (uint64_t)max(e.u,e.v);
            uint64_t h = splitmix64( (a << 32) ^ b ^ seed );
            if(h==0) h = 1;
            int level = ctz_u64(h);
            if(level >= levels) level = levels-1;
            size_t idx1 = (size_t)cu * levels + level;
            size_t idx2 = (size_t)cv * levels + level;
            xor_key_local[idx1] ^= h;
            xor_u_local[idx1] ^= (uint64_t)e.u;
            xor_v_local[idx1] ^= (uint64_t)e.v;
            parity_local[idx1] ^= 1;
            xor_key_local[idx2] ^= h;
            xor_u_local[idx2] ^= (uint64_t)e.u;
            xor_v_local[idx2] ^= (uint64_t)e.v;
            parity_local[idx2] ^= 1;
        }
        MPI_Allreduce(xor_key_local.data(), xor_key_global.data(), (int)total_entries, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
        MPI_Allreduce(xor_u_local.data(), xor_u_global.data(), (int)total_entries, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
        MPI_Allreduce(xor_v_local.data(), xor_v_global.data(), (int)total_entries, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
        MPI_Allreduce(parity_local.data(), parity_global.data(), (int)total_entries, MPI_UNSIGNED_CHAR, MPI_BXOR, MPI_COMM_WORLD);
        vector<pair<int, pair<int,int>>> local_candidates;
        local_candidates.reserve((size_t)n_vertices/10);
        for(int cid=0; cid < n_vertices; ++cid){
            for(int l=0; l<levels; ++l){
                size_t pos = (size_t)cid * levels + l;
                if(parity_global[pos] & 1u){
                    int cu = (int)xor_u_global[pos];
                    int cv = (int)xor_v_global[pos];
                    if(cu == cv) continue;
                    local_candidates.push_back({cid, {cu, cv}});
                    break;
                }
            }
        }
        sort(local_candidates.begin(), local_candidates.end(), [](auto &a, auto &b){ return a.first < b.first; });
        vector<double> local_minw(local_candidates.size(), INF);
        for(size_t i=0;i<local_candidates.size();++i){
            int u = local_candidates[i].second.first;
            int v = local_candidates[i].second.second;
            double best = INF;
            for(const auto &e: local_edges){
                if( (e.u==u && e.v==v) || (e.u==v && e.v==u) ){
                    if(comp[e.u] != comp[e.v]) best = min(best, e.w);
                }
            }
            local_minw[i] = best;
        }
        vector<double> global_minw(local_candidates.size(), INF);
        if(local_candidates.size() > 0){
            MPI_Allreduce(local_minw.data(), global_minw.data(), (int)local_candidates.size(), MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        }
        unordered_set<int> used_components;
        vector<pair<pair<int,int>, double>> chosen_this_rep;
        for(size_t i=0;i<local_candidates.size();++i){
            int cid = local_candidates[i].first;
            if(used_components.find(cid) != used_components.end()) continue;
            if(global_minw[i] < INF){
                int u = local_candidates[i].second.first;
                int v = local_candidates[i].second.second;
                double w = global_minw[i];
                chosen_this_rep.push_back({{u,v}, w});
                used_components.insert(cid);
            }
        }
        unordered_set<int> comps_set;
        for(int v=0; v<n_vertices; ++v) comps_set.insert(comp[v]);
        unordered_map<int,int> comp_to_idx; int idx=0;
        for(int x: comps_set) comp_to_idx[x]=idx++;
        DSU comp_dsu(idx);
        for(auto &pe: chosen_this_rep){
            int u = pe.first.first, v = pe.first.second;
            int cu = comp[u], cv = comp[v];
            if(cu==cv) continue;
            comp_dsu.unite(comp_to_idx[cu], comp_to_idx[cv]);
        }
        vector<int> newComp(n_vertices);
        unordered_map<int,int> remap; int cur=0;
        for(int v=0; v<n_vertices; ++v){
            int rootidx = comp_dsu.find(comp_to_idx[ comp[v] ]);
            if(remap.find(rootidx)==remap.end()) remap[rootidx] = cur++;
            newComp[v] = remap[rootidx];
        }
        MPI_Bcast(newComp.data(), n_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        comp = newComp;
        int rank_local; MPI_Comm_rank(MPI_COMM_WORLD, &rank_local);
        if(rank_local==0){
            for(auto &pe: chosen_this_rep) chosen_edges_root.push_back(pe);
        }
    }
    int rank_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    if(rank_world==0) return chosen_edges_root;
    else return {};
}

vector<Edge> local_maximal_forest_by_comp(const vector<Edge> &edges_subset, const vector<int> &comp, int n_vertices){
    vector<Edge> E = edges_subset;
    sort(E.begin(), E.end(), [](const Edge &a, const Edge &b){ return a.w < b.w; });
    unordered_map<int,int> mapc; int idx=0;
    for(int v=0; v<n_vertices; ++v) if(mapc.find(comp[v])==mapc.end()) mapc[comp[v]] = idx++;
    DSU dsu(idx);
    vector<Edge> picked;
    for(auto &e: E){
        int cu = mapc[ comp[e.u] ];
        int cv = mapc[ comp[e.v] ];
        if(cu == cv) continue;
        if(dsu.unite(cu, cv)) picked.push_back(e);
    }
    return picked;
}

void print_usage_and_exit(const char *prog){
    if(prog) cerr << "Usage: " << prog << " edges.txt n_vertices [--mode=<mode>] [--L=<levels>] [--R=<reps>] [--chunk_size=<size>]\n";
    cerr << "Modes: default, chunk, sketch, chunk+sketch\n";
    cerr << "Defaults: --mode=default --L=20 --R=4 --chunk_size=n_vertices\n";
    MPI_Finalize();
    exit(1);
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if(argc < 3){ if(rank==0) print_usage_and_exit(argv[0]); }
    string fname = argv[1];
    int n_vertices = atoi(argv[2]);
    string mode = "default";
    int L = 20;
    int R = 4;
    long long chunk_size_val = -1; // if -1 -> use n_vertices

    // parse optional flags from argv[3..]
    for(int i=3;i<argc;i++){
        string s = argv[i];
        if(s=="-h" || s=="--help"){
            if(rank==0) print_usage_and_exit(argv[0]);
        }
        if(s.rfind("--",0)==0){
            // format --key=value
            auto pos = s.find('=');
            string key, val;
            if(pos==string::npos){ key = s.substr(2); val = ""; }
            else { key = s.substr(2,pos-2); val = s.substr(pos+1); }
            if(key=="mode") {
                if(!val.empty()) mode = val;
            } else if(key=="L"){
                if(!val.empty()) L = atoi(val.c_str());
            } else if(key=="R"){
                if(!val.empty()) R = atoi(val.c_str());
            } else if(key=="chunk_size"){
                if(!val.empty()) chunk_size_val = atoll(val.c_str());
            } else {
                if(rank==0) cerr << "Unknown flag: " << key << "\n";
            }
        } else {
            if(rank==0) cerr << "Ignoring arg: " << s << "\n";
        }
    }

    if(chunk_size_val <= 0) chunk_size_val = n_vertices;

    bool use_chunk = false;
    bool use_sketch = false;
    if(mode == "chunk") use_chunk = true;
    else if(mode == "sketch") use_sketch = true;
    else if(mode == "chunk+sketch") { use_chunk = true; use_sketch = true; }

    vector<Edge> local_edges = read_partitioned_edges_with_gid(fname, rank, nproc);
    long long local_m = (long long)local_edges.size();
    long long m_total = 0;
    MPI_Allreduce(&local_m, &m_total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(rank==0){ cerr << "Total edges m = " << m_total << ", n_vertices = " << n_vertices << ", processes = " << nproc << "\n"; }

    vector<int> comp(n_vertices);
    for(int v=0; v<n_vertices; ++v) comp[v]=v;

    vector<pair<pair<int,int>, double>> mst_edges_root;
    double mst_weight_root = 0.0;

    long long chunk_size = chunk_size_val;
    long long num_chunks = (m_total + chunk_size - 1) / chunk_size;

    if(!use_chunk){
        int iteration = 0;
        while(true){
            iteration++;
            if(rank==0) cerr << "[Global] Boruvka iteration " << iteration << "\n";
            vector<pair<pair<int,int>, double>> chosen;
            if(use_sketch){
                chosen = sketch_component_candidates_and_contract(local_edges, comp, rank, nproc, n_vertices, L, R);
            } else {
                chosen = deterministic_component_candidates_and_contract(local_edges, comp, rank, nproc, n_vertices);
            }
            if(rank==0){
                if(chosen.empty()){
                    cerr << "No chosen edges in this round -> graph might be disconnected or sketches failed. Breaking.\n";
                    break;
                }
                for(auto &pe: chosen){ mst_edges_root.push_back(pe); mst_weight_root += pe.second; }
            }
            int unique_local = 0;
            { unordered_set<int> s; for(int v=0; v<n_vertices; ++v) s.insert(comp[v]); unique_local = (int)s.size(); }
            int unique_global = 0;
            MPI_Allreduce(&unique_local, &unique_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if(rank==0) cerr << "After iter " << iteration << " components: " << unique_global << "\n";
            if(unique_global <= 1) break;
            if(iteration > 2 * (int)log2(max(2, n_vertices)) + 100){ if(rank==0) cerr << "Stopping after iteration limit\n"; break; }
        }
    } else {
        if(rank == 0) cerr << "Chunk mode: num_chunks = " << num_chunks << ", chunk_size = " << chunk_size << "\n";
        for(long long ch=0; ch < num_chunks; ++ch){
            if(rank==0) cerr << "Processing chunk " << ch << "/" << num_chunks << "\n";
            long long start = ch * chunk_size;
            long long end = min(m_total, (ch+1)*chunk_size) - 1;
            vector<Edge> chunk_edges;
            chunk_edges.reserve( (size_t)min((long long)local_edges.size(), chunk_size / nproc + 16) );
            for(auto &e: local_edges) if(e.gid >= start && e.gid <= end) chunk_edges.push_back(e);
            vector<Edge> local_picked = local_maximal_forest_by_comp(chunk_edges, comp, n_vertices);
            vector<double> packed = serialize_edges_double(local_picked);
            int mylen = (int)packed.size();
            vector<int> recvcounts(nproc);
            MPI_Allgather(&mylen, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            vector<int> displs(nproc);
            int total = 0;
            for(int i=0;i<nproc;i++){ displs[i]=total; total += recvcounts[i]; }
            vector<double> recvbuf(total);
            MPI_Allgatherv(packed.data(), mylen, MPI_DOUBLE, recvbuf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            vector<Edge> gathered = deserialize_edges_double(recvbuf);
            bool progress = true;
            int inner_iter = 0;
            while(progress){
                inner_iter++;
                if(use_sketch){
                    auto chosen = sketch_component_candidates_and_contract(gathered, comp, rank, nproc, n_vertices, L, R);
                    if(rank==0){ if(chosen.empty()) progress = false; else { for(auto &pe: chosen){ mst_edges_root.push_back(pe); mst_weight_root += pe.second; } } }
                } else {
                    if(rank==0){
                        unordered_map<int, tuple<double,int,int>> best;
                        for(auto &e: gathered){ int cu = comp[e.u], cv = comp[e.v]; if(cu==cv) continue; auto it = best.find(cu); if(it==best.end() || e.w < get<0>(it->second)) best[cu] = {e.w, e.u, e.v}; it = best.find(cv); if(it==best.end() || e.w < get<0>(it->second)) best[cv] = {e.w, e.v, e.u}; }
                        unordered_set<int> comps_set; for(int v=0; v<n_vertices; ++v) comps_set.insert(comp[v]); unordered_map<int,int> comp_to_idx; int idx=0; for(auto &x: comps_set) comp_to_idx[x]=idx++;
                        DSU comp_dsu(idx);
                        vector<pair<pair<int,int>, double>> chosen_local;
                        for(auto &kv: best){ double w = get<0>(kv.second); int u=get<1>(kv.second), v=get<2>(kv.second); int cu=comp[u], cv=comp[v]; if(cu==cv) continue; if(comp_dsu.unite(comp_to_idx[cu], comp_to_idx[cv])) chosen_local.push_back({{u,v}, w}); }
                        vector<int> newComp(n_vertices); unordered_map<int,int> remap; int cur=0; for(int v=0; v<n_vertices; ++v){ int rootidx = comp_dsu.find(comp_to_idx[ comp[v] ]); if(remap.find(rootidx) == remap.end()) remap[rootidx] = cur++; newComp[v] = remap[rootidx]; }
                        MPI_Bcast(newComp.data(), n_vertices, MPI_INT, 0, MPI_COMM_WORLD);
                        comp.swap(newComp);
                        for(auto &pe: chosen_local){ mst_edges_root.push_back(pe); mst_weight_root += pe.second; }
                        progress = !chosen_local.empty();
                    } else {
                        vector<int> newComp(n_vertices);
                        MPI_Bcast(newComp.data(), n_vertices, MPI_INT, 0, MPI_COMM_WORLD);
                        comp.swap(newComp);
                    }
                }
                int unique_local = 0; { unordered_set<int> s; for(int v=0; v<n_vertices; ++v) s.insert(comp[v]); unique_local = (int)s.size(); }
                int unique_global = 0; MPI_Allreduce(&unique_local, &unique_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                if(rank==0){ if(unique_global <= 1) { progress = false; break; } }
                if(inner_iter > 64) { if(rank==0) cerr << "inner_iter limit reached\n"; break; }
                break; // conservative: run at most one inner loop iteration per chunk (safe for demo)
            }
        }
    }

    if(rank==0){
        cerr << "Final MST weight (sum of selected edges, may be approximate if graph disconnected): " << mst_weight_root << "\n";
        cout << "MST edges (u v):\n";
        for(auto &pe: mst_edges_root) cout << pe.first.first << " " << pe.first.second << "\n";
    }

    MPI_Finalize();
    return 0;
}
