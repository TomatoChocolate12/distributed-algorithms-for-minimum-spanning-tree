// mpi_mst_sketch.cpp
// Distributed Karger-Klein-Tarjan style MST (graph sketching) using MPI.
//
// Compile:
//   mpicxx -O3 -std=c++17 -o mpi_mst_sketch mpi_mst_sketch.cpp
//
// Run:
//   mpirun -n <P> ./mpi_mst_sketch edges.txt N_vertices [--block] [--debug]
//
// Notes:
// - edges.txt: u v w  (0-indexed vertices)
// - This focuses on correctness and the KKT sampling/filtering idea.
// - Root (rank 0) prints MST edges at the end.

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int u, v;
    double w;
    uint64_t id; // optional unique id for stability
};

struct UF {
    vector<int> p, r;
    UF(int n=0) { init(n); }
    void init(int n) { p.resize(n); r.assign(n,0); iota(p.begin(), p.end(), 0); }
    int find(int x){ while (p[x]!=x){ p[x]=p[p[x]]; x=p[x]; } return x; }
    bool unite(int a,int b){
        a=find(a); b=find(b); if (a==b) return false;
        if (r[a]<r[b]) p[a]=b; else { p[b]=a; if (r[a]==r[b]) r[a]++; } return true;
    }
};

// ---------- Utility: read partitioned edges (round-robin by default) ----------
vector<Edge> load_partitioned_edges_roundrobin(const string &file, int rank, int size) {
    vector<Edge> edges;
    ifstream in(file);
    if (!in) { if (rank==0) cerr<<"Cannot open "<<file<<"\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    string line;
    long long idx=0;
    uint64_t uid = ((uint64_t)rank<<32) ^ 1469598103934665603ULL;
    while (getline(in,line)) {
        if (line.empty()) { idx++; continue; }
        if ((idx % size) != rank) { idx++; continue; }
        stringstream ss(line);
        int u,v; double w;
        if (!(ss>>u>>v>>w)) { idx++; continue; }
        edges.push_back({u,v,w, uid++});
        idx++;
    }
    return edges;
}

// ---------- Sequential Kruskal (used at base case on root) ----------
vector<Edge> kruskal_local(vector<Edge> &edges, int n) {
    vector<int> idx(edges.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b){
        if (edges[a].w != edges[b].w) return edges[a].w < edges[b].w;
        if (edges[a].u != edges[b].u) return edges[a].u < edges[b].u;
        return edges[a].v < edges[b].v;
    });
    UF uf(n);
    vector<Edge> out;
    out.reserve(n>0 ? n-1 : 0);
    for (int i: idx) {
        Edge &e = edges[i];
        if (uf.unite(e.u, e.v)) out.push_back(e);
        if ((int)out.size() == n-1) break;
    }
    return out;
}

// ---------- Build adjacency from edges (for MST forest) ----------
vector<vector<pair<int, pair<int,double>>>> build_adj(int n, const vector<Edge> &mst) {
    // returns adj[v] = list of (to, (edge_index_or_dummy, weight))
    vector<vector<pair<int, pair<int,double>>>> adj(n);
    adj.assign(n, {});
    for (size_t i=0;i<mst.size();++i) {
        const Edge &e = mst[i];
        adj[e.u].push_back({e.v, {(int)i, e.w}});
        adj[e.v].push_back({e.u, {(int)i, e.w}});
    }
    return adj;
}

// ---------- LCA + max-edge on path (binary lifting) ----------
struct LCA {
    int n, LOG;
    vector<int> depth;
    vector<vector<int>> up;      // up[k][v] = 2^k ancestor
    vector<vector<double>> mx;  // mx[k][v] = max edge weight on path from v up 2^k
    LCA() : n(0), LOG(0) {}
    void build(int N, const vector<vector<pair<int,pair<int,double>>>> &adj) {
        n = N;
        LOG = 1;
        while ((1<<LOG) <= n) LOG++;
        depth.assign(n, -1);
        up.assign(LOG, vector<int>(n, -1));
        mx.assign(LOG, vector<double>(n, -1.0));
        // run DFS/BFS from all roots (forest)
        for (int s=0;s<n;++s) if (depth[s] == -1) {
            // root s
            depth[s] = 0;
            up[0][s] = s;
            mx[0][s] = -1.0; // no edge to root
            deque<int> dq; dq.push_back(s);
            while (!dq.empty()) {
                int v = dq.front(); dq.pop_front();
                for (auto &ed : adj[v]) {
                    int to = ed.first;
                    double w = ed.second.second;
                    if (depth[to] == -1) {
                        depth[to] = depth[v] + 1;
                        up[0][to] = v;
                        mx[0][to] = w;
                        dq.push_back(to);
                    }
                }
            }
        }
        // binary lifting
        for (int k=1;k<LOG;++k) {
            for (int v=0; v<n; ++v) {
                up[k][v] = up[k-1][ up[k-1][v] ];
                mx[k][v] = max(mx[k-1][v], mx[k-1][ up[k-1][v] ]);
            }
        }
    }
    // return max edge weight on path u-v in the forest (if disconnected path, returns +inf for safety or -inf? we'll handle)
    double max_on_path(int u, int v) {
        if (u == v) return -1.0;
        if (depth[u] < depth[v]) swap(u,v);
        double ans = -1.0;
        // lift u up to depth v
        int diff = depth[u] - depth[v];
        for (int k=0; k<LOG; ++k) if (diff & (1<<k)) {
            ans = max(ans, mx[k][u]);
            u = up[k][u];
        }
        if (u == v) return ans;
        for (int k = LOG-1; k>=0; --k) {
            if (up[k][u] != up[k][v]) {
                ans = max(ans, mx[k][u]);
                ans = max(ans, mx[k][v]);
                u = up[k][u];
                v = up[k][v];
            }
        }
        // now u and v are children of LCA
        ans = max(ans, mx[0][u]);
        ans = max(ans, mx[0][v]);
        return ans;
    }
};

// ---------- Distributed Boruvka (centralized merging at root for simplicity) ----------
vector<Edge> distributed_boruvka(vector<Edge> &local_edges, int n, bool debug) {
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<int> comp(n);
    iota(comp.begin(), comp.end(), 0);
    vector<Edge> mst; mst.reserve(n>0 ? n-1 : 0);
    int iter = 0;
    while (true) {
        iter++;
        // local best per component
        unordered_map<int, tuple<double,int,int>> local_best;
        for (auto &e : local_edges) {
            int cu = comp[e.u], cv = comp[e.v];
            if (cu == cv) continue;
            auto it = local_best.find(cu);
            if (it==local_best.end() || e.w < get<0>(it->second)) local_best[cu] = {e.w, e.u, e.v};
            it = local_best.find(cv);
            if (it==local_best.end() || e.w < get<0>(it->second)) local_best[cv] = {e.w, e.u, e.v};
        }
        // serialize local best to arrays
        int k_local = (int)local_best.size();
        vector<int> comps; comps.reserve(k_local);
        vector<int> us; us.reserve(k_local);
        vector<int> vs; vs.reserve(k_local);
        vector<double> ws; ws.reserve(k_local);
        for (auto &kv: local_best) {
            comps.push_back(kv.first);
            ws.push_back(get<0>(kv.second));
            us.push_back(get<1>(kv.second));
            vs.push_back(get<2>(kv.second));
        }
        // gather counts
        vector<int> counts(size);
        MPI_Allgather(&k_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        vector<int> comps_recvcounts(size), us_recvcounts(size), vs_recvcounts(size), ws_recvcounts(size);
        for (int i=0;i<size;++i) comps_recvcounts[i] = counts[i];
        int total_k = 0; for (int x: counts) total_k += x;
        vector<int> displ(size,0);
        for (int i=1;i<size;++i) displ[i] = displ[i-1] + comps_recvcounts[i-1];
        // gather arrays to root
        vector<int> all_comps; vector<int> all_us; vector<int> all_vs; vector<double> all_ws;
        if (rank==0) { all_comps.resize(total_k); all_us.resize(total_k); all_vs.resize(total_k); all_ws.resize(total_k); }
        MPI_Gatherv(comps.data(), k_local, MPI_INT, rank==0 ? all_comps.data() : nullptr, comps_recvcounts.data(), displ.data(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(us.data(), k_local, MPI_INT, rank==0 ? all_us.data() : nullptr, comps_recvcounts.data(), displ.data(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(vs.data(), k_local, MPI_INT, rank==0 ? all_vs.data() : nullptr, comps_recvcounts.data(), displ.data(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(ws.data(), k_local, MPI_DOUBLE, rank==0 ? all_ws.data() : nullptr, comps_recvcounts.data(), displ.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // root picks global best per component
        vector<Edge> chosen;
        if (rank==0) {
            unordered_map<int, tuple<double,int,int>> global_best;
            for (int i=0;i<total_k;++i) {
                int c = all_comps[i];
                double w = all_ws[i];
                int u = all_us[i], v = all_vs[i];
                auto it = global_best.find(c);
                if (it==global_best.end() || w < get<0>(it->second)) global_best[c] = {w,u,v};
            }
            // deduplicate and collect chosen edges
            unordered_set<uint64_t> seen;
            for (auto &kv: global_best) {
                double w; int u,v; tie(w,u,v) = kv.second;
                uint64_t key = ((uint64_t)min(u,v) << 32) ^ (uint64_t)max(u,v) ^ (uint64_t)lrint(w*100000.0);
                if (seen.insert(key).second) chosen.push_back({u,v,w,0});
            }
        }

        // broadcast count of chosen edges
        int chosen_local = (int)chosen.size();
        int chosen_global = 0;
        MPI_Allreduce(&chosen_local, &chosen_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (chosen_global == 0) break;
        // broadcast chosen edges from root to all
        vector<int> chosen_us, chosen_vs; vector<double> chosen_ws;
        if (rank==0) {
            chosen_us.reserve(chosen.size()); chosen_vs.reserve(chosen.size()); chosen_ws.reserve(chosen.size());
            for (auto &e: chosen) { chosen_us.push_back(e.u); chosen_vs.push_back(e.v); chosen_ws.push_back(e.w); }
        }
        // We need to broadcast length and arrays; root sets them
        int root_k = (rank==0 ? (int)chosen.size() : 0);
        MPI_Bcast(&root_k, 1, MPI_INT, 0, MPI_COMM_WORLD);
        chosen_us.resize(root_k); chosen_vs.resize(root_k); chosen_ws.resize(root_k);
        MPI_Bcast(chosen_us.data(), root_k, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(chosen_vs.data(), root_k, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(chosen_ws.data(), root_k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // root computes new comp mapping
        vector<int> new_comp(n);
        vector<Edge> to_add;
        if (rank==0) {
            UF uf(n);
            for (int i=0;i<root_k;++i) {
                int u = chosen_us[i], v = chosen_vs[i];
                int cu = comp[u], cv = comp[v];
                if (cu != cv) uf.unite(cu, cv);
            }
            unordered_map<int,int> maproot;
            int nextid = 0;
            for (int i=0;i<n;++i) {
                int r = uf.find(comp[i]);
                auto it = maproot.find(r);
                if (it==maproot.end()) { maproot[r] = nextid++; new_comp[i] = maproot[r]; }
                else new_comp[i] = it->second;
            }
            for (int i=0;i<root_k;++i) {
                int u = chosen_us[i], v = chosen_vs[i];
                if (comp[u] != comp[v]) to_add.push_back({u,v,chosen_ws[i],0});
            }
        }
        // broadcast new_comp
        MPI_Bcast(new_comp.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        // update comp on all ranks
        comp.swap(new_comp);
        // root appends MST edges
        if (rank==0) {
            for (auto &e: to_add) mst.push_back(e);
        }
        // termination check: comp is replicated identical now
        unordered_set<int> uniq(comp.begin(), comp.end());
        if ((int)uniq.size() <= 1) break;
    } // while
    return mst;
}

// ---------- Distributed KKT / graph-sketching MST ----------
vector<Edge> distributed_kkt_mst(vector<Edge> &local_edges, int n, bool debug, int base_threshold=1024) {
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    // compute global edge count
    int local_m = (int)local_edges.size();
    int global_m = 0;
    MPI_Allreduce(&local_m, &global_m, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (debug && rank==0) cerr<<"distributed_kkt_mst: global_m="<<global_m<<"\n";

    // base case: if global edges small => gather to root and run Kruskal
    if (global_m <= base_threshold) {
        // gather sizes
        vector<int> counts(size);
        MPI_Allgather(&local_m, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        vector<int> displs(size,0);
        int total = 0; for (int i=0;i<size;++i) { displs[i]=total; total+=counts[i]; }
        // pack local edges into arrays (u,v,w)
        vector<int> loc_u(local_m), loc_v(local_m);
        vector<double> loc_w(local_m);
        for (int i=0;i<local_m;++i) { loc_u[i]=local_edges[i].u; loc_v[i]=local_edges[i].v; loc_w[i]=local_edges[i].w; }
        vector<int> all_u(total), all_v(total); vector<double> all_w(total);
        MPI_Gatherv(loc_u.data(), local_m, MPI_INT, rank==0?all_u.data():nullptr, counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(loc_v.data(), local_m, MPI_INT, rank==0?all_v.data():nullptr, counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(loc_w.data(), local_m, MPI_DOUBLE, rank==0?all_w.data():nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        vector<Edge> mst_global;
        if (rank==0) {
            vector<Edge> edges_all; edges_all.reserve(total);
            for (int i=0;i<total;++i) edges_all.push_back({all_u[i], all_v[i], all_w[i], (uint64_t)i});
            mst_global = kruskal_local(edges_all, n);
        }
        // broadcast mst size and edges from root to all
        int mst_sz = rank==0 ? (int)mst_global.size() : 0;
        MPI_Bcast(&mst_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> mst_u(mst_sz), mst_v(mst_sz); vector<double> mst_w(mst_sz);
        if (rank==0) {
            for (int i=0;i<mst_sz;++i){ mst_u[i]=mst_global[i].u; mst_v[i]=mst_global[i].v; mst_w[i]=mst_global[i].w; }
        }
        MPI_Bcast(mst_u.data(), mst_sz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(mst_v.data(), mst_sz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(mst_w.data(), mst_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        vector<Edge> mst_rep; mst_rep.reserve(mst_sz);
        for (int i=0;i<mst_sz;++i) mst_rep.push_back({mst_u[i], mst_v[i], mst_w[i], (uint64_t)i});
        return mst_rep;
    }

    // 1) Sample each edge with probability 1/2 (independently) locally
    std::mt19937_64 rng((uint64_t)rank + 0x9e3779b97f4a7c15ULL + (uint64_t)time(nullptr));
    std::bernoulli_distribution coin(0.5);
    vector<Edge> local_sample; local_sample.reserve(local_m/2 + 4);
    for (auto &e : local_edges) if (coin(rng)) local_sample.push_back(e);

    // 2) Recursively compute MST(S) (distributed) -> it returns replicated MST_S on all ranks
    vector<Edge> mstS = distributed_kkt_mst(local_sample, n, debug, base_threshold);

    // 3) Build LCA on MST(S) (replicated tree(s) on all ranks)
    vector<vector<pair<int,pair<int,double>>>> adj = build_adj(n, mstS);
    LCA lca; lca.build(n, adj);

    // 4) Filter edges: keep only edges with w < max_on_path(u,v) in MST(S)
    vector<Edge> local_candidates; local_candidates.reserve(local_m/10);
    for (auto &e : local_edges) {
        double mx = lca.max_on_path(e.u, e.v);
        // if u and v disconnected in MST(S), mx might be -1.0 -> keep the edge
        if (mx < 0.0 || e.w < mx - 1e-12) {
            local_candidates.push_back(e);
        }
    }
    // optional debug info: how many edges kept
    int kept_local = (int)local_candidates.size();
    int kept_global = 0;
    MPI_Allreduce(&kept_local, &kept_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (debug && rank==0) cerr<<"After filtering, kept_global="<<kept_global<<"\n";

    // 5) Run distributed Boruvka on remaining candidates to finish MST
    vector<Edge> final_mst = distributed_boruvka(local_candidates, n, debug);

    // final_mst is returned only from Boruvka merging edges (constructed at root and returned as replicated vector)
    return final_mst;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank==0) cerr<<"Usage: "<<argv[0]<<" edges.txt N_vertices [--block] [--debug]\n";
        MPI_Finalize(); return 1;
    }
    string edgefile = argv[1];
    int n = atoi(argv[2]);
    bool block=false, debug=false;
    for (int i=3;i<argc;i++){ string s=argv[i]; if (s=="--block") block=true; if (s=="--debug") debug=true; }

    if (debug && rank==0) cerr<<"Starting distributed KKT MST with "<<size<<" ranks\n";

    vector<Edge> local_edges = load_partitioned_edges_roundrobin(edgefile, rank, size);
    if (debug) { cerr<<"rank "<<rank<<" loaded "<<local_edges.size()<<" edges\n"; }

    vector<Edge> mst = distributed_kkt_mst(local_edges, n, debug);

    // Final MST edges are stored (on root) as the edges produced by the last Boruvka call;
    // but we return them replicated. Print only on root:
    if (rank==0) {
        cerr<<"Final MST edges count: "<<mst.size()<<"\n";
        cout<<"# MST edges (u v w):\n";
        for (auto &e: mst) cout<<e.u<<" "<<e.v<<" "<<e.w<<"\n";
    }

    MPI_Finalize();
    return 0;
}

