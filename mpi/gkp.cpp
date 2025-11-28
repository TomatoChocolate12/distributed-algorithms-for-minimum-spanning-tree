#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <limits>

using namespace std;

struct Edge {
    int u, v;
    double weight;
    
    bool operator<(const Edge& other) const {
        if (weight != other.weight) return weight < other.weight;
        if (u != other.u) return u < other.u;
        return v < other.v;
    }
};

struct MergeRequest {
    int from_fragment;
    int from_node;
    double edge_weight;
    int to_node;
};

struct Statistics {
    int messages_sent;
    int messages_received;
    double start_time;
    double end_time;
    int phases_executed;
    
    Statistics() : messages_sent(0), messages_received(0), 
                   start_time(0), end_time(0), phases_executed(0) {}
};

class GKPNode {
private:
    int rank, size;
    vector<Edge> edges;
    int fragment_id;
    int parent;
    vector<int> children;
    set<pair<int,int>> mst_edges_local;
    
    int phase;
    int max_phases;
    double threshold;
    
    Statistics stats;
    
public:
    GKPNode(int r, int s) : rank(r), size(s) {
        fragment_id = rank;
        parent = -1;
        phase = 0;
        max_phases = max(1, (int)ceil(log2(s)));
    }
    
    void addEdge(int u, int v, double w) {
        edges.push_back({u, v, w});
    }
    
    void computeThreshold() {
        threshold = pow(2.0, phase) * 0.5;
    }
    
    void run() {
        stats.start_time = MPI_Wtime();
        
        for (phase = 0; phase < max_phases; phase++) {
            computeThreshold();
            executePhase();
            stats.phases_executed++;
            
            MPI_Barrier(MPI_COMM_WORLD);
            
            // Check if all nodes are in same fragment
            int all_same = checkConvergence();
            if (all_same) break;
        }
        
        stats.end_time = MPI_Wtime();
        collectMST();
    }
    
    void executePhase() {
        // Step 1: Find minimum outgoing edge
        Edge min_edge = findMinOutgoingEdge();
        
        // Step 2: Handle fragment queries
        handleFragmentQueries();
        
        // Step 3: Send merge requests
        if (min_edge.weight < numeric_limits<double>::infinity() && 
            shouldMerge(min_edge.weight)) {
            requestMerge(min_edge);
        }
        
        // Step 4: Process merge requests
        processMergeRequests();
        
        // Step 5: Update fragment structure
        broadcastFragmentUpdate();
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    Edge findMinOutgoingEdge() {
        Edge min_edge = {-1, -1, numeric_limits<double>::infinity()};
        
        for (const auto& e : edges) {
            int neighbor_frag = getFragmentID(e.v);
            
            if (neighbor_frag != fragment_id && e.weight < min_edge.weight) {
                min_edge = e;
            }
        }
        
        return min_edge;
    }
    
    int getFragmentID(int node) {
        if (node == rank) {
            return fragment_id;
        }
        
        // Send query
        int query_type = 1;
        MPI_Send(&query_type, 1, MPI_INT, node, 100, MPI_COMM_WORLD);
        stats.messages_sent++;
        
        // Receive response
        int frag_id;
        MPI_Recv(&frag_id, 1, MPI_INT, node, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        stats.messages_received++;
        
        return frag_id;
    }
    
    void handleFragmentQueries() {
        MPI_Status status;
        int flag;
        
        // Non-blocking check for queries
        for (int i = 0; i < size * 2; i++) {
            MPI_Iprobe(MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;
            
            int query;
            MPI_Recv(&query, 1, MPI_INT, status.MPI_SOURCE, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            stats.messages_received++;
            
            // Send fragment ID
            MPI_Send(&fragment_id, 1, MPI_INT, status.MPI_SOURCE, 101, MPI_COMM_WORLD);
            stats.messages_sent++;
        }
    }
    
    bool shouldMerge(double edge_weight) {
        return edge_weight <= threshold || phase >= max_phases - 1;
    }
    
    void requestMerge(const Edge& e) {
        MergeRequest req;
        req.from_fragment = fragment_id;
        req.from_node = rank;
        req.edge_weight = e.weight;
        req.to_node = e.v;
        
        MPI_Send(&req, sizeof(MergeRequest), MPI_BYTE, e.v, 200, MPI_COMM_WORLD);
        stats.messages_sent++;
    }
    
    void processMergeRequests() {
        MPI_Status status;
        int flag;
        
        vector<MergeRequest> requests;
        
        // Collect all merge requests
        for (int i = 0; i < size * 2; i++) {
            MPI_Iprobe(MPI_ANY_SOURCE, 200, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;
            
            MergeRequest req;
            MPI_Recv(&req, sizeof(MergeRequest), MPI_BYTE, status.MPI_SOURCE, 200, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            stats.messages_received++;
            
            if (req.from_fragment != fragment_id) {
                requests.push_back(req);
            }
        }
        
        // Process requests - accept the one with minimum weight
        if (!requests.empty()) {
            auto best = min_element(requests.begin(), requests.end(),
                [](const MergeRequest& a, const MergeRequest& b) {
                    return a.edge_weight < b.edge_weight;
                });
            
            acceptMerge(best->from_fragment, best->from_node);
        }
    }
    
    void acceptMerge(int other_fragment, int other_node) {
        int new_fragment_id = min(fragment_id, other_fragment);
        
        // Update parent-child relationships
        if (fragment_id > other_fragment) {
            parent = other_node;
        } else if (fragment_id < other_fragment) {
            children.push_back(other_node);
        }
        
        // Add edge to MST
        int u = min(rank, other_node);
        int v = max(rank, other_node);
        mst_edges_local.insert({u, v});
        
        fragment_id = new_fragment_id;
    }
    
    void broadcastFragmentUpdate() {
        // Send to children
        for (int child : children) {
            MPI_Send(&fragment_id, 1, MPI_INT, child, 300, MPI_COMM_WORLD);
            stats.messages_sent++;
        }
        
        // Receive from parent
        if (parent != -1) {
            MPI_Status status;
            int flag;
            MPI_Iprobe(parent, 300, MPI_COMM_WORLD, &flag, &status);
            
            if (flag) {
                int new_frag_id;
                MPI_Recv(&new_frag_id, 1, MPI_INT, parent, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                stats.messages_received++;
                fragment_id = new_frag_id;
                
                // Propagate to children
                for (int child : children) {
                    MPI_Send(&fragment_id, 1, MPI_INT, child, 300, MPI_COMM_WORLD);
                    stats.messages_sent++;
                }
            }
        }
    }
    
    int checkConvergence() {
        int local_frag = fragment_id;
        int min_frag, max_frag;
        
        MPI_Allreduce(&local_frag, &min_frag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_frag, &max_frag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        return (min_frag == max_frag) ? 1 : 0;
    }
    
    void collectMST() {
        vector<pair<int,int>> local_mst(mst_edges_local.begin(), mst_edges_local.end());
        int local_count = local_mst.size();
        
        vector<int> all_counts(size);
        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            vector<int> displs(size, 0);
            int total = 0;
            for (int i = 0; i < size; i++) {
                displs[i] = total;
                total += all_counts[i];
            }
            
            vector<pair<int,int>> all_edges(total);
            MPI_Gatherv(local_mst.data(), local_count * 2, MPI_INT,
                       all_edges.data(), nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Reconstruct and find weights
            cout << "\n=== GKP Algorithm Results ===" << endl;
            cout << "MST Edges:" << endl;
            
            set<pair<int,int>> unique_edges(all_edges.begin(), all_edges.end());
            double total_weight = 0;
            
            for (const auto& [u, v] : unique_edges) {
                // Find weight from edges
                double w = 0;
                for (const auto& e : edges) {
                    if ((e.u == u && e.v == v) || (e.u == v && e.v == u)) {
                        w = e.weight;
                        break;
                    }
                }
                cout << "  (" << u << " - " << v << ") weight: " << w << endl;
                total_weight += w;
            }
            
            cout << "Total MST Weight: " << total_weight << endl;
            cout << "Phases Executed: " << stats.phases_executed << endl;
            cout << "Execution Time: " << (stats.end_time - stats.start_time) << " seconds" << endl;
            cout << "Messages Sent (Rank 0): " << stats.messages_sent << endl;
            cout << "Messages Received (Rank 0): " << stats.messages_received << endl;
        } else {
            MPI_Gatherv(local_mst.data(), local_count * 2, MPI_INT,
                       nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    
    Statistics getStatistics() {
        return stats;
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    GKPNode node(rank, size);
    
    // Example graph
    if (rank == 0) {
        node.addEdge(0, 1, 1.0);
        node.addEdge(0, 2, 4.0);
    } else if (rank == 1) {
        node.addEdge(1, 0, 1.0);
        node.addEdge(1, 2, 2.0);
        node.addEdge(1, 3, 5.0);
    } else if (rank == 2) {
        node.addEdge(2, 0, 4.0);
        node.addEdge(2, 1, 2.0);
        node.addEdge(2, 3, 1.5);
    } else if (rank == 3) {
        node.addEdge(3, 1, 5.0);
        node.addEdge(3, 2, 1.5);
    }
    
    node.run();
    
    MPI_Finalize();
    return 0;
}

