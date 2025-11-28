#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <ctime>
#include <fstream>
#include <sstream>

using namespace std;

struct Edge {
    int u, v;
    double weight;
    int id;
    
    bool operator<(const Edge& other) const {
        if (weight != other.weight) return weight < other.weight;
        return id < other.id;
    }
};

enum MessageType {
    CONNECT, INITIATE, TEST, ACCEPT, REJECT, REPORT, CHANGEROOT, TERMINATE
};

struct Message {
    MessageType type;
    int level;
    int fragment_id;
    int state;
    double best_weight;
    int sender;
    int best_edge_id;
};

enum NodeState {
    SLEEPING, FIND, FOUND
};

struct Statistics {
    int messages_sent;
    int messages_received;
    double start_time;
    double end_time;
    
    Statistics() : messages_sent(0), messages_received(0), start_time(0), end_time(0) {}
};

class GHSNode {
private:
    int rank, size;
    vector<Edge> edges;
    map<int, int> neighbor_to_edge;
    
    int level;
    int fragment_id;
    NodeState state;
    
    int in_branch;
    int best_edge;
    double best_weight;
    int test_edge;
    int find_count;
    
    vector<int> branch_edges;
    set<int> rejected_edges;
    map<int, double> reported_weights;
    
    bool terminated;
    Statistics stats;
    
public:
    GHSNode(int r, int s) : rank(r), size(s) {
        level = 0;
        fragment_id = rank;
        state = SLEEPING;
        in_branch = -1;
        best_edge = -1;
        best_weight = numeric_limits<double>::infinity();
        test_edge = 0;
        find_count = 0;
        terminated = false;
    }
    
    void addEdge(int u, int v, double w, int id) {
        edges.push_back({u, v, w, id});
        neighbor_to_edge[v] = edges.size() - 1;
    }
    
    void sendMessage(const Message& msg, int dest, int tag = 0) {
        MPI_Send((void*)&msg, sizeof(Message), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
        stats.messages_sent++;
    }
    
    void wakeup() {
        if (state == SLEEPING && !edges.empty()) {
            auto min_edge = min_element(edges.begin(), edges.end());
            int min_idx = distance(edges.begin(), min_edge);
            
            Message msg = {CONNECT, 0, rank, 0, 0, rank, -1};
            sendMessage(msg, min_edge->v);
            
            branch_edges.push_back(min_idx);
            state = FOUND;
            level = 0;
        }
    }
    
    void handleConnect(Message& msg, int edge_idx) {
        if (state == SLEEPING) {
            wakeup();
        }
        
        if (msg.level < level) {
            branch_edges.push_back(edge_idx);
            Message response = {INITIATE, level, fragment_id, (int)state, 0, rank, -1};
            sendMessage(response, msg.sender);
            
            if (state == FIND) {
                find_count++;
            }
        } else {
            if (find(branch_edges.begin(), branch_edges.end(), edge_idx) == branch_edges.end()) {
                branch_edges.push_back(edge_idx);
            }
            
            Message response = {INITIATE, level + 1, edges[edge_idx].id, (int)FIND, 0, rank, -1};
            sendMessage(response, msg.sender);
        }
    }
    
    void handleInitiate(Message& msg, int edge_idx) {
        level = msg.level;
        fragment_id = msg.fragment_id;
        state = (NodeState)msg.state;
        in_branch = edge_idx;
        best_edge = -1;
        best_weight = numeric_limits<double>::infinity();
        
        for (int i : branch_edges) {
            if (i != in_branch) {
                Message fwd = {INITIATE, level, fragment_id, (int)state, 0, rank, -1};
                sendMessage(fwd, edges[i].v);
            }
        }
        
        if (state == FIND) {
            find_count = (int)branch_edges.size() - 1;
            test();
        }
    }
    
    void test() {
        while (test_edge < (int)edges.size()) {
            if (find(branch_edges.begin(), branch_edges.end(), test_edge) == branch_edges.end() &&
                rejected_edges.find(test_edge) == rejected_edges.end()) {
                
                Message msg = {TEST, level, fragment_id, 0, 0, rank, -1};
                sendMessage(msg, edges[test_edge].v);
                return;
            }
            test_edge++;
        }
        
        report();
    }
    
    void handleTest(Message& msg, int edge_idx) {
        if (state == SLEEPING) {
            wakeup();
        }
        
        if (msg.level > level) {
            // Defer - in simplified version, just accept
            Message response = {ACCEPT, 0, 0, 0, 0, rank, -1};
            sendMessage(response, msg.sender);
        } else if (msg.fragment_id != fragment_id) {
            Message response = {ACCEPT, 0, 0, 0, 0, rank, -1};
            sendMessage(response, msg.sender);
        } else {
            if (find(branch_edges.begin(), branch_edges.end(), edge_idx) == branch_edges.end()) {
                rejected_edges.insert(edge_idx);
            }
            Message response = {REJECT, 0, 0, 0, 0, rank, -1};
            sendMessage(response, msg.sender);
        }
    }
    
    void handleAccept(int edge_idx) {
        if (edges[test_edge].weight < best_weight) {
            best_edge = test_edge;
            best_weight = edges[test_edge].weight;
        }
        test_edge++;
        test();
    }
    
    void handleReject(int edge_idx) {
        rejected_edges.insert(test_edge);
        test_edge++;
        test();
    }
    
    void report() {
        if (find_count > 0) return;
        
        state = FOUND;
        if (in_branch != -1) {
            Message msg = {REPORT, 0, 0, 0, best_weight, rank, best_edge};
            sendMessage(msg, edges[in_branch].v);
        } else {
            // Root node
            if (best_weight < numeric_limits<double>::infinity() && best_edge != -1) {
                Message change = {CHANGEROOT, 0, 0, 0, 0, rank, -1};
                sendMessage(change, edges[best_edge].v);
            } else {
                checkTermination();
            }
        }
    }
    
    void handleReport(Message& msg) {
        if (msg.sender != rank) {
            reported_weights[msg.sender] = msg.best_weight;
        }
        
        if (msg.best_weight < best_weight) {
            best_weight = msg.best_weight;
            best_edge = msg.best_edge_id;
        }
        
        find_count--;
        if (find_count == 0 && test_edge >= (int)edges.size()) {
            report();
        }
    }
    
    void handleChangeRoot() {
        if (best_edge != -1 && best_edge < (int)edges.size()) {
            Message msg = {CONNECT, level, fragment_id, 0, 0, rank, -1};
            sendMessage(msg, edges[best_edge].v);
            branch_edges.push_back(best_edge);
        }
        checkTermination();
    }
    
    void checkTermination() {
        // Simple termination: if no more edges to explore
        if (best_weight >= numeric_limits<double>::infinity() || best_edge == -1) {
            terminated = true;
        }
    }
    
    void run(int max_iterations = 1000) {
        stats.start_time = MPI_Wtime();
        
        wakeup();
        
        int iterations = 0;
        while (!terminated && iterations < max_iterations) {
            MPI_Status status;
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            
            if (flag) {
                Message msg;
                MPI_Recv(&msg, sizeof(Message), MPI_BYTE, status.MPI_SOURCE, MPI_ANY_TAG, 
                        MPI_COMM_WORLD, &status);
                stats.messages_received++;
                
                int edge_idx = -1;
                if (neighbor_to_edge.count(msg.sender)) {
                    edge_idx = neighbor_to_edge[msg.sender];
                }
                
                switch (msg.type) {
                    case CONNECT:
                        handleConnect(msg, edge_idx);
                        break;
                    case INITIATE:
                        handleInitiate(msg, edge_idx);
                        break;
                    case TEST:
                        handleTest(msg, edge_idx);
                        break;
                    case ACCEPT:
                        handleAccept(edge_idx);
                        break;
                    case REJECT:
                        handleReject(edge_idx);
                        break;
                    case REPORT:
                        handleReport(msg);
                        break;
                    case CHANGEROOT:
                        handleChangeRoot();
                        break;
                    case TERMINATE:
                        terminated = true;
                        break;
                }
            }
            
            iterations++;
            
            // Check global termination periodically
            if (iterations % 100 == 0) {
                int local_term = terminated ? 1 : 0;
                int global_term;
                MPI_Allreduce(&local_term, &global_term, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                if (global_term) terminated = true;
            }
        }
        
        stats.end_time = MPI_Wtime();
        
        // Final synchronization
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    vector<Edge> getMSTEdges() {
        vector<Edge> mst;
        for (int idx : branch_edges) {
            if (idx >= 0 && idx < (int)edges.size()) {
                mst.push_back(edges[idx]);
            }
        }
        return mst;
    }
    
    Statistics getStatistics() {
        return stats;
    }
    
    void printResults() {
        if (rank == 0) {
            cout << "\n=== GHS Algorithm Results ===" << endl;
        }
        
        vector<Edge> mst = getMSTEdges();
        
        // Gather all MST edges
        int local_count = mst.size();
        vector<int> all_counts(size);
        MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            cout << "MST Edges:" << endl;
            double total_weight = 0;
            for (const auto& e : mst) {
                cout << "  (" << e.u << " - " << e.v << ") weight: " << e.weight << endl;
                total_weight += e.weight;
            }
            cout << "Total MST Weight: " << total_weight << endl;
            cout << "Execution Time: " << (stats.end_time - stats.start_time) << " seconds" << endl;
            cout << "Messages Sent (Rank 0): " << stats.messages_sent << endl;
            cout << "Messages Received (Rank 0): " << stats.messages_received << endl;
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    GHSNode node(rank, size);
    
    // Example graph
    if (rank == 0) {
        node.addEdge(0, 1, 1.0, 0);
        node.addEdge(0, 2, 4.0, 1);
    } else if (rank == 1) {
        node.addEdge(1, 0, 1.0, 0);
        node.addEdge(1, 2, 2.0, 2);
        node.addEdge(1, 3, 5.0, 3);
    } else if (rank == 2) {
        node.addEdge(2, 0, 4.0, 1);
        node.addEdge(2, 1, 2.0, 2);
        node.addEdge(2, 3, 1.5, 4);
    } else if (rank == 3) {
        node.addEdge(3, 1, 5.0, 3);
        node.addEdge(3, 2, 1.5, 4);
    }
    
    node.run();
    node.printResults();
    
    MPI_Finalize();
    return 0;
}
