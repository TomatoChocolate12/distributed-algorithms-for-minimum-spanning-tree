// ==========================
// GHS WITH MESSAGE COUNTING
// ==========================

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <algorithm>
#include <tuple>
#include <mpi.h>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

#define INF 100000
int num_nodes;

enum class Tag {
    CONNECT = 1,
    INITIATE = 2,
    TEST = 3,
    ACCREJ = 4,
    REPORT = 5,
    CHANGEROOT = 6,
    TERMINATE = 7
};

enum class EdgeState { BASIC = 0, BRANCH = 1, REJECTED = -1 };
enum class NodeState { SLEEPING = 0, FIND = 1, FOUND = 2 };

struct Edge {
    int weight;
    EdgeState state;
    int neighbor;
    bool operator<(const Edge& other) const { return weight < other.weight; }
};

struct MessageArgs {
    int sender;
    vector<int> payload;
};

struct Metrics {
    double startTime;
    double endTime;
    double processingTime;
    long long messagesSent;
    long long bytesSent;
    int maxLevelReached;
    Metrics() : startTime(0), endTime(0), processingTime(0),
                messagesSent(0), bytesSent(0), maxLevelReached(0) {}
};


class GHSProcess {
private:
    int rank, size;
    double startTime;

    Metrics perf;

    // Node state
    int level = 0;
    int fragmentName = 0;
    int parent = -1;
    NodeState state = NodeState::SLEEPING;

    // GHS state
    int bestWt = INF, bestNode = -1, rec = 0;
    int testNode = -1;
    bool halt = false;

    bool connectWait = false;
    bool reportWait = false;
    bool testWait = false;

    vector<Edge> edges;
    queue<pair<Tag, MessageArgs>> waitQueue;

    // ================================
    // Wrapped MPI_Send (THIS WAS MISSING)
    // ================================
    void sendMsg(void* data, int count, int dest, Tag tag) {
        perf.messagesSent++;
        perf.bytesSent += count * sizeof(int);
        MPI_Send(data, count, MPI_INT, dest, static_cast<int>(tag), MPI_COMM_WORLD);
    }

public:
    GHSProcess(int r, int s) : rank(r), size(s) {
        startTime = MPI_Wtime();
        fragmentName = r;
    }

    void loadGraphFromJSON(const string& filename) {
        ifstream f(filename);
        json data; f >> data;
        string me = to_string(rank);

        if (data.contains(me)) {
            for (auto& e : data[me].items()) {
                int nbr = stoi(e.key());
                int w = e.value();
                edges.push_back({w, EdgeState::BASIC, nbr});
            }
            num_nodes = max(num_nodes, rank + 1);
        }
        sort(edges.begin(), edges.end());
    }

    void run() {
        MPI_Barrier(MPI_COMM_WORLD);
        perf.startTime = MPI_Wtime();
        initialize();

        while (!halt) {
            MPI_Status st;
            int flag = 0;
            double t0 = MPI_Wtime();
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);

            if (flag) {
                perf.processingTime += MPI_Wtime() - t0;
                Tag tag = static_cast<Tag>(st.MPI_TAG);
                int src = st.MPI_SOURCE;
                int count;
                MPI_Get_count(&st, MPI_INT, &count);

                vector<int> buf(count);
                MPI_Recv(buf.data(), count, MPI_INT, src, (int)tag,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                dispatchMessage(tag, src, buf);
            }
            processWaitQueue();
        }

        perf.endTime = MPI_Wtime();
        perf.maxLevelReached = level;

        string f = to_string(size) + "_ghs.txt";

        if (rank == 0) { ofstream q(f, ios::trunc); }
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < size; i++) {
            if (i == rank) printMST(f);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        collectAndPrintMetrics();
    }

private:
    int getEdgeIndex(int n) {
        for (int i = 0; i < (int)edges.size(); i++)
            if (edges[i].neighbor == n) return i;
        return -1;
    }

    void processWaitQueue() {
        if (!waitQueue.empty()) {
            auto x = waitQueue.front();
            waitQueue.pop();
            dispatchMessage(x.first, x.second.sender, x.second.payload);
        }
    }

    // ========================
    // INITIALIZE
    // ========================
    void initialize() {
        state = NodeState::FOUND;
        rec = 0;

        if (edges.empty()) { halt = true; return; }

        Edge &e = edges[0];
        e.state = EdgeState::BRANCH;

        int msg = 0;
        connectWait = true;
        sendMsg(&msg, 1, e.neighbor, Tag::CONNECT);
    }

    // ========================
    // DISPATCH
    // ========================
    void dispatchMessage(Tag tag, int src, const vector<int>& p) {
        switch(tag) {
            case Tag::CONNECT:   handleConnect(src, p[0]); break;
            case Tag::INITIATE:  handleInitiate(src, p[0], p[1], p[2]); break;
            case Tag::TEST:      handleTest(src, p[0], p[1]); break;
            case Tag::ACCREJ:    handleAccRej(src, p[0]); break;
            case Tag::REPORT:    handleReport(src, p[0]); break;
            case Tag::CHANGEROOT:handleChangeRoot(); break;
            case Tag::TERMINATE: handleTerminate(src); break;
        }
    }

    // ========================
    // CONNECT
    // ========================
    void handleConnect(int src, int L) {
        int idx = getEdgeIndex(src);
        if (idx == -1) return;

        if (L < level) {
            edges[idx].state = EdgeState::BRANCH;
            int a[3] = {level, fragmentName, (int)state};
            sendMsg(a, 3, src, Tag::INITIATE);
        }
        else if (edges[idx].state == EdgeState::BASIC) {
            waitQueue.push({Tag::CONNECT, {src, {L}}});
        }
        else {
            int a[3] = {level+1, edges[idx].weight, (int)NodeState::FIND};
            sendMsg(a, 3, src, Tag::INITIATE);
        }
    }

    // ========================
    // INITIATE
    // ========================
    void handleInitiate(int src, int L, int F, int S) {
        level = L;
        fragmentName = F;
        state = (NodeState)S;
        parent = src;

        bestWt = INF;
        bestNode = -1;
        testNode = -1;
        connectWait = false;
        reportWait = false;

        for (auto &e : edges) {
            if (e.state == EdgeState::BRANCH && e.neighbor != src) {
                int a[3] = {L,F,S};
                sendMsg(a, 3, e.neighbor, Tag::INITIATE);
            }
        }

        if (state == NodeState::FIND) {
            rec = 0;
            findMin();
        }
    }

    // ========================
    // TEST
    // ========================
    void handleTest(int src, int L, int F) {
        if (L > level) {
            waitQueue.push({Tag::TEST, {src, {L,F}}});
            return;
        }

        int idx = getEdgeIndex(src);

        if (F == fragmentName) {
            if (idx != -1 && edges[idx].state == EdgeState::BASIC)
                edges[idx].state = EdgeState::REJECTED;

            int msg = -1;
            sendMsg(&msg, 1, src, Tag::ACCREJ);

            if (src == testNode)
                findMin();
        }
        else {
            int msg = 1;
            sendMsg(&msg, 1, src, Tag::ACCREJ);
        }
    }

    // ========================
    // ACCEPT / REJECT
    // ========================
    void handleAccRej(int src, int ds) {
        testWait = false;
        int idx = getEdgeIndex(src);

        if (ds == -1) {
            if (idx != -1 && edges[idx].state == EdgeState::BASIC)
                edges[idx].state = EdgeState::REJECTED;
            findMin();
        } else {
            testNode = -1;
            if (idx != -1 && edges[idx].weight < bestWt) {
                bestWt = edges[idx].weight;
                bestNode = src;
            }
            checkReport();
        }
    }

    // ========================
    // REPORT
    // ========================
    void handleReport(int src, int w) {
        if (src != parent) {
            if (w < bestWt) { bestWt = w; bestNode = src; }
            rec++;
            checkReport();
        }
        else {
            if (state == NodeState::FIND)
                waitQueue.push({Tag::REPORT, {src, {w}}});
            else if (w > bestWt)
                handleChangeRoot();
            else if (w == bestWt && bestWt == INF) {
                halt = true;
                int msg = 0;
                for (auto &e : edges)
                    if (e.state == EdgeState::BRANCH)
                        sendMsg(&msg, 1, e.neighbor, Tag::TERMINATE);
            }
        }
    }

    // ========================
    // CHANGE ROOT
    // ========================
    void handleChangeRoot() {
        int idx = getEdgeIndex(bestNode);
        if (idx == -1) return;

        if (edges[idx].state == EdgeState::BRANCH) {
            int msg = 0;
            sendMsg(&msg, 1, bestNode, Tag::CHANGEROOT);
        }
        else {
            edges[idx].state = EdgeState::BRANCH;
            int msg = level;
            connectWait = true;
            sendMsg(&msg, 1, bestNode, Tag::CONNECT);
        }
    }

    // ========================
    // TERMINATE
    // ========================
    void handleTerminate(int src) {
        halt = true;
        int msg = 0;
        for (auto &e : edges)
            if (e.state == EdgeState::BRANCH && e.neighbor != src)
                sendMsg(&msg, 1, e.neighbor, Tag::TERMINATE);
    }

    // ========================
    // FIND MIN
    // ========================
    void findMin() {
        for (auto &e : edges) {
            if (e.state == EdgeState::BASIC && !testWait) {
                testWait = true;
                testNode = e.neighbor;
                int a[2] = {level, fragmentName};
                sendMsg(a, 2, testNode, Tag::TEST);
                return;
            }
        }
        testNode = -1;
        checkReport();
    }

    // ========================
    // CHECK REPORT
    // ========================
    void checkReport() {
        int children = 0;
        for (auto &e : edges)
            if (e.state == EdgeState::BRANCH && e.neighbor != parent)
                children++;

        if (rec == children && testNode == -1 && !reportWait) {
            reportWait = true;
            state = NodeState::FOUND;
            int msg = bestWt;
            sendMsg(&msg, 1, parent, Tag::REPORT);
        }
    }

    // ========================
    // PRINT MST
    // ========================
    void printMST(const string& f) {
        ofstream o(f, ios::app);
        for (auto &e : edges)
            if (e.state == EdgeState::BRANCH && rank < e.neighbor)
                o << rank << "\t" << e.neighbor << "\t" << e.weight << "\n";
    }

    // ========================
    // METRICS
    // ========================
    void collectAndPrintMetrics() {
        double localTime = perf.endTime - perf.startTime;
        long long localMsg = perf.messagesSent;
        long long localBytes = perf.bytesSent;
        int localLvl = perf.maxLevelReached;

        double maxT;
        long long totM, totB;
        int maxLvl;

        MPI_Reduce(&localTime, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMsg, &totM, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localBytes, &totB, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localLvl, &maxLvl, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "\n=== Distributed GHS Benchmark ===\n";
            cout << "Nodes: " << size << "\n";
            cout << "Time Taken: " << maxT << " seconds\n";
            cout << "Total Messages: " << totM << "\n";
            cout << "Total Data Sent: " << totB << " bytes\n";
            cout << "Avg Data/Node: " << (totB / size) << " bytes\n";
            cout << "Max Level: " << maxLvl << "\n";
            cout << "=================================\n";
        }
    }
};


// ====================================
// MAIN
// ====================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int r, s;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &s);

    if (argc < 2) {
        if (r == 0) cerr << "Usage: ./ghs <graph.json>\n";
        MPI_Finalize();
        return 0;
    }

    GHSProcess node(r, s);
    MPI_Barrier(MPI_COMM_WORLD);
    node.loadGraphFromJSON(argv[1]);
    MPI_Barrier(MPI_COMM_WORLD);

    node.run();
    MPI_Finalize();
    return 0;
}