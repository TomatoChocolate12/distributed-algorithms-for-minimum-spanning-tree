#include <bits/stdc++.h>
#include "json.hpp" // Requires nlohmann/json library

using namespace std;

const int N = 1e3+5;
vector<pair<int, pair<int, int> > > edges;
int component[N];

void loadGraphFromJSON(const string& filename) {
    ifstream f(filename);
    
    if(!f.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    // parse the json
    nlohmann::json data;
    f >> data;
    // assuming the graph is represented as an adjacency list
    for(auto& element : data.items()) {
        int node = stoi(element.key());
        for(auto& neighbor : element.value().items()) {
            int neighborId = stoi(neighbor.key());
            int weight = neighbor.value();
            edges.push_back({weight, {node, neighborId}}); // just store weights for simplicity
        }
    }
}

int find(int x){
    while(true){
        if(x == component[x])
            return x;
        component[x] = component[component[x]];
        x = component[x];
    }
}

void merge(int u, int v){
    u = find(u), v = find(v);
    if(u != v)
        component[v] = u;
}

int main(int argc, char **argv){
    // load the graph from the json file
    if(argc < 3){
        cerr << "Usage: ./kruskals <json_file> <num_nodes>" << endl;
        return 1;
    }
    loadGraphFromJSON(argv[1]);
    int n = stoi(argv[2]);
    // Kruskal's algorithm
    for(int i = 0; i < n; ++i)
        component[i] = i;
    sort(edges.begin(), edges.end());
    vector<pair<int, pair<int, int> > > mst;
    for(auto& edge : edges){
        int w = edge.first;
        int u = edge.second.first;
        int v = edge.second.second;
        if(find(u) != find(v)){
            merge(u, v);
            mst.push_back({w, {u, v}});
        }
    }

    ofstream outfile(to_string(n) + "_kruskals.txt");
    for(auto& edge : mst){
        int w = edge.first;
        int u = edge.second.first;
        int v = edge.second.second;
        outfile << u << "\t" << v << "\t" << w << endl;
    }

    return 0;
}