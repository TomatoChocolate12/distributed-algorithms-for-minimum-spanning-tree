#include <bits/stdc++.h>
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char* argv[]){
    
    if(argc < 3){
        cerr << "Usage: " << argv[0] << " <ground truth file> <ghs file>" << endl;
        return 1;
    }
    
    ifstream file1(argv[1]);
    ifstream file2(argv[2]);

    if(!file1.is_open()){
        cerr << "Error: Could not open file " << argv[1] << endl;
        return 1;
    }
    if(!file2.is_open()){
        cerr << "Error: Could not open file " << argv[2] << endl;
        return 1;
    }

    string line1, line2;
    int line_number = 1;
    bool files_are_equal = true;

    vector<pair<int, pair<int, int> > > kruskal, ghs;

    while(getline(file1, line1)){
        stringstream iss(line1);
        int u, v, w;
        iss >> u >> v >> w;
        kruskal.push_back({w, {u, v}});
    }

    while(getline(file2, line2)){
        stringstream iss(line2);
        int u, v, w;
        iss >> u >> v >> w;
        ghs.push_back({w, {u, v}});
    }

    sort(kruskal.begin(), kruskal.end());
    sort(ghs.begin(), ghs.end());
    if(kruskal.size() != ghs.size()){
        cout << "Mismatch in number of edges: " << kruskal.size() << " (Kruskal) vs " << ghs.size() << " (GHS)" << endl;
        return 1;
    }

    for(size_t i = 0; i < kruskal.size(); i++){
        if(kruskal[i] != ghs[i]){
            cout << "Mismatch found at edge " << i+1 << ":" << endl;
            cout << "Kruskal: " << kruskal[i].second.first << " " << kruskal[i].second.second << " " << kruskal[i].first << endl;
            cout << "GHS:     " << ghs[i].second.first << " " << ghs[i].second.second << " " << ghs[i].first << endl;
            files_are_equal = false;
        }
    }

    if(files_are_equal){
        cout << "The files are identical." << endl;
    } else {
        cout << "The files differ." << endl;
    }

}