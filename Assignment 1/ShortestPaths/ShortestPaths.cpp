#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
using namespace std;

struct Path {
    string cityA;
    string cityB;
    int distance;
};

struct Node {
    string city;
    int h;
    int distance;
};

vector<Path> getPaths(ifstream& inputFile) {
    vector<Path> paths;
    string cityA, cityB;
    int distance;
    string dummyLine;
    getline(inputFile, dummyLine);

    while (inputFile >> cityA >> cityB >> distance) {
        Path path;
        path.cityA = cityA;
        path.cityB = cityB;
        path.distance = distance;
        paths.push_back(path);
    }

    return paths;
}

map<string, int> getHeuristics(ifstream& inputFile) {
    map<string, int> heuristics;
    string city;
    int h;

    string x;
    getline(inputFile, x);

    while (inputFile >> city >> h) {
        heuristics[city] = h;
    }

    return heuristics;
}

void printPaths(const vector<Path>& paths) {
  
    for (const auto& path : paths) {
        cout << path.cityA << " " << path.cityB << " " << path.distance << endl;
    }
}

void A_search(string target, vector<Path>& paths, Node parent, map<string, int>& heuristics) {
    if(parent.city == target){
                cout<<target<<" reached"<<endl;
                return;
            }
    int min_val = INT_MAX;
    Node next_node;
    for (Path& path : paths) {
        if (path.cityA == parent.city || path.cityB == parent.city ) {
            Node child;
            string outgoing = path.cityA == parent.city ? path.cityB : path.cityA;
            child.city = outgoing;
            child.distance = parent.distance + path.distance;
            child.h = heuristics[outgoing];
            //compare child's estimate distance to target plus the distance from the parent 
            //to the child with current minimum
            if(child.h+child.distance < min_val){
                min_val = child.h+child.distance;
                next_node = child;
            }
        
        }
    }
    cout<<next_node.city<< " Total distance: " <<next_node.distance<<endl;
    A_search(target, paths, next_node, heuristics);

}

void greedy_search(string target, vector<Path>& paths, Node parent, map<string, int>& heuristics) {
    int min_h = INT_MAX;
    Node next_node;
    for (Path& path : paths) {
        if (path.cityA == parent.city || path.cityB == parent.city ) {
            Node child;
            string outgoing = path.cityA == parent.city ? path.cityB : path.cityA;
            child.city = outgoing;
            if(outgoing == target){
                cout<<target<<" reached, total distance: "<<parent.distance + path.distance<<endl;
                return;
            }
            child.distance = parent.distance + path.distance;
            child.h = heuristics[outgoing];
            //compare child's estimate distance to target with current minimum
            if(child.h<min_h){
                min_h = child.h;
                next_node = child;
            }
        
        }
    }
    cout<<next_node.city<< " Total distance: " <<next_node.distance<<endl;
    greedy_search(target, paths, next_node, heuristics);

}

int main() {
    ifstream inputFile1("input2.txt");
    vector<Path> paths = getPaths(inputFile1);
    ifstream inputFile2("input3.txt");
    map<string, int> heuristics = getHeuristics(inputFile2);
    Node root;
    root.city = "Malaga";
    root.h = heuristics["Malaga"];
    root.distance = 0;
    cout << "Greedy search started: "<<endl;
    cout<<root.city<< " Total distance: " <<root.distance<<endl;
    greedy_search("Valladolid", paths, root, heuristics);
    cout << endl;
    cout << "A* search started: "<<endl;
    cout<<root.city<< " Total distance: " <<root.distance<<endl;
    A_search("Valladolid", paths, root, heuristics);
    return 0;
}