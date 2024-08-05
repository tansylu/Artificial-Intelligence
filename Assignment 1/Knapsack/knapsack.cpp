#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <fstream>
#include <chrono>
using namespace std;

struct Item {
    int id;
    int b;
    int w;
};

struct Node {
    int depth;
    int benefit;
    int weight;
    vector<int> set;
};

void bfs(vector<Item>& items, int maxWeight) {
    queue<Node> q;
    Node source = {0, 0, 0, {}};
    q.push(source);

    int bestBenefit = 0;
    vector<int> optimalIncludedItems;

    while (!q.empty()) {
        Node currNode = q.front();
        q.pop();

        if (currNode.depth == items.size() || currNode.weight > maxWeight) {
            if (currNode.benefit > bestBenefit && currNode.weight <= maxWeight) {
                bestBenefit = currNode.benefit;
                optimalIncludedItems = currNode.set;
            }
            continue;
        }
        //represents state where current node is not included
        Node leftChild = currNode;
        leftChild.depth++;
        q.push(leftChild);
        //represents state where current node is included
        Node rightChild = currNode;
        rightChild.depth++;
        rightChild.weight += items[currNode.depth].w;
        rightChild.benefit += items[currNode.depth].b;
        rightChild.set.push_back(items[currNode.depth].id);
        q.push(rightChild);
        }

    cout << "Result: " << bestBenefit << "\n";
    cout << "Included items: ";
    for (int id : optimalIncludedItems) {
        cout << id << " ";
    }
    cout << "\n";
}

void dfs(vector<Item>& items, int maxWeight) {
    stack<Node> S;
    Node source = {0, 0, 0, {}};
    S.push(source);

    int bestBenefit = 0;
    vector<int> optimalIncludedItems;

    while (!S.empty()) {
        Node currNode = S.top();
        S.pop();

        if (currNode.depth == items.size() || currNode.weight > maxWeight) {
            if (currNode.benefit > bestBenefit && currNode.weight <= maxWeight) {
                bestBenefit = currNode.benefit;
                optimalIncludedItems = currNode.set;
            }
            continue;
        }

        Node leftChild = {currNode.depth + 1, currNode.benefit, currNode.weight, currNode.set};
        S.push(leftChild);

        Node rightChild = {currNode.depth + 1, currNode.benefit + items[currNode.depth].b, currNode.weight + items[currNode.depth].w, currNode.set};
        rightChild.set.push_back(items[currNode.depth].id);
        S.push(rightChild);
    }

    cout << "Result: " << bestBenefit << "\n";
    cout << "Included items: ";
    for (int id : optimalIncludedItems) {
        cout << id << " ";
    }
    cout << "\n";
}

int main() {
    int maxWeight = 420;
    ifstream f("input.txt");
    if (!f.is_open()) {
        cout << "Error opening file\n";
        return 1;
    }
    string l;
    getline(f, l);
    vector<Item> items; 
    Item curr; 
    while (f >> curr.id >> curr.b >> curr.w) {
        items.push_back(curr);
        cout << curr.id << " " << curr.b << " " << curr.w << "\n";
    }

    f.close();

    auto start = std::chrono::high_resolution_clock::now();
    bfs(items, maxWeight);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> bfsTime = end - start;
    std::cout << "BFS time: " << bfsTime.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    dfs(items, maxWeight);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dfsTime = end - start;
    std::cout << "DFS time: " << dfsTime.count() << " seconds\n";

    return 0;
}