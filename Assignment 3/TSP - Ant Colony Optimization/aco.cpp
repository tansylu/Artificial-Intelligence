#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include <set>
#include <algorithm>
#include <chrono>
using namespace std;

struct Location {
    int id;
    double x;
    double y;
};

struct Individual {
    vector<int> order;
    double distance;
};

vector<Location> parse(const string& filename) {
    vector<Location> locations;
    ifstream inputFile(filename);

    if (!inputFile) {
        cout << "error opening file";
        exit(1);   
    }

    Location temp;
    while (inputFile >> temp.id >> temp.x >> temp.y) {
        locations.push_back(temp);
    }

    inputFile.close();

    return locations;
}



/*
ACO algorithm description (TSP) 
• Initialize the pheromone intensity on the edges
 • Position ants at different cities 
 • Each ant moves from city to city based on probability
  • After all ants complete their paths, calculate their length and save the shortest path. 
  Update the pheromone on all edges 
  • Iterate untill the maximum number of iterations or covergence 
*/

class Berlin {
private:
    mt19937 rng;
    vector<vector<double>> pheromones;
    vector<vector<double>> heuristics;
    int n = 0;
    double best_res= 1e9;
    double alpha = 1.0;
    double beta = 2.0;

public:
    Berlin() {
        rng.seed(random_device()());
        vector<Location> locations = parse("input.txt");
        n = locations.size();
        heuristics = vector<vector<double>>(n, vector<double>(n));
        pheromones = vector<vector<double>>(n, vector<double>(n, 1.0)); // Initialize all  to 1

        // Calculate distances between each pair of locations
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double h = sqrt(pow(locations[i].x - locations[j].x, 2) + pow(locations[i].y - locations[j].y, 2));
                heuristics[i][j] = h;
                heuristics[j][i] = h;
            }
        }
    }

    double get_edge_probability(int x, int y) {
        return pow(pheromones[x][y], alpha) * pow(heuristics[x][y], -beta);
    }

    void get_random_path_from(int currentCity, vector<int> & path, double & res_length) {
        vector<int> temp_path;
        double distance = 0.0;
        temp_path.push_back(currentCity);
        int curr_temp = currentCity;
        while (temp_path.size() < n) {
            double total_p = 0.0;
            vector<int> possible_next_city;
            for (int k = 0; k < n; ++k) {
                if (find(temp_path.begin(), temp_path.end(), k) != temp_path.end()) // already visited
                    continue;
                total_p += get_edge_probability(curr_temp, k);
                possible_next_city.push_back(k);
            }
            uniform_real_distribution<double> distribution(0.0, total_p);
            double random_num = distribution(rng);
            double x = 0.0;
            for (int c : possible_next_city) {
                x += get_edge_probability(curr_temp, c);
                if (random_num <= x) {
                    distance += heuristics[curr_temp][c];
                    curr_temp = c;
                    temp_path.push_back(c);
                    break;
                }
            }
        }
        temp_path.push_back(currentCity);
        distance += heuristics[curr_temp][currentCity];
        path = temp_path;
        res_length = distance;
    }


pair<vector<int>, double> run_aco_ants(int number_ants, int generations) {
    double best_res = std::numeric_limits<double>::max();
    vector<int> best_path = {};
    vector<double> best_results;

    for (int gen = 0; gen < generations; ++gen) {
        // Evaporate
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                pheromones[i][j] *= 0.981;

        auto new_pheromones = pheromones; // copy weights
        // Run the whole batch
        for (int q = 0; q < number_ants; ++q) {
            vector<int> path = {};
            double res_length = 0;
            get_random_path_from(1, path, res_length);

            if (res_length < best_res) {
                best_res = res_length;
                best_path = path;
            }

            double diff = res_length - best_res + 0.05;
            double deposit_amount = 0.01 / diff;
            for (int i = 0; i < n + 1; ++i) {
                int x = path[i % n];
                int y = path[(i + 1) % n];
                new_pheromones[x][y] += deposit_amount;
                new_pheromones[x][y] += deposit_amount;
            }
        }

        // Update the weights after normalizing
        for (int i = 0; i < n; ++i) {
            double total_edges = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i == j)
                    continue;
                total_edges += new_pheromones[i][j];
            }
            for (int j = 0; j < n; ++j) {
                // multiplying by 2 since every node has two neighbors eventually
                pheromones[i][j] = 2 * new_pheromones[i][j] / total_edges;
            }
        }
        best_results.push_back(best_res);
    }
    std::ofstream file("best_results.txt");
    for (const auto &res : best_results) {
        file << res << "\n";
        cout << res <<endl;
    }
    file.close();
    return {best_path, best_res};}
};

int main() {
    Berlin berlin;
    pair<vector<int>, double> best = {{}, std::numeric_limits<double>::max()};
    best = berlin.run_aco_ants(100, 40);
    cout << "Best Result: " << best.second << endl;


    // Print the best path
    cout << "Best path: ";
    for (int city : best.first) {
        cout << city << " ";
    }
    cout << endl;

    return 0;
}