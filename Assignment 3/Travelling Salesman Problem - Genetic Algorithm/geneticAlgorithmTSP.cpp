#include <fstream>
#include <vector>
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

Individual tournament(const std::vector<Individual>& population) {
    int k = 5; 
    Individual best;
    bool flag = true;
    std::set<int> selectedIndexes;

    for (int i = 0; i < k; ++i) {
        int index;
        do {
            index = std::rand() % population.size();
        } while (selectedIndexes.count(index) > 0); 

        selectedIndexes.insert(index);

        if (flag || population[index].distance < best.distance) {
            best = population[index];
            flag = false;
        }
    }

    return best;
}

vector<int> shuffler(vector<int>order) {
    int size = order.size();
    for (int i = 0; i < size; ++i) {
        int swapWith = i + rand() / (RAND_MAX / (size - i) + 1);
        swap(order[i], order[swapWith]);
    }
    return order;
}

double fitnessScore(const vector<int>& order, const vector<Location>& locations) {
    double score = 0;
    for (int i = 0; i < order.size() - 1; ++i) {
        score += sqrt(pow(locations[order[i+1]].x - locations[order[i]].x, 2) + pow(locations[order[i+1]].y - locations[order[i]].y, 2));
    }

    score += sqrt(pow(locations[order[0]].x - locations[order.back()].x, 2) + pow(locations[order[0]].y - locations[order.back()].y, 2));
    return score;
}

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

vector<Individual> initPopulation(const vector<int>& defaultOrder, int n, const vector<Location>& locations, Individual& best) {
    vector<Individual> population;
    double distance, bestDistance = 100000;
    for (int i = 0; i < n; ++i) {
        vector<int> newOrder =  shuffler(defaultOrder);
        distance = fitnessScore(newOrder, locations);
        if (distance < bestDistance) {
            bestDistance = distance;
            best.order = newOrder;
            best.distance = distance;
        }
        population.push_back({newOrder, distance});
    }
    return population;
}

vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    int size = parent1.size();

    int start = rand() % size;
    int end = rand() % size;
    if (start > end) swap(start, end);

    vector<int> offspring(size, -1);
    for (int i = start; i <= end; ++i) {
        offspring[i] = parent1[i];
    }

    for (int i = 0, j = 0; i < size && j < size; ++j) {

        if (find(offspring.begin(), offspring.end(), parent2[j]) != offspring.end()) continue;

        while (offspring[i] != -1) i++; 
        offspring[i++] = parent2[j];
    }

    return offspring;
}

vector<int> mutateInsert(vector<int> order) {
    int i = rand() % order.size();
    int j;
    do {
        j = rand() % order.size();
    } while (i == j);
    int gene = order[i];
    order.erase(order.begin() + i);
    order.insert(order.begin() + j, gene);
    return order;
}

void runGeneticAlgorithm(vector<Individual>& population, const vector<Location>& locations, Individual& best, int generationsTotal, int elite) {
    double mutRate = 0.20; 
    ofstream outFile("performance.txt");
    for (int generation = 0; generation < generationsTotal; ++generation) {
        vector<Individual> newPopulation;
        sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) { return a.distance < b.distance; });
        for (int i = 0; i < elite; ++i) {
            newPopulation.push_back(population[i]);
        }
        for (int i = elite; i < population.size(); ++i) {
            Individual parent1 = tournament(population);
            Individual parent2 = tournament(population);
            vector<int> childOrder = crossover(parent1.order, parent2.order);
            if ((double)rand() / RAND_MAX < mutRate) {
                childOrder = mutateInsert(childOrder); 
            }
            double childDistance = fitnessScore(childOrder, locations);
            newPopulation.push_back({childOrder, childDistance});
        }
        population = newPopulation;
        for (const Individual& individual : population) {
            // cout << individual.distance<<" "<< generation<<endl;
            if (individual.distance < best.distance) {
                best = individual;
                cout << "Generation: " << generation << "\n"; 
                cout << "Best distance so far: " << best.distance << "\n"; 
                outFile << generation << " " << best.distance << "\n"; // Write the generation number and best distance
            }
        }
      
    }
    outFile.close();
}

int main() {
    unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    std::srand(seed);
    vector<Location> locations = parse("input.txt");
    cout<< locations.size()<<endl;
    vector<int> defaultOrder(locations.size());
    for (int i = 0; i < locations.size(); ++i) {
        defaultOrder[i] = i;
    }
    Individual best;
    int totalCalcs = 250000;
    int generationsTotal = 1000; 
    int populationSize = totalCalcs / generationsTotal; 
    vector<Individual> population = initPopulation(defaultOrder, populationSize, locations, best);
    cout << "Initial best distance: " << best.distance << "\n"; 
    runGeneticAlgorithm(population, locations, best, generationsTotal, populationSize / 10); 
    cout << "Final best distance: " << best.distance << "\n"; 
    cout << "Best order: ";
    auto it = find(best.order.begin(), best.order.end(), 1);
    if (it != best.order.end()) {
        int index = distance(best.order.begin(), it);
        for (int i = index; i < best.order.size(); ++i) {
            cout << best.order[i] << " ";
        }
        for (int i = 0; i < index; ++i) {
            cout << best.order[i] << " ";
        }
    } else {
        for (int i : best.order) {
            cout << i << " ";
        }
    }
    cout << endl;
    cout << "\n";

    return 0;
}