#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
using namespace std;
//Tansylu Akhmetova 25/02/2024
// A number will appear once in each column.
// A number will appear once in each row.
// A number will appear once in each 3x3 small grid.

vector<vector<vector<int>>> parseSudokus(string filename) {
    ifstream file(filename);
    string line;
    vector<vector<vector<int>>> sudokus;
    while (getline(file, line)) { 
        vector<vector<int>> sudoku;
        for (int i = 0; i < 9; ++i) { 
            if (getline(file, line)) {
                vector<int> row;
                for (char c : line) {
                    if (c >= '0' && c <= '9') {
                        row.push_back(c - '0'); 
                    }
                }
                sudoku.push_back(row);
            }
        }
        sudokus.push_back(sudoku);
        if (sudokus.size() == 10) break; 
    }
    file.close();
    return sudokus;
}

bool isPossible(vector<vector<int>>& sudoku, int row, int col, int num) {
    for(int i = 0; i < 9; i++) {
        if(sudoku[row][i] == num || sudoku[i][col] == num) {
            return false;
        }
    }

    int startRow = row - row % 3;
    int startCol = col - col % 3;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            if(sudoku[i + startRow][j + startCol] == num) {
                return false;
            }
        }
    }

    return true;
}

bool solveSudoku(vector<vector<int>>& sudoku) {
    auto start = chrono::high_resolution_clock::now();
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            if (sudoku[row][col] == 0) { // empty spot found
                for (int num = 1; num <= 9; num++) {//iterate 1-9
                    if (isPossible(sudoku, row, col, num)) {// try to see if the number fits
                        sudoku[row][col] = num;// set the cell
                        if (solveSudoku(sudoku)) { // recurse to solve rest of sudoku with this update
                            auto stop = chrono::high_resolution_clock::now(); 
                            auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
                            if(duration.count()>0)cout << "Took more than 1 second: " << duration.count() << "seconds" << endl;
                            return true;
                        }
                        sudoku[row][col] = 0; 
                    }
                }
                return false; //tried every number and none worked
            }
        }
    }//terminates when no more empty spots found
    return true; 
}

bool validateSudoku(vector<vector<int>>& sudoku) {
 
    for(int row = 0; row < 9; row++) {
        vector<bool> seen(10, false);
        for(int col = 0; col < 9; col++) {
            if(seen[sudoku[row][col]]) {
                return false;
            }
            seen[sudoku[row][col]] = true;
        }
    }

    for(int col = 0; col < 9; col++) {
        vector<bool> seen(10, false);
        for(int row = 0; row < 9; row++) {
            if(seen[sudoku[row][col]]) {
                return false;
            }
            seen[sudoku[row][col]] = true;
        }
    }

    for(int startRow = 0; startRow < 9; startRow += 3) {
        for(int startCol = 0; startCol < 9; startCol += 3) {
            vector<bool> seen(10, false);
            for(int row = 0; row < 3; row++) {
                for(int col = 0; col < 3; col++) {
                    if(seen[sudoku[row + startRow][col + startCol]]) {
                        return false;
                    }
                    seen[sudoku[row + startRow][col + startCol]] = true;
                }
            }
        }
    }

    return true;
}

int main() {
    vector<vector<vector<int>>> sudokus = parseSudokus("input.txt");
    int sudokuCount = 1;
    for (vector<vector<int>> sudoku : sudokus) {
        cout << "SUDOKU " << sudokuCount++ << endl;
        bool flag = solveSudoku(sudoku);
        if(!flag){
            cout << "couldnt solve this sudoku" << endl;
            continue;
        }
        else{
            flag = validateSudoku(sudoku);
            cout << flag<<endl;
            for (vector<int> row : sudoku) {
                for (int cell : row) {
                    cout << cell << " ";
                }
                cout << endl;
            }
        }
    }
    return 0;
}
