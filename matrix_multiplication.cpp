#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

using namespace std;

vector<long long> read_matrix(const string& filename, size_t& size)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    file >> size;

    vector<long long> matrix_value(size * size);
    for (int i = 0; i < size * size; ++i)
    {
        if (!(file >> matrix_value[i]))
        {
            std::cerr << "Error: Not enough data in file " << filename << " for " << size << "x" << size << " matrix" << endl;
            exit(1);
        }
    }
    return matrix_value;
}

void write_matrix(const string& filename, const vector<long long>& matrix_flat, size_t& size, auto& duration)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not create/open file " << filename << endl;
        exit(1);
    }

    file << size << endl;
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            file << matrix_flat[i * size + j] << (j == size - 1 ? "" : " ");
        }

        file << endl;
    }

    file << "Время выполнения задачи: " << duration << endl;
    file << "Объём задачи: " << size << " - порядок матриц";


}

int main()
{
    string file_A = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\matrix_A.txt";
    string file_B = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\matrix_B.txt";
    string result_file = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\result_C.txt";

    size_t A_matrix_size, B_matrix_size;
    vector<long long> A_matrix_values = read_matrix(file_A, A_matrix_size);
    vector<long long> B_matrix_values = read_matrix(file_B, B_matrix_size);

    if (A_matrix_size != B_matrix_size)
    {
        cerr << "Error: Matrix dimensions do not match (" << A_matrix_size << "*" << A_matrix_size << " and " << B_matrix_size << "*" << B_matrix_size << ")" << endl;
        return 1;
    }

    size_t N = A_matrix_size;
    vector<long long> result_matrix(N * N, 0LL);

    auto start_time = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            long long sum = 0LL;
            for (size_t k = 0; k < N; ++k)
            {
                sum += A_matrix_values[i * N + k] * B_matrix_values[k * N + j];
            }

            result_matrix[i * N + j] = sum;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end_time - start_time;

    cout << "Task size: " << N << endl;
    cout << "Execution time (seconds): " << duration << endl;
    cout << "Result written to: " << result_file << endl;

    write_matrix(result_file, result_matrix, N, duration);

    return 0;
}