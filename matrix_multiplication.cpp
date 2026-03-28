#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <omp.h>

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

void write_matrix(const string& filename, const vector<long long>& matrix_flat, size_t& size)
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
}


void write(const string& filename, const vector<double>& times, const vector<int>& threads)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not create/open file " << filename << endl;
        exit(1);
    }
    file << "Количество потоков:" << " " << "Время выполнения" << "\n";

    for (int i = 0; i < times.size(); ++i)
    {
        file << threads[i] << " " << times[i] << "\n";
    }

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
        cerr << "Error: Matrix dimensions do not match" << endl;
        return 1;
    }

    size_t N = A_matrix_size;
    string statistics = "C:\\Users\\Адель\\Desktop\\lab-1\\Data\\statistics_data_" + to_string(N) + ".txt";
    vector<long long> result_matrix(N * N, 0LL);

    int num_threads = 1; 
    vector<double> times = {};
    vector<int> threads = {};
  
    while (num_threads<=16)
    {
        omp_set_num_threads(num_threads);

        cout << "Running with " << num_threads << " threads..." << endl;

        auto start_time = chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int i = 0; i < (int)N; ++i)
        {
            for (size_t k = 0; k < N; ++k)
            {
                long long temp = A_matrix_values[i * N + k];

                for (size_t j = 0; j < N; ++j)
                {
                    result_matrix[i * N + j] += temp * B_matrix_values[k * N + j];
                }
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end_time - start_time;

        cout << "Task size: " << N << "x" << N << endl;
        cout << "Execution time (ms): " << duration.count() << endl;
        cout << "\n";

        times.push_back(duration.count());
        threads.push_back(num_threads);

        if (num_threads!=16)
            fill(result_matrix.begin(), result_matrix.end(), 0LL);

        num_threads++;
    }

    write(statistics, times, threads);
    write_matrix(result_file, result_matrix, N);

    return 0;
}