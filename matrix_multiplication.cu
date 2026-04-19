%% writefile multiplication.cu
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

vector<long long> read_matrix(const string& filename, size_t& size)
{
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Ошибка: Не удалось открыть файл " << filename << endl;
        exit(1);
    }

    file >> size;
    vector<long long> matrix_value(size * size);
    for (int i = 0; i < size * size; ++i)
    {
        if (!(file >> matrix_value[i]))
        {
            cerr << "Ошибка данных в " << filename << endl;
            exit(1);
        }
    }

    return matrix_value;
}

void write_matrix(const string& filename, const vector<long long>& matrix_flat, size_t& size, double duration_ms)
{
    ofstream file(filename);

    file << size << endl;
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            file << matrix_flat[i * size + j] << (j == size - 1 ? "" : " ");
        }

        file << endl;
    }

    file << "Время выполнения: " << duration_ms << " ms" << endl;
}

__global__ void matrixMultiply(const long long* A, const long long* B, long long* C, size_t N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        long long sum = 0LL;
        for (size_t k = 0; k < N; ++k)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main()
{
    int block_size = 32;

    string file_A = "/content/matrix_A_800.txt";
    string file_B = "/content/matrix_B_800.txt";
    string result_file = "/content/result_C_" + to_string(block_size) + ".txt";

    size_t A_size, B_size;
    vector<long long> h_A = read_matrix(file_A, A_size);
    vector<long long> h_B = read_matrix(file_B, B_size);

    size_t N = A_size;
    size_t bytes = N * N * sizeof(long long);
    vector<long long> h_C(N * N);

    long long* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(block_size, block_size);
    dim3 blocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    auto start = chrono::high_resolution_clock::now();
    matrixMultiply << <blocks, threads >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> duration = end - start;
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cout << "N = " << N << ", Блок = " << block_size << "x" << block_size << endl;
    cout << "Время выполнения : " << duration.count() << " мс" << endl;

    write_matrix(result_file, h_C, N, duration.count());

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}