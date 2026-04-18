# parallel-programming
# Отчёт по лабораторной работе: Перемножение матриц на C++ с использованием технологии MPI

## 1. Задание
 Модифицировать программу из л/р №1 для параллельной работы по технологии MPI.
 Провести серию экспериментов с разными размерами матриц (примерно 200, 400, 800, 1200, 1600, 2000),
 с разным количеством вычислительных ядер (1, 2, 4, 8 и т.д.).

## 2. Описание работы скриптов

### 2.1. `generate_matrices.py`
- Использует `numpy.random.randint` для генерации целочисленных значений.
- Порядок матрицы `n_matrix` задан внутри скрипта (например, `n_matrix = 1000`).
- Записывает `n_matrix` в первую строку текстового файла, затем сгенерированный элементы матрицы построчно.

### 2.2. `verification_matrix.py`
- Загружает матрицы A, B и результат C++ из файлов по фиксированным путям.
- Считывает порядок квадратных матриц `n_matrix` из первой строки каждого файла.
- Вычисляет эталонное произведение через `np.dot` и выполняет точное поэлементное сравнение с результатом C++ с помощью `np.array_equal`.
- При успешном совпадении сохраняет эталонную матрицу в файл `verification_result_C.txt` для возможности визуального анализа и выводит подтверждение в консоль.
- В случае ошибки выводит уведомление и завершает работу.

### 2.3. `matrix_multiplication.cpp`
- Матрицы считываются из текстовых файлов и хранятся в `vector<long long>` как одномерные массивы для повышения локальности данных.
- Доступ к элементу `(i, j)` осуществляется по формуле `i * N + j,` где `N` — порядок квадратной матрицы.
- Параллелизация вычислений реализована средствами `MPI`: главный процесс (rank 0) распределяет строки матрицы `A` между процессами через `MPI_Scatterv`, рассылает матрицу `B` целиком через `MPI_Bcast` и собирает итоговые части результата через `MPI_Gatherv`.
- Вычисление элементов результирующей матрицы производится по классическому алгоритму с порядком обхода циклов `i -> j -> k`.
- Измерение времени выполнения программы осуществляется с помощью `chrono::high_resolution_clock`.
- Результаты замеров (время в мс) и итоговая рассчитанная матрица записываются в `result_C.txt`.

### 2.4. Код `matrix_multiplication.cpp`

```C++
#include <mpi.h>
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
        std::cerr << "Ошибка: Не удалось открыть файл " << filename << endl;
        exit(1);
    }

    file >> size;

    vector<long long> matrix_value(size * size);
    for (int i = 0; i < size * size; ++i)
    {
        if (!(file >> matrix_value[i]))
        {
            std::cerr << "Ошибка: Недостаточно данных в файле " << filename << " для " << size << "x" << size << " матриц" << endl;
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
        cerr << "Ошибка открытия файла:  " << filename << endl;
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

    file << "Время выполнения задачи: " << duration.count() << " ms" << endl;
    file << "Объём задачи: " << size << " - порядок матриц";
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

    string file_A = "C:\\Users\\Адель\\Desktop\\lab-1\\lab_3\\parallel-programming\\Data_for_size=2000\\matrix_A.txt";
    string file_B = "C:\\Users\\Адель\\Desktop\\lab-1\\lab_3\\parallel-programming\\Data_for_size=2000\\matrix_B.txt";
    string result_file = "C:\\Users\\Адель\\Desktop\\lab-1\\lab_3\\result_C.txt";

    size_t N = 0;
    vector<long long> A_matrix_values;
    vector<long long> B_matrix_values;
    vector<long long> result_matrix;

    if (rank == 0)
    {
        size_t A_matrix_size, B_matrix_size;
        A_matrix_values = read_matrix(file_A, A_matrix_size);
        B_matrix_values = read_matrix(file_B, B_matrix_size);

        if (A_matrix_size != B_matrix_size)
        {
            cerr << "Ошибка: Размеры матрицы не совпадают" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        N = A_matrix_size;
        result_matrix.resize(N * N, 0LL);
    }

    unsigned long long N_ull = N;
    MPI_Bcast(&N_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N_ull);

    if (rank != 0)
    {
        B_matrix_values.resize(N * N);
    }

    MPI_Bcast(B_matrix_values.data(), N * N, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    int rows_per_proc = N / size_mpi;
    int remainder = N % size_mpi;

    vector<int> sendcounts(size_mpi);
    vector<int> displs(size_mpi);

    int offset = 0;
    for (int i = 0; i < size_mpi; ++i)
    {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_elements = sendcounts[rank];
    int local_rows = local_elements / N;

    vector<long long> local_A(local_elements);
    vector<long long> local_C(local_elements, 0LL);

    chrono::time_point<chrono::high_resolution_clock> start_time;
    if (rank == 0)
    {
        start_time = chrono::high_resolution_clock::now();
    }

    MPI_Scatterv(rank == 0 ? A_matrix_values.data() : nullptr, sendcounts.data(), displs.data(), MPI_LONG_LONG,
        local_A.data(), local_elements, MPI_LONG_LONG,
        0, MPI_COMM_WORLD);

    for (size_t i = 0; i < local_rows; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            long long sum = 0LL;
            for (size_t k = 0; k < N; ++k)
            {
                sum += local_A[i * N + k] * B_matrix_values[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    MPI_Gatherv(local_C.data(), local_elements, MPI_LONG_LONG,
        rank == 0 ? result_matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_LONG_LONG,
        0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end_time - start_time;

        cout << "Объём задачи: " << N << endl;
        cout << "Время выполнения задачи (миллисекунды): " << duration.count() << endl;
        cout << "Результаты записаны в файл : " << result_file << endl;

        write_matrix(result_file, result_matrix, N, duration);
    }

    MPI_Finalize();
    return 0;
}
```

## 3. Результаты экспериментов

Для демонстрации работы программы были проведены эксперименты с разным количеством вычислительных ядер (1, 2, 3, 4, 5, 6, 7, 8)
для матриц порядка 200, 400, 800, 1200, 1600, 2000.

### 3.1. При порядке равном 200

![Зависимость времени от потоков для N=200](Figures/Figure_N=200.png)

- На начальном этапе (`1-4` ядра) наблюдается стабильное снижение времени до минимума в `1,92 мс`. На этом отрезке выигрыш от распараллеливания превышает затраты на организацию `MPI-процессов`.
- На `5-6` ядрах наблюдается скачок времени выполнения вверх. Это связано с дисбалансом нагрузки, так как 200 не делится нацело на 5 или 6, строки распределяются неравномерно, поэтому один из процессов работает дольше остальных, задерживая общую сборку результата.
- На `8-ми` ядрах время снова снижается (`до 1,72 мс`) благодаря идеальной кратности. Нагрузка распределяется поровну (`200 / 8 = 25` строк на процесс), что минимизирует простои и оптимизирует коллективные операции `MPI`.

### 3.2. При порядке равном 400

![Зависимость времени от потоков для N=400](Figures/Figure_N=400.png)

- На `1-7` ядрах наблюдается устойчивое снижение времени выполнения с `47,5 мс` до `9 мс`. Увеличение вычислительной сложности задачи позволило системе более эффективно использовать параллельные ресурсы.
- При переходе к `8` ядрам происходит рост времени выполнения до `12,5 мс`. Несмотря на идеальную кратность (`400 / 8 = 50` строк на процесс), на данном этапе накладные расходы на синхронизацию и сборку данных начинают превалировать над выигрышем от дробления задачи.

### 3.3. При порядке равном 800

![Зависимость времени от потоков для N=800](Figures/Figure_N=800.png)

- Время выполнения резко сокращается с `353 мс` до `116 мс`. В этом диапазоне ресурсов параллельное выполнение эффективно, так как объем вычислений значительно превышает затраты на синхронизацию потоков.
- После достижения локального минимума на `4` ядрах начинается постепенная деградация производительности. Время выполнения на `8` ядрах увеличивается до `136 мс`.  

### 3.4. При порядке равном 1200

![Зависимость времени от потоков для N=1200](Figures/Figure_N=1200.png)

- На `1-6` ядрах наблюдается наиболее эффективный участок снижения времени выполнения с `1314 мс` до `434 мс`. Рост объема данных позволяет равномерно загрузить ядра, сводя к минимуму влияние системных прерываний и задержек инициализации.
- После достижения минимума (`434 мс`) на `6` ядрах время выполнения начинает расти и на `8` ядрах составляет `445 мс`.

### 3.5. При порядке равном 1600

![Зависимость времени от потоков для N=1600](Figures/Figure_N=1600.png)

- На `1-8` ядрах время выполнения сокращается наиболее интенсивно с `10,7 с` до `2,27 с`. Большой объем обрабатываемых данных позволяет процессорам работать с максимальной загрузкой, при этом доля времени на межпроцессорное взаимодействие остается незначительной относительно полезных вычислений.


### 3.6. При порядке равном 2000

![Зависимость времени от потоков для N=2000](Figures/Figure_N=2000.png)

- Наблюдается плавное и монотонное снижение времени выполнения во всем диапазоне с `12,8 с` на одном ядре до `3 с` на восьми ядрах.
- Это свидетельствует о том, что вычислительная нагрузка стала достаточно велика, чтобы полностью нивелировать влияние системных шумов и задержек связи.

## 4. Выводы

- Параллельные вычисления демонстрируют наибольшую эффективность на матрицах высокого порядка (`N >= 1600`), где высокая вычислительная сложность задачи полностью оправдывает затраты на межпроцессорное взаимодействие.
- Для малых и средних размерностей матриц (порядка от `200` до `800`) оптимальным является использование `4–6` ядер. Дальнейшее увеличение числа процессов приводит к деградации производительности из-за дисбаланса нагрузки при некратном распределении строк.
- При больших значениях `N`, например, `N = 2000`, наблюдается наиболее стабильное и предсказуемое ускорение. В этом режиме алгоритм эффективно использует все доступные вычислительные узлы, минимизируя влияние системных шумов и задержек синхронизации.