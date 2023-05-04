// подключение библиотек С++
#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>

// библиотеки cuda
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define at(arr, x, y) (arr[(x) * size + (y)]) 

#define max_thread 32

// начальные переменные
// минимальное значение максимальной ошибки
double eps = 1E-6;
// максимальное количество итераций
int iter_max = 1E6;
// размер матрицы
int size = 128;
// вывод матрицы (да/нет)
bool mat = false;
// количество итераций до обновления ошибки
int iter_update = 100;

// функции для работы на видеокарте
// вычисление одного шага алгоритма для матрицы B
__global__ void calc(double *A, double *B, int size)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // чтобы не изменять границы
    if ( (j == 0) || (i == 0) || (i == (size-1)) || (j == (size-1)) ) 
		return;

    at(B, i, j) = 0.25 * (at(A, i-1, j) + at(A, i, j-1) + at(A, i+1, j) + at(A, i, j+1));
}

// вычитание элемента матрицы B из элемента матрицы A 
// результат записывается в матрицу B
__global__ void sub(double *A, double *B, int size)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	at(B, i, j) = at(A, i, j) - at(B, i, j);
}

// шаг заполнения матрицы A начальными значениями
__global__ void init(double *A, int size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = 10.0 / (size-1) * i;

    at(A, 0, i)      = 10.0 + step;
    at(A, i, 0)      = 10.0 + step;
    at(A, i, size-1) = 20.0 + step;
    at(A, size-1, i) = 20.0 + step;
}

int main(int argc, char **argv)
{
    cudaSetDevice(3);
    // ввод данных из консоли
    for(int arg = 0; arg < argc; arg += 1)
    { 
        if(strcmp(argv[arg], "-error") == 0)
        {   
            // ошибка
            eps = atof(argv[arg+1]);
            arg += 1;
        }
        else if(strcmp(argv[arg], "-iter") == 0)
        {
            // итерации
            iter_max = atoi(argv[arg+1]);
            arg += 1;
        }
        else if(strcmp(argv[arg], "-size") == 0)
        {
            // размер
            size = atoi(argv[arg+1]);
            arg += 1;
        }
        else if(strcmp(argv[arg], "-iter_update") == 0)
        {
            // итерации до вычисления ошибки
            iter_update = atoi(argv[arg+1]);
            arg += 1;
        }
        else if(strcmp(argv[arg], "-mat") == 0)
        {
            // вывод
            mat = true;
            arg += 1;
        }
    }

    // количество элементов матрицы
    int total_size = size * size;

    // вывод параметров
    std::cout << "Settings:                 "                << std::endl;
    std::cout << "\tMin error:              " << eps         << std::endl;
    std::cout << "\tMax iteration:          " << iter_max    << std::endl;
    std::cout << "\tSize:                   " << size        << std::endl;
    std::cout << "\tIterations to update:   " << iter_update << std::endl;
    std::cout << "\tPrint the matrix (1/0): " << mat         << std::endl;
    std::cout << std::endl;
    
    cudaError_t crush;

    // создание потока
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // создание графа
    cudaGraph_t     graph;
    cudaGraphExec_t graph_instance;

    // количество блоков в гриде и нитей в блоке соответственно
    dim3 blocks_in_grid   = dim3(size / max_thread, size / max_thread);
    dim3 threads_in_block = dim3(max_thread, max_thread);

    // вывод значений блоков и нитей
    std::cout << "Settings for cuda:      "                        << std::endl;
    std::cout << "\tMax blocks in grid:   " << size / max_thread   << std::endl;
    std::cout << "\tMax threads in block: " << max_thread          << std::endl;
    std::cout << std::endl;

    // создание матриц для работы на видеокарте
    double *A_device, *A_new_device, *cudaError, *temp_storage = NULL;
    size_t temp_size = 0;

    // выделение памяти для матриц на видеокарте
    cudaMalloc(&A_device,     sizeof(double) * total_size);
    cudaMalloc(&A_new_device, sizeof(double) * total_size);
    cudaMalloc(&cudaError,    sizeof(double) * 1);

    dim3 thread = size < 1024 ? size : 1024;
    dim3 block = size / (size < 1024 ? size : 1024);

    // заполнение матриц начальными значениями (значения границ)
    init<<<block, thread>>>(A_device, size);
    init<<<block, thread>>>(A_new_device, size);
    // cudaMemcpy(A_new_device, A_device, sizeof(double) * total_size, cudaMemcpyDeviceToDevice);

    // вычисление tmp_size (размера temp_storage)
    cub::DeviceReduce::Max(temp_storage, temp_size, A_new_device, cudaError, total_size, stream);
    cudaMalloc(&temp_storage, temp_size);


    // начало графа  ================================================================================
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // выполнение шагов алгоритма столько раз, сколько необходимо до обновления ошибки
    for(int i = 0; i < iter_update; i += 2)
    {
        calc<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_device, A_new_device, size);
        calc<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_new_device, A_device, size);
    }

    // вычитание A_new_device из A_device
    sub<<<blocks_in_grid, threads_in_block, 0, stream>>>(A_device, A_new_device, size);

    // вычисление ошибки
    cub::DeviceReduce::Max(temp_storage, temp_size, A_new_device, cudaError, total_size, stream);

    // заполнения границ матрицы A_new_device начальными значениями
    init<<<block, thread, 0, stream>>>(A_new_device, size);
    
    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0);
        
    // конец графа =================================================================================

    int iter = 0;
    double error = 1.0;

    // алгоритм поиска ошибки с помощью графа
    while( (iter < iter_max) && (error > eps) )
    {
        cudaGraphLaunch(graph_instance, stream);
        cudaMemcpyAsync(&error, cudaError, sizeof(double), cudaMemcpyDeviceToHost, stream);
        //if(error != 0)
        //std::cout << error << std::endl;
        iter += iter_update;
    }
    
    // std::cout << crush << std::endl;

    // вывод результатов
    std::cout << "Result:       "          << std::endl;
    std::cout << "\tIterations: " << iter  << std::endl;
    std::cout << "\tError:      " << error << std::endl;

    // вывод матрицы
    if(mat)
    {
        double* A = new double[total_size];
        cudaMemcpyAsync(&A, A_device, sizeof(double), cudaMemcpyDeviceToHost, stream);
        for(int i = 0; i < size; i += 1)
        {
            for(int j = 0; j < size; j += 1)
                std::cout << at(A_device, i, j) << ' ';
            std::cout << std::endl;
        }
    }

    // удаление потока и графа
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);

    // очищение памяти от матриц на видеокарте
    cudaFree(A_device);
    cudaFree(A_new_device);
    cudaFree(temp_storage);

    return 0;
}

// /usr/local/cuda/bin/nvcc -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.0/lib64 -lcudnn task4.cu -o task4
