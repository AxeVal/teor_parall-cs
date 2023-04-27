// подключение библиотек С++
#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>

// библиотека cublas
#include <cuda_runtime.h>
#include <cublas_v2.h>

// для работы с одномерными массивами как с двумерными
#define at(arr, x, y) (arr[(x) * size + (y)]) 

// инициализаци значений переменных по умолчанию
// максимальное количество итераций
int iter_max = 1E6;
// минимальное значение ошибки
double eps   = 1E-6;
// размер стороны квадратной матрицы
int size     = 128;
bool mat     = false;

int main(int argc, char **argv)
{
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
        else if(strcmp(argv[arg], "-mat") == 0)
        {
            // вывод матрицы (да/нет)
            mat = true;
            arg += 1;
        }
    }

    // вывод значений переменных
    std::cout << "EPS: "           << eps      << std::endl;
    std::cout << "Max iteration: " << iter_max << std::endl;
    std::cout << "Size: "          << size     << std::endl;
    std::cout << std::endl;
    
    // выделение памяти для массивов
    double* A    = new double[size * size];
	double* Anew = new double[size * size];
	
    // инициализация граничных значений
    for(int i = 0; i < size; i += 1) 
    {
        at(A, 0, i)        = 10.0 / (size - 1) * i + 10;
        at(A, i, 0)        = 10.0 / (size - 1) * i + 10;
        at(A, size - 1, i) = 10.0 / (size - 1) * i + 20;
        at(A, i, size - 1) = 10.0 / (size - 1) * i + 20;
     }

    // начальные значения переменных
    double error  = 1.0;
    int iteration = 0;

    // создание хэндлера
    cublasHandle_t handler;
	cublasCreate(&handler);

// копирование массивов на видеокарту
    #pragma acc data copyin(A[:size * size]) create(Anew[:size * size])
    {
        #pragma acc parallel loop 
        // заполнение граней массива начальными значениями
        for(int j = 0; j < size; j += 1) 
        {
            at(Anew, j, 0)      = at(A, j, 0);
            at(Anew, 0, j)      = at(A, 0, j);
            at(Anew, j, size-1) = at(A, j, size-1);
            at(Anew, size-1, j) = at(A, size-1, j);
        }

        while ( (error > eps) && (iteration < iter_max) )
        {
            // вычисление матрицы Anew
            #pragma acc parallel loop
            for(int i = 1; i < size - 1; i += 1) 
            {
                #pragma acc loop
                for(int j = 1; j < size - 1; j += 1) 
                    at(Anew, i, j) = 0.25 * ( at(A, i, j+1) + at(A, i, j-1) + at(A, i-1, j) + at(A, i+1, j) );
            }

            // вычисление матрицы A (шаг после Anew)
            #pragma acc parallel loop
            for(int i = 1; i < size-1; i += 1)
            {
                #pragma acc loop
                for(int j = 1; j < size - 1; j += 1) 
                    at(A, i, j) = 0.25 * ( at(Anew, i, j+1) + at(Anew, i, j-1) + at(Anew, i-1, j) + at(Anew, i+1, j) );
            }

            // прибваляем 2, потомучто вычислены обе матрицы (2 итерации)
            iteration += 2;

            // каждые 100 итераций вычисляется ошибка
            if (iteration % 100 == 0)
            {
                int idx = 0;
                double alpha = -1.0;
                
                #pragma acc host_data use_device(Anew, A)
                {
                    // A[] += alpha*Anew[]  или  A[] = A[]-Anew[]
                    cublasDaxpy(handler, size * size, 
                                &alpha, Anew, 1, A, 1);
                    // нахождение индекса элемента с максимальной ошибкой
                    cublasIdamax(handler, size * size, 
                                A, 1, &idx);
                }

                // возвращаем ошибку на host
                #pragma acc update host(A[idx-1]) 
                // индексация функции cublasIdamax начинается с 1, поэтому idx-1
                error = abs(A[idx-1]);

                #pragma acc host_data use_device(Anew, A)
                cublasDcopy(handler, size * size, Anew, 1, A, 1);
            }
        }

        if(mat)
        {
            #pragma acc update host(A[:size * size]) 
            for(int i = 0; i < size; i += 1)
            {
                for(int j = 0; j < size; j += 1) 
                    std::cout << at(A, i, j) << ' ';
                std::cout << std::endl;
            }
        }
    }

    // удаление хэндлера
    cublasDestroy(handler);
    
    // вывод результатов
    std::cout << "Result: "                  << std::endl;
    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Error: "      << error     << std::endl;
    std::cout << std::endl;

    // удаление памяти массивов
    delete[] A;
    delete[] Anew;

    // возвращаем результат работы main(), default return 0
    return 0;
}
