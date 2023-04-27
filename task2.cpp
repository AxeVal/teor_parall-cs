// подключение библиотек С++
#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>

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
            // вывод (да/нет)
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
	
    memset(A, 0, size * size * sizeof(double));

    // инициализация A
    for(int i = 0; i < size; i += 1) 
    {
        at(A, 0, i)      = 10.0 / (size - 1) * i + 10;
        at(A, i, 0)      = 10.0 / (size - 1) * i + 10;
        at(A, size-1, i) = 10.0 / (size - 1) * i + 20;
        at(A, i, size-1) = 10.0 / (size - 1) * i + 20;
     }
   
    double error  = 1.0;
    int iteration = 0;

    // копирование массивов на видеокарту
    #pragma acc data copyin(A[:size * size]) create(Anew[:size * size])
    {
        #pragma acc parallel loop 
        // заполнение граней массива Anew начальными значениями
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
            for( int i = 1; i < size-1; i += 1) 
            {
                #pragma acc loop
                    for(int j = 1; j < size - 1; j += 1) 
                        at(Anew, i, j) = 0.25 * ( at(A, i, j+1) + at(A, i, j-1) + at(A, i-1, j) + at(A, i+1, j) );
            }

            // вычисление матрицы A        
            #pragma acc parallel loop
            for(int i = 1; i < size-1; i += 1)
            {
            #pragma acc loop
                for(int j = 1; j < size - 1; j += 1) 
                    at(A, i, j) = 0.25 * ( at(Anew, i, j+1) + at(Anew, i, j-1) + at(Anew, i-1, j) + at(Anew, i+1, j) );
            }

            iteration += 2;

            if (iteration % 100 == 0)
            {
                error = 0.0;
                // возвращаем ошибку на host
                #pragma acc parallel loop reduction(max:error)
                for(int i = 1; i < size - 1; i += 1)
                {
                    #pragma acc loop reduction(max:error)
                        for(int j = 1; j < size - 1; j += 1)
                            error = fmax(error, fabs(at(A, i, j) - at(Anew, i, j)));
                }
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
