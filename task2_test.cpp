#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <chrono>

double eps = 1E-6;
int iter_max = 1E6;
int size = 10;

#ifdef size_128
        size = 128
#endif
#ifdef size_256
        size = 256;
#endif
#ifdef size_512
        size = 512
#endif
#ifdef size_1024
       size = 1024
#endif

void initArr(double** A)
{
    A[0][0] = 10;
    A[0][size - 1] = 20;
    A[size - 1][0] = 20;
    A[size - 1][size - 1] = 30;
    
    double step = 10 / size;

    for(int i = 1; i < size - 1; i += 1) 
    {
        A[0][i]        = A[0][i - 1]        + step;
        A[i][0]        = A[i - 1][0]        + step;
        A[size - 1][i] = A[size - 1][i - 1] + step;
        A[i][size - 1] = A[i - 1][size - 1] + step;
    }
}

int main(int argc, char *argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();

    for(int arg = 0; arg < argc; arg += 1)
    {
        std::stringstream stream;
        if(strcmp(argv[arg], "-error") == 0)
        {
            stream << argv[arg+1];
            stream >> eps;
        }
        else if(strcmp(argv[arg], "-iter") == 0)
        {
            stream << argv[arg + 1];
            stream >> iter_max;
        }
        else if(strcmp(argv[arg], "-size") == 0)
        {
            stream << argv[arg+1];
            stream >> size;
        }
    }

    std::cout << "Settings:\n\tEPS: " << eps << "\n\tMax iteration: " << iter_max << "\n\tSize: " << size << 'x' << size << "\n\n";
 
    double** A = new double*[size];
    for(int i = 0; i < size; i += 1) 
        A[i] = new double[size];

    double** Anew = new double*[size];
    for(int i = 0; i < size; i += 1) 
        Anew[i] = new double[size];

    initArr(A);

    double error = 1.0;
    int iteration = 0;

#pragma acc data copy(F[:size][:size]) create(Fnew[:size][:size])
    while ( (error > eps) && (iteration < iter_max) )
    {
        error = 0.0;

#pragma acc parallel loop reduction(max:error)
        for (int j = 1; j < size - 1; j += 1) 
        {
#pragma acc loop reduction(max:error)
            for (int i = 1; i < size - 1; i += 1) 
            {
                Anew[j][i] = 0.25 * ( A[j][i + 1] + A[j][i - 1] + A[j - 1][i] + A[j + 1][i] );
                error = fmax( error, fabs(Anew[j][i] - A[j][i]) );
            }
        }
#pragma acc parallel loop
        for (int j = 1; j < size - 1; j += 1) 
        {
#pragma acc loop 
            for (int i = 1; i < size - 1; i += 1) 
            {
                A[j][i] = Anew[j][i];
            }
        }
        iteration += 1;
    }

    std::cout << "Result: " << std::endl;
    std::cout << "\tIterations: " << iteration << std::endl;
    std::cout << "\tError: " << error << std::endl;

    for (size_t i = 0; i < size; i += 1) 
        delete[] A[i];
    delete[] A;

    for (size_t i = 0; i < size; i += 1) 
        delete[] Anew[i];
    delete[] Anew;

    std::cout << "\nTotal time = " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " sec" << std::endl;

    return 0;
}
