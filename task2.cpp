#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <chrono>

double eps = 1E-6;
int iter_max = 1E6;
int size = 10;


void initArr(double** A)
{
	A[0][0] = 10;
	A[0][size - 1] = 20;
	A[size - 1][0] = 20;
	A[size - 1][size - 1] = 30;

	double step = 10 / size;

	int S = size - 1;

	for (int i = 1; i < size - 1; i += 1)
	{
		A[i][S] = A[i - 1][S] + step;
		A[i][0] = A[i - 1][0] + step;
		A[0][i] = A[0][i - 1] + step;
		A[S][i] = A[S][i - 1] + step;
	}
}


int main(int argc, char* argv[])
{
	for (int arg = 0; arg < argc; arg += 1)
	{
		std::stringstream stream;
		if (strcmp(argv[arg], "-error") == 0)
		{
			stream << argv[arg + 1];
			stream >> eps;
		}
		else if (strcmp(argv[arg], "-iter") == 0)
		{
			stream << argv[arg + 1];
			stream >> iter_max;
		}
		else if (strcmp(argv[arg], "-size") == 0)
		{
			stream << argv[arg + 1];
			stream >> size;
		}
	}

	int S = size - 1;

	std::cout << "Settings:" << std::endl;
	std::cout << "\tEPS: " << eps << std::endl;
	std::cout << "\tTotal iteration: " << iter_max << std::endl;
	std::cout << "\tSize: " << size << std::endl;

	double** A = new double* [size];
	for (int i = 0; i < size; i += 1)
		A[i] = new double[size];

	double** Anew = new double* [size];
	for (int i = 0; i < size; i += 1)
		Anew[i] = new double[size];

	initArr(A);

	double error = 1.0;
	int iteration = 0;

#pragma acc data copy(A[:size][:size]) create(Anew[:size][:size])
	while ((error > eps) && (iteration < iter_max))
	{
		error = 0.0;

#pragma acc parallel loop reduction(max:error)
		for (int i = 1; i < S; i += 1)
		{
#pragma acc loop reduction(max:error)
			for (int j = 1; j < S; j += 1)
			{
				Anew[i][j] = 0.25 * (A[i][j + 1] + A[i][j - 1] + A[i + 1][j] + A[i - 1][j]);
				error = fmax(error, fabs(Anew[i][j] - A[i][j]));
			}
		}
#pragma acc parallel loop collapse (2)
		for (int i = 1; i < S; i += 1)
		{
			for (int j = 1; j < S; j += 1)
				A[i][j] = Anew[i][j];
		}

		iteration += 1;
	}

	std::cout << "Result: " << std::endl;
	std::cout << "\tIterations: " << iteration << std::endl;
	std::cout << "\tError: " << error << std::endl;

#pragma acc loop
	for (size_t i = 0; i < size; i += 1)
	{
		delete[] A[i];
		delete[] Anew[i];
	}
	delete[] A;
	delete[] Anew;

	return 0;
}

