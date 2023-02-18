#include <stdio.h>
#include <math.h>

const int N = 10000000;

double func(double* darr)
{
	double sum = 0;
	double pi = acos(-1);
	#pragma acc kernels //parallel loop
	for (int i = 0; i < N; i += 1)
	{
		darr[i] = sin(2 * i * pi / N);
		sum += darr[i];
	}
	return sum;
}

double* darr[10000000];

int main()
{
	double sum = func(darr);

	printf("%d\n", sum);

	return 0;
}
