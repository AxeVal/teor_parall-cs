#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 10000000
#define pi 3.14159265358979323846

#ifdef Double
typedef double type;
#else
typedef float type;
#endif

type darr[N];
double sum = 0;

int main()
{
	double time_1 = 0.0, time_2 = 0.0, time_all = 0.0;
	#pragma acc data create(darr[:N]) copy(sum)
	{
		clock_t begin_1 = clock();
		#pragma acc kernels
		for (int i = 0; i < N; i += 1)
		{
			#ifdef Double
				darr[i] = sin(2 * pi * i / N);
			#else
				darr[i] = sinf(2 * pi * i / N);
			#endif 
		}
		clock_t end_1 = clock();

		clock_t begin_2 = clock();
		for (int i = 0; i < N; i += 1)
		{
			sum += darr[i];
		}
		clock_t end_2 = clock();

		time_1 += (double)(end_1 - begin_1);
		time_2 += (double)(end_2 - begin_2);
	}
	printf("sum is %.25f\n", sum);
	printf("first cycle: %f\n", time_1);
	printf("second cycle: %f\n", time_2);

	time_all = time_1 + time_2;

	printf("all prog: %f\n", time_all);

	return 0;
}