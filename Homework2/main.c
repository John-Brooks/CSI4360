#include "main.h"
#include <stdio.h>
#include "performance_clock.h"

int main(int argc, char** argv)
{
    if (argc != 2)
		printf("Invalid number of arguments. format is ./main N\n");
    
	size_t n = strtoul(argv[1], NULL, 10);

    void* mat_a = allocate_double_matrix(n, n);
    void* mat_b = allocate_double_matrix(n, n);
    void* mat_c = allocate_double_matrix(n, n);
    
    start_clock();
    naive_matmul(n, mat_a, mat_b, mat_c);
    stop_clock();
    print_result("naive implementation", get_clock_result_seconds());

    start_clock();
    single_optimization(n, mat_a, mat_b, mat_c);
    stop_clock();
    print_result("single optimization", get_clock_result_seconds());

    return 0;
}

void single_optimization(int N, double *A, double *B, double *C)
{
    int i, j, k;
    double temp;
    for (i=0; i<N; i++)
    {
        int i_n = i*N;
        for (j=0; j < N; j++)
        {
            temp = C[i_n + j];
            for (k=0; k<N; k++)
                temp += A[i_n + k] * B[k*N + j];
            C[i_n + j] = temp;
        }
    }
}

void naive_matmul(int N, double *A, double *B, double *C)
{
    int i, j, k;
    for (i=0; i<N; i++)
        for (j=0; j < N; j++)
            for (k=0; k<N; k++)
                C[i*N + j] = C[i*N + j] + A[i*N + k] * B[k*N + j];
}

void print_result(const char* function_name, double time)
{
	printf("%s took: ", function_name);
	print_time_seconds(time);
	printf("\n");
}	

void* allocate_double_matrix(size_t m, size_t n)
{
	return malloc(sizeof(double) * m * n);
}