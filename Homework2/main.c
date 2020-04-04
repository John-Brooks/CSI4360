#include "main.h"
#include <stdio.h>
#include "performance_clock.h"
#include <immintrin.h>

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

void simd_multi(int N, double *A, double *B, double *C)
{
    int i, j, k, block;
    double temp;
    __m256d SIMD_A, SIMD_B, SIMD_LSUM, SIMD_RSUM;
 
    double* local_A = malloc(sizeof(double)* 4);
    double* local_B = malloc(sizeof(double)* 4);
    double* local_SUM = malloc(sizeof(double)* 4);
    double* running_SUM = malloc(sizeof(double)* 4);
    SIMD_RSUM = _mm256_loadu_pd(running_SUM);
    for (i=0; i<N; i++)
    {
        for (j=0; j < N; j++)
        {
            for(block = 0; block < N/4; block++)
            {
                for (k=0; k<N; k+=4)
                {
                    local_A = &A[i*N + k];
                    local_B[0] = B[k*N + j];
                    local_B[1] = B[(k+1)*N + j];
                    local_B[2] = B[(k+2)*N + j];
                    local_B[3] = B[(k+3)*N + j];
                    SIMD_A = _mm256_loadu_pd(local_A);
                    SIMD_B = _mm256_loadu_pd(local_B);
                    SIMD_LSUM = _mm256_mul_pd(SIMD_A, SIMD_B);
                    SIMD_RSUM = _mm256_add_pd(SIMD_LSUM, SIMD_RSUM);
                }
                double sum[4];
                _mm256_storeu_pd(sum, SIMD_RSUM);
                C[i*N + j] = sum[0] + sum[1] + sum[2] + sum[3];
                //printf("made a sum\n");
            }    
        }
    }
    /*printf("a\n");
    free(local_A);
    printf("b\n");
    free(local_B);
    printf("c\n");
    free(local_SUM);
    printf("d\n");
    free(running_SUM);*/
}

void single_optimization(int N, double *A, double *B, double *C)
{
    int i, j, k;
    double temp;
    for (i=0; i<N; i++)
    {
        for (j=0; j < N; j++)
        {
            temp = C[i*N + j];
            for (k=0; k<N; k++)
                temp += A[i*N + k] * B[k*N + j];
            C[i*N + j] = temp;
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