#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 4096 //number of floats processed per SM
#define BLOCK_DIM 64//sqrt of BLOCK_SIZE
#define MAX_THREADS 1024 //Limit of GTX 1080

using namespace std;

timespec start_time;
timespec stop_time;
void start_clock();
void stop_clock();
double get_clock_result_seconds();
void print_time_seconds(double seconds);

void PrintPartialMatrix(size_t n, float* matrix)
{
    if(n < 5)
    {
        printf("Matrix is too small to print.\n");
        return;
    }

    for(size_t i = 0; i < 5; ++i)
    {
        for(size_t j = 0; j < 5; j++)
        {
            printf("%0.2f\t", matrix[(i*n) + j]);
        }
        printf("\n");
    }
}

__global__ void transpose_one_to_one(size_t n, float* input, float* output)
{
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if(i >= n*n)
        return;
    output[((i % n)*n) + (i / n)] = input[i];
}
__global__ void transpose_optimized(size_t n, int elm_per_thread, float* input, float* output)
{
    __shared__ float s_data[BLOCK_SIZE];

    unsigned int block_level_index = threadIdx.x*elm_per_thread;
    unsigned int start = (blockIdx.x * blockDim.x) + block_level_index;
    unsigned int stop = start + elm_per_thread;

    int s_idx = block_level_index;
    for(int i = start; i < stop; i++, s_idx++)
    {   
        if(i >= n*n)
            break;
        s_data[s_idx] = input[i];
    }
    __syncthreads();

    s_idx = block_level_index;
    for(int i = start; i < stop; i++, s_idx++)
    {
        if(i >= n*n)
            break;
        output[((i % n)*n) + (i / n)] = s_data[s_idx];
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Not enough arguments\n");
        printf("Usage is ./a.out [matrix dim]\n");
        return 1;
    }

    string dimension_arg = argv[1];
    size_t N = 0;
    try{
        N = stoul(dimension_arg, nullptr, 10);
    }
    catch(...){
        printf("Matrix dimension argument %s is not valid\n", dimension_arg.c_str());
        return 2;
    }

    size_t matrix_size = N*N*sizeof(float);
    
    dim3 block_structure(BLOCK_DIM, BLOCK_DIM);

    float* d_input_matrix;
    float* d_resultant_matrix;
    cudaMalloc((void **)&d_input_matrix, matrix_size);
    cudaMalloc((void **)&d_resultant_matrix, matrix_size);


    float* h_input_matrix = new float[N*N];
    float* h_resultant_matrix_1 = new float[N*N];
    float* h_resultant_matrix_2 = new float[N*N];
    srand(time(nullptr));
    for( int i = 0; i < N*N; i++)
        h_input_matrix[i] = (float)rand() / (float)RAND_MAX;

    cudaMemcpy(d_input_matrix, h_input_matrix, matrix_size, cudaMemcpyHostToDevice);

    int num_blocks = ((N*N)/MAX_THREADS);
    num_blocks += (N*N) % MAX_THREADS > 0 ? 1 : 0;
    int num_threads = MAX_THREADS < N*N ? MAX_THREADS : N*N;
    printf("num_blocks: %i\n", num_blocks);
    printf("num_threads: %i\n", num_threads);
    
    start_clock();
    transpose_one_to_one<<<num_blocks, num_threads>>>(N, d_input_matrix, d_resultant_matrix);
    cudaDeviceSynchronize();
    stop_clock();
    printf("naive time:\t");
    print_time_seconds(get_clock_result_seconds());
    printf("\n\n");

    cudaMemcpy(h_resultant_matrix_1, d_resultant_matrix, matrix_size, cudaMemcpyDeviceToHost);

    size_t real_block_size = N*N < BLOCK_SIZE ? N*N : BLOCK_SIZE;
    num_blocks = ((N*N)/real_block_size);
    num_blocks += (N*N) % real_block_size > 0 ? 1 : 0;
    num_threads = MAX_THREADS < N*N ? MAX_THREADS : N*N;
    int elm_per_thread = real_block_size / num_threads;
    printf("num_blocks: %i\n", num_blocks);
    printf("num_threads: %i\n", num_threads);
    printf("elm_per_thread: %i\n", elm_per_thread);
    start_clock();
    transpose_optimized<<<num_blocks, num_threads>>>(N, elm_per_thread, d_input_matrix, d_resultant_matrix);
    cudaDeviceSynchronize();
    stop_clock();
    printf("optimized time:\t");
    print_time_seconds(get_clock_result_seconds());
    printf("\n");

    cudaMemcpy(h_resultant_matrix_2, d_resultant_matrix, matrix_size, cudaMemcpyDeviceToHost);


    if (memcmp(h_resultant_matrix_1, h_resultant_matrix_2, matrix_size) != 0)
    {
        printf("Results DO NOT match!\n");\
        for(size_t i = 0; i < matrix_size/sizeof(float); i++)
        {
            if(h_resultant_matrix_1[i] != h_resultant_matrix_2[i])
            {
                printf("index %lu doesn't match\n", i);
                printf("Input:\n");
                PrintPartialMatrix(N, h_input_matrix);
                printf("\nOutput 1:\n");
                PrintPartialMatrix(N, h_resultant_matrix_1);
                printf("\nOutput 2:\n");
                PrintPartialMatrix(N, h_resultant_matrix_2);
                break;
            }
        }
    }
    else 
    {
        printf("Results match\n");
        printf("Input:\n");
        PrintPartialMatrix(N, h_input_matrix);
        printf("\nOutput 1:\n");
        PrintPartialMatrix(N, h_resultant_matrix_1);
        printf("\nOutput 2:\n");
        PrintPartialMatrix(N, h_resultant_matrix_2);
    }

    cudaFree(d_input_matrix);
    cudaFree(d_resultant_matrix);
    delete[] h_input_matrix;
    delete[] h_resultant_matrix_1;
    delete[] h_resultant_matrix_2;


    return 0;
}

void start_clock()
{
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start_time);
}
void stop_clock()
{
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stop_time);
}
double get_clock_result_seconds()
{
	double result = stop_time.tv_sec - start_time.tv_sec;
	result += (double)(stop_time.tv_nsec - start_time.tv_nsec) / 1000000000;
	return result;
}
void print_time_seconds(double seconds)
{
#ifdef _WIN32
	printf("%0.3f seconds", seconds);
#elif __linux__
	printf("%0.9f seconds", seconds);
#endif
}