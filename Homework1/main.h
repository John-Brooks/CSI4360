#pragma once
#include <time.h>

typedef void(*matrix_mult_funct)(size_t, size_t, const void*, const void*, void*);

typedef struct complex {
	float real;
	float imaginary;
} complex;

struct timespec start_time;
struct timespec stop_time;
void start_clock();
void stop_clock();
double get_clock_result_seconds();
void print_time_seconds(double seconds);
void print_result(const char* function_name, double time);
void run_matrix_multi_funct(const char* name, matrix_mult_funct, size_t m, size_t n, const void* a, const void* b, void* c);
void initialize_matrix_multi_params(size_t m, size_t n, float* a, float* b, float* c);
void* allocate_float_matrix(size_t m, size_t n);
void rand_initialize_float_matrix(float* matrix, size_t size);
void complex_matvecmul_simplearray(size_t m, size_t n, const void* a, const void* b, void* c);
void complex_matvecmul_aos(size_t m, size_t n, const void* a, const void* b, void* c);
float mat_term_multiply_real(const float* a, const float* b, const float* c, const float* d);
float mat_term_multiply_imaginary(const float* a, const float* b, const float* c, const float* d);