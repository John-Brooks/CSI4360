#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "main.h"

#define PRINT_MATRIX_INIT 0

int main(int argc, char** argv)
{
	if (argc != 3)
		printf("Invalid number of arguments. format is ./main M N\n");

	size_t m = strtoul(argv[1], NULL, 10);
	size_t n = strtoul(argv[2], NULL, 10);

	void* mat_a = allocate_float_matrix(m, n);
	void* mat_b = allocate_float_matrix(n, 1);
	void* mat_c = allocate_float_matrix(m, 1);

	run_matrix_multi_funct("Simple array", (matrix_mult_funct)complex_matvecmul_simplearray, m, n, mat_a, mat_b, mat_c);
	run_matrix_multi_funct("Array of structs", (matrix_mult_funct)complex_matvecmul_aos, m, n, mat_a, mat_b, mat_c);

	free(mat_a);
	free(mat_b);
	free(mat_c);

	return 0;
}
void print_result(const char* function_name, double time)
{
	printf("%s took: ", function_name);
	print_time_seconds(get_clock_result_seconds());
	printf("\n");
}	

void run_matrix_multi_funct(const char* name, matrix_mult_funct funct, size_t m, size_t n, const void* a, const void* b, void* c)
{
	initialize_matrix_multi_params(m, n, (float*)a, (float*)b, (float*)c);

	start_clock();
	funct(m, n, a, b, c);
	stop_clock();

	print_result(name, get_clock_result_seconds());
}

inline float mat_term_multiply_real(const float* a, const float* b, const float* c, const float* d)
{
	return ((*a) * (*c) - (*b) * (*d));
}

inline float mat_term_multiply_imaginary(const float* a, const float* b, const float* c, const float* d)
{
	return ((*a) * (*d) + (*b) * (*c));
}

void complex_matvecmul_simplearray(size_t m, size_t n, const void* void_a, const void* void_b, void* void_c)
{
	const float* mat_a = (const float*)void_a;
	const float* mat_b = (const float*)void_b;
	float* mat_c = (float*)void_c;

	const float* mat_a_imaginary = mat_a + m * n;
	const float* mat_b_imaginary = mat_b + m;
	float* mat_c_imaginary = mat_c + m;

	const float* a = NULL;
	const float* b = NULL;
	const float* c = NULL;
	const float* d = NULL;
	float* product_real = NULL;
	float* product_imaginary = NULL;

	for (size_t row = 0; row < m; row++)
	{
		c = mat_b + row;
		d = mat_b_imaginary + row;
		product_real = mat_c + row;
		product_imaginary = mat_c_imaginary + row;
		for (size_t col = 0; col < n; col++)
		{
			a = mat_a + row * col;
			b = mat_a_imaginary + row * col;
			*product_real += mat_term_multiply_real(a, b, c, d);
			*product_imaginary += mat_term_multiply_imaginary(a, b, c, d);
		}
	}
}

void complex_matvecmul_aos(size_t m, size_t n, const void* void_a, const void* void_b, void* void_c)
{
	const complex* mat_a = (const complex*)void_a;
	const complex* mat_b = (const complex*)void_b;
	complex* mat_c = (complex*)void_c;

	const float* a = NULL;
	const float* b = NULL;
	const float* c = NULL;
	const float* d = NULL;
	float* product_real = NULL;
	float* product_imaginary = NULL;

	for (size_t row = 0; row < m; row++)
	{
		c = &mat_b[row].real;
		d = &mat_b[row].imaginary;
		product_real = &mat_c[row].real;
		product_imaginary = &mat_c[row].imaginary;
		for (size_t col = 0; col < n; col++)
		{
			a = &mat_a[row * col].real;
			b = &mat_a[row * col].imaginary;
			*product_real += mat_term_multiply_real(a, b, c, d);
			*product_imaginary += mat_term_multiply_imaginary(a, b, c, d);
		}
	}
}

void initialize_matrix_multi_params(size_t m, size_t n, float* a, float* b, float* c)
{
	rand_initialize_float_matrix(a, m*n);
	rand_initialize_float_matrix(b, n);
	memset(c, 0, sizeof(float) * m);
}

void rand_initialize_float_matrix(float* matrix, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		matrix[i] = (float)rand() / (float)RAND_MAX;
#if PRINT_MATRIX_INIT
		printf("%f\n", matrix[i]);
#endif
	}
}

void* allocate_float_matrix(size_t m, size_t n)
{
	return malloc(sizeof(float) * m * n * 2);
}

void start_clock()
{
	start_time = clock();
}
void stop_clock()
{
	stop_time = clock();
}
double get_clock_result_seconds()
{
	return (double)(stop_time - start_time) / CLOCKS_PER_SEC;
}
void print_time_seconds(double seconds)
{
#ifdef _WIN32
	printf("%0.3f seconds", seconds);
#elif __linux__
	printf("%0.9f seconds", seconds);
#endif
}