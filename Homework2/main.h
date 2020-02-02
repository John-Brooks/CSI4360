#pragma once

#include <stdlib.h>

void print_result(const char* function_name, double time);
void* allocate_double_matrix(size_t m, size_t n);
void naive_matmul(int N, double *A, double *B, double *C);
void single_optimization(int N, double *A, double *B, double *C);