#pragma once

#include <time.h>

struct timespec start_time;
struct timespec stop_time;

void start_clock();
void stop_clock();
double get_clock_result_seconds();
void print_time_seconds(double seconds);