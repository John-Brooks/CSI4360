#include "performance_clock.h"
#include <stdio.h>

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
	printf("%0.9f seconds", seconds);
}