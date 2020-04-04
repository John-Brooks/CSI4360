#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "topology_var.h"

extern void calculate_node_levels();

//in a separate file
extern void xgft_mmf_OPT (char* filename, double* throughput, int* iteration_count, double *exec_time);

struct timeval t;

int main() 
{
  int i;
  int traffic_count;
  int iteration_count=0;
  double exec_time=0.0;
  double  opt_param_throughput=0;

  //initialize params for an 11,664 node fat tree network
  int h=3;
  int M[MAX_H]={18, 18, 36};
  int W[MAX_H] = {1, 18, 18};
  long long int BW[MAX_H] = {1, 1, 1};

  xgft_topology_init(h, M, W, BW, XGFT_KPATH_ROUTING);
  xgft_mmf("input_file",&opt_param_throughput, &iteration_count, &exec_time);

 return 0;
}


