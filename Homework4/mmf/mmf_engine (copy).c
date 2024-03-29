#include <omp.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "topology.h"

//#define _SMALL

#ifdef _SMALL
#define MAX_ITEM 40000LL
#else
#define MAX_ITEM 400000000LL
#endif

#define MAX_SWITCH 4000

//macro to describe subtree orientation (uplinks/downlinks)
#define UP 1
#define DOWN 0
#define SATURATION_GAP 0.00000001


extern int baseL[MAX_H+1];
extern int sizeL[MAX_H+1];
extern int label_array[MAX_NODE][MAX_H+1];// precomputed labels for each nodeid
extern double rate_allocation_vector[MAX_ITEM];  

// a large linkedlist
extern long long int value[MAX_ITEM];
extern long long int next[MAX_ITEM];
extern long long int head;


extern long long int flow_vector[MAX_ITEM];


// extern function prototypes
double timediff(struct timeval start, struct timeval end);
long long int listmalloc();
void init_list();


/** 
 * Data structures associated with sub-fat-tree  based saturation
 * 
 */
long long int nodehead[MAX_NODE][2];
long long int nodetail[MAX_NODE][2];
long long int saturated_nodelist_head;
long long int saturated_nodelist_tail;

/**
 * Initializes the sub-fat-tree level variables
*/
void init_node_list()
{
  int i;
  for (i=0; i<totNode; i++)
      nodehead[i][0] = nodehead[i][1]= MAX_ITEM;
      nodetail[i][0]=nodetail[i][1]=MAX_ITEM;
}



/**
 * XGFT Utility function
 * Given a physical node ID, returns the fanout of the node, depending on its fat-tree level.
 * Uses label_array[][] to lookup node level rather than calculating
 * Alternatively, could have used baseL[] to find node level
 * Requires prior population of xgft_W(global) through xgft_topology_init() function
 */
int fan_out_opt(int node_id){
  int i,fanout=1;
  if(node_id>totNode){
    printf("Illegal node ID %d\n", node_id);
    exit(0);
  }
  else {
    for(i=label_array[node_id][0];i>0;i--)
       fanout *=xgft_W[i];
  }
  return fanout;
}



/** 
 * Used to enable multiple xgft subtree saturations at a  time
 */
void insertsaturatednode(int nodeid, int direction){
  long long int i,j;
  i = listmalloc();
  value[i] = nodeid;
  j = listmalloc();
  value[j]=direction;

  next[i]=j;
  next[j] = MAX_ITEM;
  if(saturated_nodelist_head==MAX_ITEM){
     saturated_nodelist_head = i;
  }else{
    next[saturated_nodelist_tail]=i;
  }
  saturated_nodelist_tail=j;

}

/**
 * Inserts(maps) a flow with its associated sub-fat-trees
 * @param flowid 
 * @param nodeid fat-tree vertex representative of a sub-fat-tree
 * @direction either UP or DOWN 
 */
void insertflow(long long int flowid, long long int nodeid, int direction)
{
  long long int i;
  i = listmalloc();
  value[i] = flowid;
  next[i] = MAX_ITEM;
  if (nodehead[nodeid][direction] == MAX_ITEM)
      nodehead[nodeid][direction] = nodetail[nodeid][direction] = i;
  else {
      next[nodetail[nodeid][direction]] = i;
      nodetail[nodeid][direction] = i;
  }
}


/** function to calculate label recursively
 * @param label label instance (integer array) passed by reference
 * @param starting_index recursive variable, the number of digits to permute
 * i.e. starting index=3 generates all possible <W3, W2,W1> sublabels
 * @param label_0 the data(level id) to stuff at index 0
 * @param flow_id ID of the flow to insert at the permutated label corresponding node
 * @param direction direction of flow_id at 'location of insertion(node)'
 * upon hitting recursion boundary, generate the label after adding label_0(param) at index 0
 * then maps flow_id(param)  corresponding to the calculated label and given direction(param)
 */
void permute_and_insert(int *label,int starting_index, int label_0, int flow_id, int direction){
  int physical_id,x;
  int label_dup[MAX_H+1];

  if(starting_index==0){		//boundary condition
    label[0] = label_0;
    compute_nodeid(&physical_id, label);
    insertflow(flow_id, physical_id, direction);
  }
  else{
    for(x=0;x<xgft_W[starting_index-1];x++){	//not necessary to permute...wild card values would suffice
      copy_label(label_dup,label);
      label_dup[starting_index]= x;	//+1 to make sure we dnt mess up the [0] index of the label array
      permute_and_insert(label_dup, starting_index-1,label_0, flow_id, direction);
    }
 }
}



/**
 * MMF rate calculation routine based on Subtree based saturation 
 */

long long int xgft_mmf(char* filename, double *tot_mmf_bw, int* iteration_count, double *exec_time)
{

#ifdef _OPENMP
omp_set_nested(1);
#endif

  struct timeval t1,t2,t3,t4,t5;	//to calc subtotal/exec_time components
  FILE *fd;

  /** variables to process flows/SD pairs from trafficfile*/
  char buf[1000], *ch;
  int next_s, next_d;

  /** general purpose */
  long long int i, j, k;

  /** counters */
  long long int tot_flow=0;

  /* Variables/storage specific to the NON-LP iterative algorithm*/
  int iterator=0;			//used to return value by reference
  int all_saturated=0;                  //boundary condition flag
  int min_rate_limit_node;		//ID of the saturated xgft subtree/node
  int min_rate_limit_direction;	        //saturated subtree direction(UP/DOWN)
  double  min_rate_limit;
  double rate_limit[MAX_NODE][2];
  int count=0;
  //static long long int cumula_count=0;
  static long long int cumula_count=0;



  int nca_h;	//temporary NCA height
  int nodeid;	//temporary node id var
  int src_clone[MAX_H+1];
  int dst_clone[MAX_H+1];

  // %% Step 1: initilizing
  gettimeofday(&t1, NULL);

  //1.1 open file
  if ((fd = fopen(filename, "r")) == NULL) {
    printf("file %s does not exist.\n", filename);
    exit(0);
  }

  //1.2 init linked list
  init_list();
  init_node_list();

  //1.3 init arrays
  for(i=0;i<totNode;i++)
    compute_label(i, label_array[i]);
  for(i=0;i<MAX_ITEM;i++)
   rate_allocation_vector[i]=-1; 	//all unsaturated



  // %% Step 2: Read from file, insert into local data strint r_label[MAX_H+1]; ucture
  gettimeofday(&t2, NULL);
  printf("Time to initialize: %lf\n", timediff(t1,t2));
  fflush(0);
  ch = fgets(buf, 1000, fd);
  i = sscanf(buf, "%d %d", &next_s, &next_d);
  while (next_s > -1) {
    if (next_s == next_d) {
      ch = fgets(buf, 1000, fd);
      i = sscanf(buf, "%d %d", &next_s, &next_d);
      continue;
    }


    //to touch or annotate all the subtrees of a given SD pair,
    // we dont really need to spit out routing information
    //and due to symmetry, we dont need up/down routing either
    //instead, use up routing for both src and dst
    // be sure to add the nca level only once
    // there fore, up routing from src to nca_h, and up routing from dst to (nca_h-1)

    // calculate gap between leaf level and NCA of the given SD pair
    //complexity: |D|*|MAX_H| label_array lookups and |V| compute_label() calls
   // if calculated dynamically, would require 2*|D| compute_label() calls
    nca_h = xgft_h;
    while (label_array[next_s][nca_h] == label_array[next_d][nca_h]) nca_h--;
    //primer
    copy_label(src_clone,label_array[next_s]);
    copy_label(dst_clone,label_array[next_d]);
    insertflow(tot_flow,next_s, UP);
    insertflow(tot_flow, next_d,DOWN);
    for(i=1;i<nca_h;i++){		//for each switch level in bottom up traversal EXCLUDING NCA level
      src_clone[0] = dst_clone[0]=i;
      src_clone[i] = dst_clone[i] = 0;	// static version of DMODK..always pick link 0;

      compute_nodeid(&nodeid, src_clone);
      insertflow(tot_flow,nodeid, UP);	//generate all permutations from ith to 0th digits of the label

      compute_nodeid(&nodeid, dst_clone);
      insertflow(tot_flow,nodeid, DOWN);	//generate all permutations from ith to 0th digits of the label
    }

    tot_flow += 1;
    ch = fgets(buf, 1000, fd);
    i = sscanf(buf, "%d %d", &next_s, &next_d);
  }

  if(fd) fclose(fd);


  //Step %% 3: begin iterative algorithm
  gettimeofday(&t3, NULL);
  printf("Time to read from input: %lf\n", timediff(t2,t3));
  printf("Paths processed during input %lld\n", tot_flow);




  while(!all_saturated){
    iterator++;
    all_saturated=1;
    min_rate_limit = LLONG_MAX;		//temporary
    saturated_nodelist_head=saturated_nodelist_tail=MAX_ITEM;
    min_rate_limit_node=-1;

    //STEP 1: Find the most rate limiting/max loaded Node
    //iterations are done over nodeid since nodeids can
    //directly access data structure

    int minnodeid,mind;
    #pragma omp parallel for reduction(min:min_rate_limit) 
    for(i=0;i<xgft_h;i++){
      int fanout_var = fan_out_opt(baseL[i]);
      int d;
      int flow_per_node;
      double used_BW;
      long long int ptr,ptr2;

      for(nodeid=baseL[i];nodeid<baseL[i+1];nodeid++){
        for(d=UP;d>=DOWN;d--){
          flow_per_node=0;
          used_BW=0;

          ptr=nodehead[nodeid][d];
          if(ptr==MAX_ITEM) continue;
          while(ptr!=MAX_ITEM){
            if(rate_allocation_vector[value[ptr]]<0)flow_per_node++;
            else used_BW+=rate_allocation_vector[value[ptr]];
            ptr = next[ptr];
          }

          //used_BW/=fanout_var;
          //check for nodes which dnt directly get saturates but all flows flowing through it saturates elsewhere
          if(flow_per_node){
            rate_limit[nodeid][d] = ((xgft_BW[i]*fanout_var) - used_BW)*1.0/flow_per_node;
            {
               if(rate_limit[nodeid][d] < min_rate_limit){
                 min_rate_limit = rate_limit[nodeid][d];
                 //minnodeid=nodeid; mind=d;
                 //Reset the saturated list..since there is only one nodeid that saturates at new value
                // saturated_nodelist_head = saturated_nodelist_tail=MAX_ITEM;
	         //then insert the newly saturated node;
               //  insertsaturatednode(nodeid,d);
               }
               //else if (fabs(rate_limit-min_rate_limit)< SATURATION_GAP){
               //  insertsaturatednode(nodeid,d);
              // }
            } //openMP..end critical region

          } else nodehead[nodeid][d] = nodetail[nodeid][d]=MAX_ITEM;	//saturate the switch whose all associated flows are saturated
        }	//end d loop
      }	//end nodeid loop
    }  //end i loop
    long long ptr;
    count=0;
    //taking saturation out of main loop
    for(i=0;i<xgft_h;i++){
      //j = (i)?xgft_W[i-1]:1;
      for(nodeid=baseL[i];nodeid<baseL[i+1];nodeid++){        
        if(fabs(rate_limit[nodeid][0]-min_rate_limit)< SATURATION_GAP){
          ptr=nodehead[nodeid][0];
          while(ptr!=MAX_ITEM){
            if(rate_allocation_vector[value[ptr]]<0){
              rate_allocation_vector[value[ptr]]=min_rate_limit;
              count +=1;
              //printf("saturating flow #%d from uplink of node %d, counter set to %d\n", value[ptr], nodeid, count);
            }
            ptr = next[ptr];
          }
          nodehead[nodeid][0] = nodetail[nodeid][0]=MAX_ITEM;
        }
        if(fabs(rate_limit[nodeid][1]-min_rate_limit)< SATURATION_GAP){
          ptr=nodehead[nodeid][1];
          while(ptr!=MAX_ITEM){
            if(rate_allocation_vector[value[ptr]]<0){
              rate_allocation_vector[value[ptr]]=min_rate_limit;
              count+=1;
              //printf("saturating flow #%d from downlink of node %d, counter set to %d\n", value[ptr], nodeid, count);

            }
            ptr = next[ptr];
          }
          //nodehead[nodeid][0] = nodetail[nodeid][0]=MAX_ITEM;
          nodehead[nodeid][1] = nodetail[nodeid][1]=MAX_ITEM;
        }
      }
    }




    cumula_count+=count;
   
    //printf("At end of Iteration %d, max limiting rate :  %lf, new flows saturated: %d \n", iterator, min_rate_limit,count);
    //printf("%lld flows out of %lld saturated so far.\n",cumula_count,tot_flow);

    //STEP 4: evaluate loop boundary condition
    count=0;
    for(i=0;i<tot_flow;i++){
      if(rate_allocation_vector[i]<0){
         all_saturated=0;
         break;
      }
    }


  }//end while

  //Step 5: end iterative algorithm
  gettimeofday(&t4, NULL);
  printf("Time to run iterative algo: %lf, #iterations = %d\n", timediff(t3,t4), iterator);
  fflush(0);

  *tot_mmf_bw=0.0;
  fd = fopen("output","w");
  for(i=0;i<tot_flow;i++){
     fprintf(fd, "Flow[%lld] allocation: %3.2lf\n",i, rate_allocation_vector[i]*100);
  }
  fclose(fd);
  for(i=0;i<tot_flow;i++){
     *tot_mmf_bw= *tot_mmf_bw + rate_allocation_vector[i];
  }

  gettimeofday(&t5, NULL);
  printf("Time to calculate final mmf_bw: %lf\n", timediff(t4,t5));
  fflush(0);

  printf("total elapsed time at MMF: %lf\n", timediff(t1,t5));
  fflush(0);

  *iteration_count = iterator;
  *exec_time = timediff(t2,t5);

  printf("Average per-flow MMF rate is %lf\n", *tot_mmf_bw/tot_flow);

}




