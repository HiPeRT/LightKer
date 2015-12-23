#include "lk_host.h"

data_t *host_data = 0;
unsigned int count_iter = 0, max_iter = 10;

/*
 * lkInitAppData - Allocate application-specific data_t using lkDeviceAlloc
 */
void lkInitAppData(data_t **dataPtr, res_t **resPtr, int numsm)
{  
  lkHostAlloc((void **) &host_data, sizeof(data_t) * numsm);
} // lkInitAppData

/*
 * lkRetrieveData - Retrieve app results
 */
int lkRetrieveData(res_t * resPtr, int sm)
{
  res_t r;
//   printf("[EXAMPLE] sm %d \n", sm);
  lkHostAlloc((void **) &r, sizeof(int));
  
  lkMemcpyFromDevice(&r, &resPtr[sm], sizeof(int));
  
  /* 
   * Check: each thread performed
   * int n = blockIdx.x * 100 + d_strlen((const char *) data->str);
   */
  printf("[EXAMPLE] Checking results from SM %u\n", sm);
  int expected_n = sm * 100 + strlen((const char *) host_data[sm].str);
  expected_n *= lkNumThreadsPerSM();
  if(expected_n != r.num)
    printf("[EXAMPLE] Error! sm %u expected %d, got %d\n", sm, expected_n, r.num);
  
  return r.num;
}

/*
 * lkSmallOffload - Offlad all of the SMs
 *                  RETURN 0 if you want the engine to go on and invoke lkWork(No)Cuda, !=0 otherwise
 */
int lkSmallOffload(data_t *data, int sm)
{
//   printf("[EXAMPLE] assigning data \"%s\" to block %d\n", d.str, sm);
  char buf[L_MAX_LENGTH];
  sprintf(buf, "prova_%d_%d", sm, count_iter);
  strncpy(host_data[sm].str, buf, L_MAX_LENGTH);
  lkMemcpyToDevice(&data[sm], &host_data[sm], L_MAX_LENGTH);
  
//   printf("[EXAMPLE] assigned data \"%s\" to block %d\n", (char *) host_data[sm].str, sm);
  
  return 0;
}

/*
 * lkSmallOffloadMultiple - Offlad all of the SMs
 *                          RETURN 0 if you want the engine to go on and invoke lkWork(No)Cuda, !=0 otherwise
 */
int lkSmallOffloadMultiple(data_t *data, int smnum)
{
//   printf("[EXAMPLE] smnum %d data 0x%x count_iter %u max_iter %u\n",
//       smnum, _mycast_ data, count_iter, max_iter);
  
  for(int sm =0; sm<smnum; sm++)
    lkSmallOffload(data, sm);
  
  return ++count_iter != max_iter;
}

/*
 * lkWorkCuda - Perform your work
 *              RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkCuda(volatile data_t *data, volatile res_t *res)
{
//   printf("[EXAMPLE] Hi! I'm core %d of SM %d and I'm working on data ''%s''\n", threadIdx.x, blockIdx.x, data->str);
  
  if(threadIdx.x == 0)
    res->num = 0;
  
  __syncthreads();
  clock_t clock_count = 200000;
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  
  while (clock_offset < clock_count)
  {
    clock_offset = clock() - start_clock;
  }
  
  /* Data has the format prova_<NUM_BLOCK>_<NUM_TEST> */
  
  //printf("[EXAMPLE] strlen(data.str) is %d\n" d_strlen(data->str)));
  int n = blockIdx.x * 100 + d_strlen((const char *) data->str);
  //res->num = n;
  atomicAdd((int *)&res->num, n);
//   printf("[EXAMPLE] Work done, returning %d\n", res->num);
  
  return 0;
}


/*
 * lkWorkNoCuda - Perform your work
 *                RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkNoCuda(volatile data_t *data, volatile res_t *res)
{
  printf("[EXAMPLE] Hi! I'm block %d [NOCUDA]\n", blockIdx.x);
  return 1;
}

