#include <stdio.h>
#include "data.h"

void CHECK_RESULTS(const char * input_str, unsigned int num, int numThreads, int sm)
{
#if 0
  /* 
   * Check: each thread performed
   * int n = blockIdx.x * 100 + d_strlen((const char *) data->str);
   */
  printf("[WORK] Checking results from SM %u...", sm);
  int expected_n = sm * 100 + strlen(input_str);
  expected_n *= numThreads;
  if(expected_n != num)
    printf(" Error! sm %u expected %d, got %d\n", sm, expected_n, num);
  else
    printf(" Check passed!\n");
#endif
}

__device__ void WORK(const char * str, unsigned int * num)
{  
//   printf("[WORK] blk %u thrd %u will work on '%s' WORK_TIME %u\n", blockIdx.x, threadIdx.x, str, WORK_TIME);
  if(threadIdx.x == 0)
    *num = 0;
  
  __syncthreads();
  clock_t clock_count = WORK_TIME;
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  
  while (clock_offset < clock_count)
  {
    clock_offset = clock() - start_clock;
  }
  
  /* Data has the format prova_<NUM_BLOCK> */
#if 1
  unsigned int n = blockIdx.x * 100 + d_strlen(str);
  atomicAdd((int *)num, n);
#endif
}

void INIT_DATA(char *str, unsigned int num)
{
  char buf[L_MAX_LENGTH];
  sprintf(buf, "prova_%d", num);
//   printf("[WORK] block %d will work on string \"%s\"\n", num, buf);
//   printf("[WORK] str @0x%x\n", _mycast_ str);
  strncpy(str, buf, L_MAX_LENGTH);
}