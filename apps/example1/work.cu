void CHECK_RESULTS(const char * input_str, unsigned int num, int numThreads, int sm)
{
  /* 
   * Check: each thread performed
   * int n = blockIdx.x * 100 + d_strlen((const char *) data->str);
   */
  printf("[EXAMPLE] Checking results from SM %u...", sm);
  int expected_n = sm * 100 + strlen(input_str);
  expected_n *= numThreads;
  if(expected_n != num)
    printf(" Error! sm %u expected %d, got %d\n", sm, expected_n, num);
  else
    printf(" Check passed!\n");
  
}

__device__ void WORK(const char * str, unsigned int * num)
{
  unsigned int n;
  if(threadIdx.x == 0)
    *num = 0;
  
  __syncthreads();
  clock_t clock_count = 200000;
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  
  while (clock_offset < clock_count)
  {
    clock_offset = clock() - start_clock;
  }
  
  /* Data has the format prova_<NUM_BLOCK> */
  
  //printf("[EXAMPLE] strlen(str) is %d\n" d_strlen(data->str)));
  n = blockIdx.x * 100 + d_strlen(str);
  atomicAdd((int *)num, n);
}

void INIT_DATA(char *str, int num)
{
  char buf[L_MAX_LENGTH];
  sprintf(buf, "prova_%d", num);
  printf("[EXAMPLE] block %d will work on string \"%s\"\n", num, buf);
  printf("[EXAMPLE] str @0x%x\n", _mycast_ str);
  strncpy(str, buf, L_MAX_LENGTH);
}