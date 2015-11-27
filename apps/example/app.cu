
void lkInitAppData(data_t **dataPtr, int numblocks)
{
  log("numblocks %d\n", numblocks);
//   checkCudaErrors(cudaHostAlloc((void **)data, numblocks * sizeof(data_t), cudaHostAllocDefault));
  lkDeviceAlloc((void **) dataPtr, numblocks * sizeof(data_t));
  log("Allocated data 0x%x @0x%x\n", _mycast_ *dataPtr, _mycast_ dataPtr);
}

int lkRetrieveData(data_t *data, int sm)
{
  log("sm %d \n", sm);
  return 0;
}

extern cudaStream_t backbone_stream;
int lkSmallOffload(data_t *data, int sm)
{
  log("sm %d \n", sm);
  data_t d;
  lkHostAlloc((void **) &d, L_MAX_LENGTH);
  strncpy(d.str, "prova", L_MAX_LENGTH);
  
  log("assigning data \"%s\" to thread %d\n", d.str, sm);
  lkMemcpyToDevice(&data[0], &d, L_MAX_LENGTH);
  
  log("assigned data \"%s\" to thread %d\n", (char *) d.str, sm);

  return 0;
}

int lkSmallOffloadMultiple(data_t *data, dim3 blknum)
{
  log("blknum.x %d \n", blknum.x);
  for(int sm =0; sm<blknum.x; sm++)
    lkSmallOffload(data, sm);
  return 0;
}

__device__ int lkWorkNoCuda(volatile data_t data)
{
  log("Hi! I'm block %d [NOCUDA]\n", blockIdx.x);
//   log("Hi! I'm block %d and I'm working on data ''%s'' [NOCUDA]\n", blockIdx.x, data.str);
//   clock_t clock_count = 200000;
//   clock_t start_clock = clock();
//   clock_t clock_offset = 0;
//   while (clock_offset < clock_count)
//           clock_offset = clock() - start_clock;
  return LK_EXEC_OK;
}

__device__ int lkWorkCuda(volatile data_t data)
{
  log("Hi! I'm block %d [CUDA]\n", blockIdx.x);
  log("Hi! I'm block %d and I'm working on data ''%s'' [CUDA]\n", blockIdx.x, data.str);
  clock_t clock_count = 200000;
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  if (threadIdx.x == 0) {
    while (clock_offset < clock_count)
    {
      log("Working...\n");
      clock_offset = clock() - start_clock;
    }
  }
  log("Work done [CUDA]\n", blockIdx.x);
  return LK_EXEC_OK;
}
