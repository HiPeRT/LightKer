// #include "../../head/head.h"
// #include "../../head/utils.h"

void lkInitAppData(data_t **data, int numblocks)
{
  checkCudaErrors(cudaHostAlloc((void **)data, numblocks * sizeof(data_t), cudaHostAllocDefault));
}

int lkRetrieveData(data_t *data, int sm,
                  cudaStream_t *backbone_stream)
{
#if 0
    do {
        checkCudaErrors(cudaMemcpyAsync(&trig[sm], &d_trig[sm], sizeof(trig_t),
                cudaMemcpyDeviceToHost, *backbone_stream));
        log("waiting (retrieve) for %d [%d]\n",  _vcast(trig[sm].to_device), _vcast(trig[sm].from_device));
    } while (_vcast(trig[sm].from_device) != THREAD_FINISHED);

    _vcast(trig[sm].to_device) = THREAD_NOP;
    checkCudaErrors(cudaMemcpyAsync(&d_trig[sm], &trig[sm], sizeof(trig_t),
            cudaMemcpyHostToDevice, *backbone_stream));
    log("retrieve %d %d\n", _vcast(trig[sm].from_device), _vcast(trig[sm].to_device));
#endif
    return 0;
}

int lkSmallOffload(data_t *data, int sm, cudaStream_t *backbone_stream)
{
  strncpy(data[sm].str, "prova", L_MAX_LENGTH);
  log("assigned data \"%s\" to thread %d\n", (char *) data[sm].str, sm);

  return 0;
}

int lkSmallOffloadMultiple(data_t *data, dim3 blknum,  cudaStream_t *backbone_stream)
{
  for(int sm =0; sm<blknum.x; sm++)
    lkSmallOffload(data, sm, backbone_stream);
  return 0;
}

__device__ int lkWorkNoCuda(volatile data_t data)
{
  log("Hi! I'm block %d and I'm working on data ''%s'' [NOCUDA]\n", blockIdx.x, data.str);
  clock_t clock_count = 200000;
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count)
          clock_offset = clock() - start_clock;
  return LK_EXEC_OK;
}

__device__ int lkWorkCuda(volatile data_t data)
{
  log("Hi! I'm block %d and I'm working on data ''%s'' [CUDA]\n", blockIdx.x, data.str);
  clock_t clock_count = 200000;
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  if (threadIdx.x == 0) {
          while (clock_offset < clock_count)
                  clock_offset = clock() - start_clock;
  }
  return LK_EXEC_OK;
}
