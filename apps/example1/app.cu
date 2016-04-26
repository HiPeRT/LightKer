#include "lk_host.h"

data_t *host_data = 0;
unsigned int count_iter = 0, max_iter = 1;

/* Contains INIT_DATA, WORK and CHECK_RESULTS functions */
#include "work.cu"

/*
 * lkInitAppData - Allocate application-specific data_t using lkDeviceAlloc
 */
void lkInitAppData(data_t **dataPtr, res_t **resPtr, int numsm)
{ 
  int sm;
  lkHostAlloc((void **) &host_data, sizeof(data_t) * numsm);
  
  for(sm=0; sm<numsm; sm++)
    INIT_DATA(host_data[sm].str, sm);
} // lkInitAppData


/*
 * lkSmallOffload - Offload all of the SMs
 *                  RETURN 0 if you want the engine to go on and invoke lkWork(No)Cuda, !=0 otherwise
 */
int lkSmallOffload(data_t *data, int sm)
{
//   printf("[EXAMPLE] assigning data \"%s\" to block %d\n", host_data[sm].str, sm);
  lkMemcpyToDevice(&data[sm], &host_data[sm], L_MAX_LENGTH);
  
  return 0;
}

/*
 * lkSmallOffloadMultiple - Offlad all of the SMs
 *                          RETURN 0 if you want the engine to go on and invoke lkWork(No)Cuda, !=0 otherwise
 */
int lkSmallOffloadMultiple(data_t *data, int smnum)
{
//   printf("[EXAMPLE] smnum %d data 0x%x count_iter %u max_iter %u\n", smnum, _mycast_ data, count_iter, max_iter);
  
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

  WORK((const char *) data->str, (unsigned int *) &res->num);

  printf("[EXAMPLE] Block %d returns %d \n", blockIdx.x, res->num);
  
  return 0;
}

/*
 * lkRetrieveData - Retrieve app results
 */
int lkRetrieveData(res_t * resPtr, int sm)
{
  res_t r;
//   printf("[EXAMPLE] sm %d \n", sm);
  lkHostAlloc((void **) &r, sizeof(int));
  
  lkMemcpyFromDevice(&r, &resPtr[sm], sizeof(int));
  
  CHECK_RESULTS((const char *) host_data[sm].str, r.num, lkNumThreadsPerSM(), sm);
  
  return r.num;
}

/*
 * lkWorkNoCuda - Perform your work
 *                RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkNoCuda(volatile data_t *data, volatile res_t *res)
{
  return 1;
}

