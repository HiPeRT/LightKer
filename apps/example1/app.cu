#include "lk_host.h"
#include "lk_utils.h"
#include "data.h"

unsigned char DATA_ON_DEVICE = 1;
unsigned char RES_ON_DEVICE = 1;

data_t *host_data = 0;
res_t *host_res = 0;
unsigned int count_iter = 0, max_iter = 10;

/* Contains INIT_DATA, WORK and CHECK_RESULTS functions */
#include "work.cu"

/*
 * lkInitAppData - Allocate application-specific data_t using lkDeviceAlloc
 */
void lkInitAppData(data_t **dataPtr, res_t **resPtr, int numSm)
{ 
  int sm;
    
  if(DATA_ON_DEVICE)
  {
    lkHostAlloc((void **) &host_data, sizeof(data_t) * numSm);
    lkDeviceAlloc((void **) dataPtr, sizeof(data_t) * numSm);
  }
  else
  {
    lkHostAlloc((void **) dataPtr, sizeof(data_t) * numSm);
    host_data = *dataPtr;
  }
  
  if(RES_ON_DEVICE)
  {
    lkHostAlloc((void **) &host_res, sizeof(res_t) * numSm);
    lkDeviceAlloc((void **) resPtr, sizeof(res_t) * numSm);
  }
  else
  {
    lkHostAlloc((void **) resPtr, sizeof(res_t) * numSm);
    host_res = *resPtr;
  }
  
  for(sm=0; sm<numSm; sm++)
    INIT_DATA(host_data[sm].str, sm);
} // lkInitAppData


/*
 * lkSmallOffload - Offload all of the SMs
 *                  RETURN 0 if you want the engine to go on and invoke lkWork[No]Cuda, !=0 otherwise - FIXME
 */
int lkSmallOffload(data_t *dataPtr, int sm)
{
//   printf("[EXAMPLE1] assigning data \"%s\" block %d sizeof(data_t) is %lu\n", host_data[sm].str, sm, sizeof(data_t));
//   printf("[EXAMPLE1] dataPtr 0x%x\n", _mycast_ dataPtr);
  printf("lkSmallOffload not implemented!\n");
  
//   if(DATA_ON_DEVICE)
//     lkMemcpyToDevice(&dataPtr[sm], &host_data[sm], sizeof(data_t));
  
  return 1;
}

/*
 * lkSmallOffloadMultiple - Offlad all of the SMs
 *                          RETURN 0 if you want the engine to go on and invoke lkWork(No)Cuda, !=0 otherwise - FIXME
 */
int lkSmallOffloadMultiple(data_t *dataPtr, int numSm)
{
//   printf("[EXAMPLE1] numSm %d dataPtr 0x%x count_iter %u max_iter %u\n", numSm, _mycast_ dataPtr, count_iter, max_iter);
    
  if(DATA_ON_DEVICE)
    lkMemcpyToDevice(&dataPtr[0], &host_data[0], sizeof(data_t) *numSm);

  return ++count_iter != max_iter;
}
/*
 * lkWorkCuda - Perform your work
 *              RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkCuda(volatile data_t *dataPtr, volatile res_t *res)
{
//   printf("[EXAMPLE1] Hi! I'm core %d of SM %d and I'm working on data ''%s''\n", threadIdx.x, blockIdx.x, data->str);

  WORK((const char *) dataPtr->str, (unsigned int *) &res->num);

//   printf("[EXAMPLE1] Block %d returns %d \n", blockIdx.x, res->num);
  
  return 0;
}

/*
 * lkRetrieveData - Retrieve app results
 */
void lkRetrieveData(res_t * resPtr, int sm)
{
//   printf("[EXAMPLE1] sm %d \n", sm);
  printf("lkRetrieveData not implemented!\n");
  return;
  
//   if(RES_ON_DEVICE)
//     lkMemcpyFromDevice(&host_res[sm], &resPtr[sm], sizeof(res_t));
  
//   CHECK_RESULTS((const char *) host_data[sm].str, resPtr[sm].num, lkNumThreadsPerSM(), sm);
}


/*
 * lkRetrieveDataMultiple - Retrieve app results
 */
void lkRetrieveDataMultiple(res_t * resPtr, unsigned int numSm)
{
//   printf("[EXAMPLE1] sm %d \n", sm);

  if(RES_ON_DEVICE)
    lkMemcpyFromDevice(&host_res[0], &resPtr[0], sizeof(res_t) * numSm);
  
//   for(int sm =0; sm<numSm; sm++)
//     CHECK_RESULTS((const char *) host_data[sm].str, host_res[sm].num, lkNumThreadsPerSM(), sm);
}

/*
 * lkWorkNoCuda - Perform your work
 *                RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkNoCuda(volatile data_t *data, volatile res_t *res)
{
  printf("lkWorkNoCuda not implemented!\n");
  return 1;
}

