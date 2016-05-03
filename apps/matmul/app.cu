#include "include/lk_host.h"
#include "core/lk_utils.h"
#include "data.h"
#include "include/lk_device.h"

// data_t *host_data = 0;
// res_t *host_res = 0;
#define applog(_s, ...)                                             \
{                                                                   \
  printf("[MATMUL] [%s] " _s, __func__, ##__VA_ARGS__);             \
}

float array1_h[WIDTH][WIDTH], array2_h[WIDTH][WIDTH],
      result_array_h[WIDTH][WIDTH], M_result_array_h[WIDTH][WIDTH];
/*
 * lkInitAppData - Allocate application-specific data_t using lkDeviceAlloc
 */
void lkInitAppData(data_t **dataPtr, res_t **resPtr, int numSm)
{
//    applog("dataPtr 0x%x resPtr 0x% numSm %d\n", _mycast_ dataPtr, _mycast_ resPtr, numSm);
//    applog("numSm %d\n", numSm);
   
  int i , j;
  //input in host array
  for ( i = 0 ; i<WIDTH ; i++ )
  {
    for (j = 0 ; j<WIDTH ; j++ )
    {
      array1_h[i][j] = 1 ;
      array2_h[i][j] = 2 ;
    }
  }
  
  float *array1_d = NULL, *array2_d = NULL, *result_array_d = NULL; // device
  lkDeviceAlloc((void **) &array1_d, WIDTH*WIDTH*sizeof (int));
  lkDeviceAlloc((void **) &array2_d, WIDTH*WIDTH*sizeof (int));
  lkDeviceAlloc((void **) &result_array_d, WIDTH*WIDTH*sizeof (int));
  
  lkHostAlloc((void **) dataPtr, sizeof(data_t) * numSm);
  applog("dataPtr @0x%x\n", _mycast_  *dataPtr);
  
//   return;
  
  for(i=0; i<numSm; i++)
  {
//   applog("dataPtr @0x%x\n", _mycast_  *dataPtr);
    (*dataPtr)[i].array1_d = array1_d;
    (*dataPtr)[i].array2_d = array2_d;
    (*dataPtr)[i].result_array_d = result_array_d;
  }
  
  // Big offload
  lkMemcpyToDevice ( array1_d , array1_h , WIDTH*WIDTH*sizeof (int));
  lkMemcpyToDevice ( array2_d , array2_h , WIDTH*WIDTH*sizeof (int));
  
//   applog("array1_h @0x%x array2_h @0x%x\n", _mycast_  array1_h, _mycast_  array2_h);
//   applog("array1_d @0x%x array1_d @0x%x\n", _mycast_  array1_d, _mycast_  array2_d);
  
//   unsigned int nclusters = lkNumClusters();
//   unsigned int nrows = WIDTH / nclusters;
  
//   applog("Nclusters %u nrows %u\n", nclusters, nrows);
    
  applog("Done \n");
} // lkInitAppData

/*
 * lkSmallOffload - Offload all of the SMs
 *                  RETURN 0 if you want the engine to go on and invoke lkWork[No]Cuda, !=0 otherwise - FIXME
 */
int lkSmallOffload(data_t *data, int sm)
{
  return 0;
}

unsigned int count_iter = 0, max_iter = 2;
/*
 * lkSmallOffloadMultiple - Offlad all of the SMs
 *                          RETURN 0 if you want the engine to go on and invoke lkWork(No)Cuda, !=0 otherwise - FIXME
 */
int lkSmallOffloadMultiple(data_t *data, int numSm)
{
   applog("numSm %d data 0x%x count_iter %u max_iter %u\n", numSm, _mycast_ data, count_iter, max_iter);
  
  return ++count_iter != max_iter;
}
/*
 * lkWorkCuda - Perform your work
 *              RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkCuda(volatile data_t *data, volatile res_t *res)
{
  applog("Hi! I'm core %d of cluster %d\n", lkGetCoreID(), lkGetClusterID());
  applog("array1_d @0x%x array2_d @0x%x\n", _mycast_  data->array1_d, _mycast_  data->array2_d);
  
//   // calculate thread id
//   unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
//   unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
//   printf("blockIdx %d %d threadIdx %d %d ==> col %hu row %hu Md[0] %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, col, row, Md[0]);
//   Pd[row*WIDTH + col] = 0;
//   
// //   WORK(Md, Nd, Pd, WIDTH, col, row);
//   for (int k = 0 ; k<WIDTH ; k++ )
//   {
//     Pd[row*WIDTH + col] += Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;
// //     printf("k %d Md[%d] %f Nd[%d] %f Pd[%d] %f\n", k, row * WIDTH + k, Md[row * WIDTH + k ], k * WIDTH + col, Nd[ k * WIDTH + col], row*WIDTH + col, Pd[row*WIDTH + col]);
//   }

//   printf("[MATMUL] Block %d returns %d \n", blockIdx.x, res->num);
  
  return 0;
}

/*
 * lkRetrieveData - Retrieve app results
 */
void lkRetrieveData(res_t * resPtr, int sm)
{
//   printf("[MATMUL] sm %d \n", sm);
  
//   lkMemcpyFromDevice(&host_res[sm], &resPtr[sm], sizeof(res_t));
  
//   CHECK_RESULTS((const char *) host_data[sm].str, r.num, lkNumThreadsPerSM(), sm);
}


/*
 * lkRetrieveDataMultiple - Retrieve app results
 */
void lkRetrieveDataMultiple(res_t * resPtr, unsigned int numSm)
{
//   printf("[MATMUL] sm %d \n", sm);
//   
//   lkMemcpyFromDevice(&host_res[0], &resPtr[0], sizeof(res_t) * numSm);
}

/*
 * lkWorkNoCuda - Perform your work
 *                RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkNoCuda(volatile data_t *data, volatile res_t *res)
{
  return 1;
}

#if 1

/* Main */
int main(int argc, char **argv)
{  
  applog("\n");
  /* Global vars */
  dim3 blkdim = (1);
  dim3 blknum = (1);
  bool cudaMode = true;
  int shmem = 0;
  
  /* Timers for profing */
  struct timespec app_start, app_stop;
  long app_total = 0;
  
  data_t *data = 0;
  res_t *res = 0;
  
  lkParseCmdLine(argc, argv, &blknum, &blkdim, &shmem, &cudaMode);
  
  lkInit(blknum.x, blkdim.x, shmem, cudaMode, &data, &res);
  
  /** WORK */
  int more = 1;
  unsigned int num_loops = 0;  
 
  applog("Device is ready to go!\n");

  clock_gettime(CLOCK_MONOTONIC, &app_start);
  
  while (more)
  {
    /** SMALL OFFLOAD (WORK) **/
    more = lkSmallOffloadMultiple(data, blknum.x);
    
    /** TRIGGER (WORK) **/
    lkTriggerMultiple();
    
    /** WAIT (WORK) **/
    lkWaitMultiple();
    
    /** RETRIEVE DATA (WORK) **/
    lkRetrieveDataMultiple(res, blknum.x);
    
//     struct timespec ts;
//     ts.tv_sec  = 0;
//     ts.tv_nsec = 6000L;
//     if(nanosleep(&ts, NULL) < 0)
//     {
//       printf("Error in sleep!\n");
//     }
      
    num_loops++;
  } // work loop
  
  clock_gettime(CLOCK_MONOTONIC, &app_stop);
  app_total += clock_getdiff_nsec(app_start, app_stop);
    
    
  /** DISPOSE (DISPOSE) **/
  lkDispose();
   
  GETTIME_LOG("[TOTAL] %lu\n", app_total);
  GETTIME_LOG("[ITER] %lu\n", app_total / num_loops);
}
#endif