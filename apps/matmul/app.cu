#include "include/lk_host.h"
#include "core/lk_utils.h"
#include "data.h"
#include "include/lk_device.h"

#if 1
/* NOTE if you perform no logs, it crashes with -O2. Still to be explored. */
#define applog(...)
#else
#define applog(_s, ...)                                             \
{                                                                   \
  printf("[MATMUL] [%s] " _s, __func__, ##__VA_ARGS__);             \
}
#endif

float array1_h[WIDTH][WIDTH], array2_h[WIDTH][WIDTH],
      result_array_h[WIDTH][WIDTH], M_result_array_h[WIDTH][WIDTH];
data_t hostData[16];
res_t hostRes[16];

/*
 * lkInitAppData - Allocate application-specific data_t using lkDeviceAlloc
 */
void lkInitAppData(data_t **hostDataPtr, data_t **devDataPtr, res_t **hostResPtr, res_t ** devResPtr, int numSm)
{
//    applog("devDataPtr 0x%x resPtr 0x% numSm %d\n", _mycast_ devDataPtr, _mycast_ resPtr, numSm);
//    applog("numSm %d\n", numSm);
  
  *hostDataPtr = hostData;
  *hostResPtr = hostRes;
  applog("hostDataPtr @0x%x hostResPtr @0x%x\n", _mycast_  *hostDataPtr, _mycast_ *hostResPtr);
  
  lkDeviceAlloc((void **) devDataPtr, sizeof(data_t) * numSm);
  lkDeviceAlloc((void **) devResPtr, sizeof(res_t) * numSm);
  applog("devDataPtr @0x%x devResPtr @0x%x\n", _mycast_  *devDataPtr, _mycast_ *devResPtr);
   
  int i , j;
  //input in host array
  for (i=0; i<WIDTH; i++)
  {
    for (j=0; j<WIDTH; j++)
    {
      array1_h[i][j] = 1;
      array2_h[i][j] = 2;
    }
  }
  
  float *array1_d = NULL, *array2_d = NULL, *result_array_d = NULL; // device
  lkDeviceAlloc((void **) &array1_d, WIDTH*WIDTH*sizeof (int));
  lkDeviceAlloc((void **) &array2_d, WIDTH*WIDTH*sizeof (int));
  lkDeviceAlloc((void **) &result_array_d, WIDTH*WIDTH*sizeof (int));
  applog("array1_h @0x%x array2_h @0x%x array1_d @0x%x array1_d @0x%x\n", _mycast_  array1_h, _mycast_  array2_h, _mycast_  array1_d, _mycast_  array2_d);
  
  for(i=0; i<numSm; i++)
  {
    (*hostDataPtr)[i].array1_d = array1_d;
    (*hostDataPtr)[i].array2_d = array2_d;
    (*hostDataPtr)[i].startRow = -1;
    (*hostResPtr)[i].result_array_d = result_array_d;
  }
  
  // Big offload
  
  // Cannot copy hostDataPtr because it still misses a field. But I can copy hostResPtr as it's RO
  
  lkMemcpyToDevice ( &(*devResPtr)[0] , &(*hostResPtr)[0], sizeof(res_t) * numSm);
  
  lkMemcpyToDevice ( array1_d , array1_h , WIDTH*WIDTH*sizeof (int));
  lkMemcpyToDevice ( array2_d , array2_h , WIDTH*WIDTH*sizeof (int));  
  
  applog("Done \n");
} // lkInitAppData

/*
 * lkSmallOffload - Offload all of the SMs
 *                  RETURN 0 if you want the engine to go on and invoke lkWork[No]Cuda, !=0 otherwise - FIXME
 */
int lkSmallOffload(data_t *hostDataPtr, data_t *devDataPtr, int sm, int startRow)
{
//   applog("hostDataPtr 0x%x devDataPtr 0x%x sm %d\n", _mycast_ hostDataPtr, _mycast_ devDataPtr, sm);
  
  hostDataPtr[sm].startRow = startRow;
  
  lkMemcpyToDevice(&devDataPtr[sm], &hostDataPtr[sm], sizeof(data_t));
  
  return 0;
}

/*
 * lkSmallOffloadMultiple - Offlad all of the SMs
 *                          RETURN 0 if you want the engine to go on and invoke lkWork(No)Cuda, !=0 otherwise - FIXME
 */
int lkSmallOffloadMultiple(data_t *hostDataPtr, data_t *devDataPtr, int numSm)
{
  return 0;
}

/*
 * lkWorkCuda - Perform your work
 *              RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkCuda(volatile data_t *data, volatile res_t *res)
{
  applog("Hi! I'm core %d of cluster %d\n", lkGetCoreID(), lkGetClusterID());
  applog("data @x%x\n", _mycast_ data);
  applog("array1_d @0x%x array2_d @0x%x startRow %d\n", _mycast_  data->array1_d, _mycast_  data->array2_d, data->startRow);
  
  float *Md = data->array1_d,
        *Nd = data->array2_d,
        *Pd = res->result_array_d;
  
  for(int i=0; i<TILE_WIDTH; i++)
  {
    int row = data->startRow + i;
    for(int col=0; col<WIDTH; col++)
    {
      Pd[row*WIDTH + col] = 0;
        
      for (int k = 0 ; k<WIDTH ; k++ )
      {
        Pd[row*WIDTH + col] += Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;
    //     printf("k %d Md[%d] %f Nd[%d] %f Pd[%d] %f\n", k, row * WIDTH + k, Md[row * WIDTH + k ], k * WIDTH + col, Nd[ k * WIDTH + col], row*WIDTH + col, Pd[row*WIDTH + col]);
      }
    }
  }
  applog("Done \n");
  
  return 0;
}

/*
 * lkRetrieveData - Retrieve app results
 */
void lkRetrieveData(res_t *hostResPtr, res_t *devResPtr, res_t * resPtr, int sm)
{
}

/*
 * lkRetrieveDataMultiple - Retrieve app results
 */
void lkRetrieveDataMultiple(res_t *hostResPtr, res_t *devResPtr, unsigned int numSm)
{
  applog("hostResPtr @0x%x devResPtr @0x%x result_array_d @0x%x\n", _mycast_ &hostResPtr[0], _mycast_ &devResPtr[0],  _mycast_ hostResPtr->result_array_d);
  
  lkMemcpyFromDevice(&M_result_array_h[0][0], hostResPtr->result_array_d, WIDTH*WIDTH*sizeof(float));
  
} // lkRetrieveDataMultiple

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
  /* Timers for profing */
  struct timespec app_start, app_stop;
  long app_total = 0;
  
  /* Global vars */
  dim3 blkdim = (1);
  dim3 blknum = (1);
  bool cudaMode = true;
  int shmem = 0;
  
  
  data_t *hostDataPtr = 0, *devDataPtr = 0;
  res_t *hostResPtr = 0, *devResPtr = 0;
  
  
  lkParseCmdLine(argc, argv, &blknum, &blkdim, &shmem, &cudaMode);
  
//   lkInit(blknum.x, blkdim.x, shmem, cudaMode, &data, &res);
  lkInit(blknum.x, blkdim.x, shmem, cudaMode, &hostDataPtr, &hostResPtr, &devDataPtr, &devResPtr);
  
  applog("Device is ready to go!\n");
  applog("There are %d clusters!\n", lkNumClusters());
  
  /** WORK */
  
  unsigned int i, smCount, num_loops = 0;  
  
  clock_gettime(CLOCK_MONOTONIC, &app_start);
  for(i=0, smCount=0; i<WIDTH/TILE_WIDTH; i++, smCount = (smCount+1) % lkNumClusters())
  {    
    int startRow = i * TILE_WIDTH;
    applog("i %u smCount %u startRow %d\n", i, smCount, startRow);
    lkWaitSM(smCount);
    lkSmallOffload(&hostDataPtr[0], &devDataPtr[0], smCount, startRow);
    lkTriggerSM(smCount);
    
    num_loops++;
  }
    
  clock_gettime(CLOCK_MONOTONIC, &app_stop);
  app_total += clock_getdiff_nsec(app_start, app_stop);
    
  lkRetrieveDataMultiple(&hostResPtr[0], &devResPtr[0], 0);
    
  /** DISPOSE (DISPOSE) **/
  lkDispose();
  
  //printf the result array
  for ( i = 0 ; i<WIDTH ; i++ )
  {
    for (int j = 0 ; j < WIDTH ; j++ )
    {
      printf ("%f   ", M_result_array_h[i][j] ) ;
    }
    printf ("\n") ;
  }
   
  GETTIME_LOG("[TOTAL] %lu\n", app_total);
  if(num_loops)
    GETTIME_LOG("[ITER] %lu\n", app_total / num_loops);
}
#endif