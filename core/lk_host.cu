#include <assert.h>

/* LK internal headers */
#include "lk_head.h"
#include "lk_utils.h"

/* APP specific */
#include "data.h"
void lkInitAppData(data_t **data, int numblocks);

/* Mailbox */
int lkMailboxInit(cudaStream_t stream);

cudaStream_t backbone_stream, kernel_stream;

/* Check SMs exec status */
lk_result_t *lk_h_results, *lk_d_results;

/* # of SMs */
// unsigned int wg;

/* For debugging */
extern mailbox_elem_t *d_to_device, *d_from_device, *h_to_device, *h_from_device;


/** INIT phase */

/*
 * lkLaunch - Starts the kernel.
 *            Formerly known as 'init'
 */
void lkLaunch(void (*kernel) (volatile mailbox_elem_t *, volatile mailbox_elem_t *,  volatile data_t *, volatile res_t *, lk_result_t *),
              data_t *data, res_t * res, lk_result_t *results,
              dim3 blknum, dim3 blkdim, int shmem)
{
  int wg = blknum.x;
//   log("# Blocks %d #Threads %d SHMEM dim %d\n", blknum.x, blkdim.x, shmem);

  // Trigger initialization
  for (int sm = 0; sm < wg; sm++)
    lkHToDevice(sm) = THREAD_NOP;
  
  lkMailboxFlush(true);
  kernel <<< blknum, blkdim, shmem, kernel_stream >>> (d_to_device, d_from_device, data, res, results);
  
  // Wait for LK thread(s) to be ready to work
  for (int sm = 0; sm < wg; sm++)
  {
//     log("Waiting for SM #%d @ 0x%x %d\n", sm, _mycast_ &d_from_device[sm], h_from_device[sm]);
    while(lkHFromDevice(sm) != THREAD_NOP)
    {
      lkMailboxFlushSM(false, sm);
    }
//     log("SM #%d is ready to go\n", sm);
  }
//   log("Done.\n");
} // lkLaunch

/*
 * lkInit - Initialization subroutine
 */
void lkInit(dim3 blknum, dim3 blkdim, int shmem, bool cudaMode, data_t ** data, res_t ** res)
{ 
  log("Number of Blocks: %d number of threads per block: %d, shared memory dim: %d\n", blknum.x, blkdim.x, shmem);
  struct timespec spec_start, spec_stop;
  int deviceCount;
  

  unsigned int wg = blknum.x;
  
  /* Boot phase */
  GETTIME_TIC;
  cudaDeviceReset();
  
  cudaGetDeviceCount(&deviceCount);
  /* Get device properties */
  int device;
  for (device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    log("[boot] Device canMapHostMemory: %s.\n", deviceProp.canMapHostMemory ? "yes" : "no");
    log("[boot] Device %d has async engine count %d.\n", device, deviceProp.asyncEngineCount);
  }
  
  /* Create kernel and backbone (mailbox) streams */
  checkCudaErrors(cudaStreamCreate(&kernel_stream));
  checkCudaErrors(cudaStreamCreate(&backbone_stream));
  
  GETTIME_TOC;
  GETTIME_LOG("boot(init) %lu\n", clock_getdiff_nsec(spec_start, spec_stop));
  
  
  /* Alloc phase: mailboxes*/
  GETTIME_TIC;
  
  /* Mailboxes */
  
  if(lkMailboxInit())
    die("Mailbox initialization failed\n");
  
  /* Results from Device */
  checkCudaErrors(cudaHostAlloc((void **)&lk_h_results, wg * sizeof(lk_result_t), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc((void **)&lk_d_results, wg * sizeof(lk_result_t)));

  /* App-specific data and res containers */
  checkCudaErrors(cudaHostAlloc((void **) data, wg * sizeof(data_t), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc((void **) res, wg * sizeof(res_t)));
  verb("Allocated data 0x%x @0x%x\n", _mycast_ *data, _mycast_ data);
  verb("Allocated res 0x%x @0x%x\n", _mycast_ *res, _mycast_ res);
  
  /* Call application-specific initialization of data
   * 'Big offload' is performed here */
  verb("Invoke app-specific data initialization data ptr @0x%x it is 0x%x, res ptr @0x%x it is 0x%x\n",
       _mycast_ data, _mycast_ *data, _mycast_ res, _mycast_ *res);
  lkInitAppData(data, res, wg);
  
  GETTIME_TOC;
  GETTIME_LOG("alloc(init) %lu\n", clock_getdiff_nsec(spec_start, spec_stop));
  
  /* Launch phase */
  GETTIME_TIC;
  lkLaunch(cudaMode ? lkUniformPollingCuda : lkUniformPollingNoCuda,
           *data, *res, &lk_d_results[0], blknum, blkdim, shmem);
  GETTIME_TOC;
  GETTIME_LOG("launch(init) %lu\n", clock_getdiff_nsec(spec_start, spec_stop));
  
  printf("--- LIGHTKERNEL STARTED ---\n");  
  
} // lkInit

/** WORK phase */

/*
 * lkTriggerSM - Order the given SM to start working.
 *               Formerly known as 'work'
 */
void lkTriggerSM(int sm, dim3 blknum)
{
  LK_WARN_NOT_SUPPORTED();
#if 0
  log("SM %d blknum %d\n", sm, blknum.x);
  assert(sm <= blknum.x);
//   assert(_vcast(trig[sm].from_device) != THREAD_WORK);
  assert(lkFromDevice(h_trig, sm) != THREAD_WORK);

//   _vcast(trig[sm].to_device) = THREAD_WORK;
  lkToDevice(h_trig, sm) = THREAD_WORK;
//   checkCudaErrors(cudaMemcpyAsync(&d_trig[sm], &trig[sm], sizeof(trig_t), cudaMemcpyHostToDevice, *backbone_stream));
  lkFlushMailboxSM(true, sm);
  print_trigger("[lkTriggerSM]", trig, sm);
#endif
}


/*
 * lkTriggerMultiple - Order all SMs to start working.
 */
void lkTriggerMultiple(dim3 blknum)
{
  log("blknum=%d\n", blknum.x);

  for(int sm=0; sm<blknum.x; sm++)
  {
    while(lkHFromDevice(sm) != THREAD_NOP)
    {
      lkMailboxFlushSM(false, sm);
    }
//     log("Triggering SM #%d\n", sm);
    lkHToDevice(sm) = THREAD_WORK;
  }
//   lkMailboxPrint("lkTriggerMultiple", 0);
  
//   log("Transfering all mailboxes to Device..\n");
  lkMailboxFlush(true);
  
  log("Done.\n");  
} // lkTriggerMultiple

/*
 * lkWaitSM - Busy wait until the given sm is working. Trigger to_device is restored to state "THREAD_NOP".
 *            Formerly known as 'sm_wait'
 */
void lkWaitSM(int sm, dim3 blknum)
{
  log("SM #%d\n", sm);
  
  if(lkHToDevice(sm) != THREAD_WORK)
  {
    printf("SM #%d was not triggered!", sm);
    return;
  }
  
  do
  {
    lkMailboxFlushSM(false, sm);
  }
  while (lkHFromDevice(sm) != THREAD_FINISHED);
  
  
  lkHToDevice(sm) = THREAD_NOP;
  lkMailboxFlushSM(true, sm);
  
  /* We should wait for mailbox_from to become _NOP, but it would be a waste of time */
  
  log("Done.\n");  
}


/** DISPOSE phase */

/*
 * lkDispose - Order to the kernel to exit and wait for its termination.
 *             Formerly known as 'dispose'
 */
void lkDispose(dim3 blknum)
{
  int wg = blknum.x;
  log("There are %d SMs\n", wg);
//   return;
  for (int sm = 0; sm < wg; sm++)
  {
//     lkMailboxPrint("lkDispose", sm);
//     log("Stopping SM #%d\n", sm);
    lkHToDevice(sm) = THREAD_EXIT;
  }
  
//   log("Writing mailbox to device..\n");
  lkMailboxFlush(true);
  
//   log("Waiting for persistent LK thread to shut down...\n");
  checkCudaErrors(cudaDeviceSynchronize());
  log("Persistent LK thread stopped.\n");
  
  checkCudaErrors(cudaGetLastError());

//   log("Resetting Mailbox..\n");
  lkMailboxFree();
  
//   log("Resetting device..\n");
  checkCudaErrors(cudaDeviceReset());
  
  log("Done.\n");
} // lkDispose


extern dim3 blkdim;
extern dim3 blknum;

int lkNumThreadsPerSM()
{
  return blkdim.x;
}

int lkNumSMs()
{
  return blknum.x;
}
