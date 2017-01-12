/*
 *  LightKer - Light and flexible GPU persistent threads library
 *  Copyright (C) 2016  Paolo Burgio
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>

#include <pthread.h>
pthread_t syncThread;
volatile char syncThread_run = 1;
void *syncMalboxFrom(void * fake)
{
  log("\n");
  while(syncThread_run)
    lkMailboxFlush(false);
  log("Out of loop\n");
  pthread_exit(NULL);
  log("Exiting...\n");
}

/* LK internal headers */
#include "lk_globals.h"
#include "lk_utils.h"
#include "lk_time.h"

/* APP specific */
#include "data.h"

/* You might want to comment this, e.g., to profiling an "empty-yet-not-optimized" lkWait */
#define OPTIMIZE_LKWAIT

cudaStream_t backbone_stream, kernel_stream;

/* Check SMs exec status */
lk_result_t *lk_h_results, *lk_d_results;

/* Global vars */
dim3 blkdim = (1);
dim3 blknum = (1);
bool cudaMode = true;
int shmem = 0;

#ifdef OPTIMIZE_LKWAIT
unsigned int numWorkingSM = 0;
unsigned char *workingSM = NULL;
#endif /* OPTIMIZE_LKWAIT */

/* For debugging */
extern mailbox_elem_t *d_to_device, *d_from_device, *h_to_device, *h_from_device;


/** Useful APIs */
int lkNumThreadsPerSM()
{
  return blkdim.x;
}

int lkNumClusters()
{
  return blknum.x;
}


/** INIT phase */

/*
 * lkLaunch - Starts the kernel.
 *            Formerly known as 'init'
 */
void lkLaunch(void (*kernel) (volatile mailbox_elem_t *, volatile mailbox_elem_t *,  volatile data_t *, volatile res_t *, lk_result_t *),
              data_t *devDataPtr, res_t *devResPtr, lk_result_t *results,
              dim3 blknum, dim3 blkdim, int shmem)
{
//   log("# Blocks %d #Threads %d SHMEM dim %d\n", blknum.x, blkdim.x, shmem);

  // Trigger initialization
  for (int sm = 0; sm < blknum.x; sm++)
    lkHToDevice(sm) = THREAD_NOP;
  
  lkMailboxFlush(true);
  kernel <<< blknum, blkdim, shmem, kernel_stream >>> (d_to_device, d_from_device, devDataPtr, devResPtr, results);
  
  // Wait for LK thread(s) to be ready to work
  for (int sm = 0; sm < blknum.x; sm++)
  {
    while(lkHFromDevice(sm) != THREAD_NOP)
      lkMailboxFlushSM(false, sm);
  }
  
//   log("Done.\n");
} // lkLaunch

/*
 * lkInit - Initialization subroutine
 */
             
void lkInit(unsigned int blknum_x, unsigned int blkdim_x,
						int shmem, bool cudaMode, 
						data_t **hostDataPtr, res_t **hostResPtr,
						data_t **devDataPtr, res_t **devResPtr)
{ 
  log("Number of Blocks: %d number of threads per block: %d, shared memory dim: %d\n", blknum_x, blkdim_x, shmem);
  
  struct timespec spec_start, spec_stop;
  int deviceCount;
  dim3 tmp(blknum_x);
  blknum = tmp;
  dim3 tmp2(blkdim_x);
  blkdim = tmp2;
  
  /* Boot phase */
  GETTIME_TIC;

#ifdef OPTIMIZE_LKWAIT
  numWorkingSM = 0;
#endif /* OPTIMIZE_LKWAIT */
  
  workingSM = (unsigned char *) malloc(blknum_x * sizeof(unsigned char));
  for(int sm = 0; sm < blknum_x; sm++)
    workingSM[sm] = 0;
  
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
  boot_total = clock_getdiff_nsec(spec_start, spec_stop);
  
  /* Alloc phase: mailboxes*/
  GETTIME_TIC;
  
  /* Mailboxes */
  
  if(lkMailboxInit())
    die("Mailbox initialization failed\n");
  
  /* Results from Device */
  checkCudaErrors(cudaHostAlloc((void **)&lk_h_results, blknum_x * sizeof(lk_result_t), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc((void **)&lk_d_results, blknum_x * sizeof(lk_result_t)));

  /* App-specific data and res containers */
//   checkCudaErrors(cudaHostAlloc((void **) data, blknum_x * sizeof(data_t), cudaHostAllocDefault));
//   checkCudaErrors(cudaMalloc((void **) res, blknum_x * sizeof(res_t)));
  
  GETTIME_TOC;
  alloc_total = clock_getdiff_nsec(spec_start, spec_stop);
  
  /* Call application-specific initialization of data
   * 'Big offload' is performed here */
  verb("Invoke app-specific data initialization hostDataPtr @0x%x it is 0x%x,hostResPtr @0x%x it is 0x%x",
       _mycast_ hostDataPtr, _mycast_ *hostDataPtr, _mycast_ hostResPtr, _mycast_ *hostResPtr);
  verb(" devDataPtr @0x%x it is 0x%x, devResPtr @0x%x it is 0x%x\n",
       _mycast_ devDataPtr, _mycast_ *devDataPtr, _mycast_ devResPtr, _mycast_ *devResPtr);
  GETTIME_TIC;
  lkInitAppData(hostDataPtr, devDataPtr, hostResPtr, devResPtr, blknum_x);
  GETTIME_TOC;
  appalloc_total = clock_getdiff_nsec(spec_start, spec_stop);
//   verb("Allocated data 0x%x @0x%x\n", _mycast_ *data, _mycast_ data);
//   verb("Allocated res 0x%x @0x%x\n", _mycast_ *res, _mycast_ res);
  
  /* Launch phase */
  GETTIME_TIC;
  lkLaunch(cudaMode ? lkUniformPollingCuda : lkUniformPollingNoCuda,
           *devDataPtr, *devResPtr, &lk_d_results[0], blknum, blkdim, shmem);
  GETTIME_TOC;
  launch_total = clock_getdiff_nsec(spec_start, spec_stop);
    
  int rc = pthread_create(&syncThread, NULL, syncMalboxFrom, (void *) 0);
  if(rc)
    die("ERROR; return code from pthread_create() is %d\n", rc);
  
  printf("--- LIGHTKERNEL STARTED ---\n");  
  
} // lkInit

/** WORK phase */

/*
 * lkTriggerSM - Order the given SM to start working.
 *               Formerly known as 'work'
 */
void lkTriggerSM(int sm)
{
  struct timespec spec_start, spec_stop;
//   LK_WARN_NOT_SUPPORTED();
  log("SM %d\n", sm);
#ifdef OPTIMIZE_LKWAIT
  if(workingSM[sm])
  {
    warning("SM %d is already running!\n", sm);
//     lkWaitSM(sm);
  }
#endif

  GETTIME_TIC;
  lkHToDevice(sm) = THREAD_WORK;
  GETTIME_TOC;
  lkTriggerMultipleTime1 = clock_getdiff_nsec(spec_start, spec_stop);
  
//   log("Transfering all mailboxes to Device..\n");
  GETTIME_TIC;
  lkMailboxFlushAsync(true);
  GETTIME_TOC;
  lkTriggerMultipleTime2 = clock_getdiff_nsec(spec_start, spec_stop);
  
  
#ifdef OPTIMIZE_LKWAIT
  GETTIME_TIC;
  numWorkingSM ++;
  workingSM[sm] = 1;
  GETTIME_TOC;
  lkTriggerMultipleTime3 = clock_getdiff_nsec(spec_start, spec_stop);
#endif /* OPTIMIZE_LKWAIT */
    
  log("Done.\n");  

} // lkTriggerSM

long lkWaitTime1 = 0, lkWaitTime2 = 0, lkWaitTime3 = 0;
long lkTriggerMultipleTime1 = 0, lkTriggerMultipleTime2 = 0, lkTriggerMultipleTime3 = 0;
/*
 * lkTriggerMultiple - Order all SMs to start working.
 */
// int nflush;
void lkTriggerMultiple()
{
  struct timespec spec_start, spec_stop;
  log("blknum=%d\n", blknum.x);
//   lkMailboxFlushAsync(false);
  
#ifdef OPTIMIZE_LKWAIT
  if(numWorkingSM != 0)
  {
    // Some SM is working!
    warning("Some SM is already running!\n");
//     return;
  }
#endif /* OPTIMIZE_LKWAIT */

  GETTIME_TIC;
  for(int sm=0; sm<blknum.x; sm++)
  {
#if 0
    /* This is not necessary, if you use LK "in a good way", that is, 
       if you always "lkWaitMultiple" before issuing a lkTriggerMultiple */
    while(lkHFromDevice(sm) != THREAD_NOP)
    {
      lkMailboxFlushSM(false, sm);
    }
#endif
    lkHToDevice(sm) = THREAD_WORK;
  }
  GETTIME_TOC;
  lkTriggerMultipleTime1 = clock_getdiff_nsec(spec_start, spec_stop);
//     log("Triggering SM #%d\n", sm);
//   printf("Spent %lu ns waiting for the GPU to become ready\n", clock_getdiff_nsec(spec_start, spec_stop));
//   printf("Spent %lu ns waiting for the GPU to become ready (%d flush)\n", lkTriggerMultipleTime1, nflush);
//   lkMailboxPrint("lkTriggerMultiple", 0);
  
//   log("Transfering all mailboxes to Device..\n");
  GETTIME_TIC;
  lkMailboxFlushAsync(true);
  GETTIME_TOC;
  lkTriggerMultipleTime2 = clock_getdiff_nsec(spec_start, spec_stop);
  
#ifdef OPTIMIZE_LKWAIT
  GETTIME_TIC;
  numWorkingSM += blknum.x;
  for(int sm = 0; sm < blknum.x; sm++)
    workingSM[sm] = 1;
  GETTIME_TOC;
  lkTriggerMultipleTime3 = clock_getdiff_nsec(spec_start, spec_stop);
#endif /* OPTIMIZE_LKWAIT */
    
  log("Done.\n");  
} // lkTriggerMultiple

unsigned int lkProfiling = 0;
/*
 * lkWaitSM - Busy wait until the given sm is finished. Trigger to_device is restored to state "THREAD_NOP".
 *            Formerly known as 'sm_wait'
 */
void lkWaitSM(int sm)
{
  struct timespec spec_start, spec_stop;
  log("SM #%d\n", sm);

  if(workingSM[sm] == 0)
    return;
  
  GETTIME_TIC;
  do
  {
    lkMailboxFlushSM(false, sm);
  }
  while (lkHFromDevice(sm) != THREAD_FINISHED && !lkProfiling);
  GETTIME_TOC;
  if(lkProfiling) lkWaitTime2 += clock_getdiff_nsec(spec_start, spec_stop);
  
  GETTIME_TIC;
  lkHToDevice(sm) = THREAD_NOP;
  lkMailboxFlushSMAsync(true, sm);
  GETTIME_TOC;
  if(lkProfiling) lkWaitTime3 += clock_getdiff_nsec(spec_start, spec_stop);
  
  /* We should wait for mailbox_from to become _NOP, but it would be a waste of time */
  
#ifdef OPTIMIZE_LKWAIT
  numWorkingSM--;
  workingSM[sm] = 0;
#endif /* OPTIMIZE_LKWAIT */
  
  log("Done.\n");  
}

/*
 * lkWaitMultiple - Busy wait until all sms are finished. Trigger to_device is restored to state "THREAD_NOP".
 */
void lkWaitMultiple()
{
  struct timespec spec_start, spec_stop;
  log("numSm %d\n", blknum.x);
#ifdef OPTIMIZE_LKWAIT
  log("numWorkingSM %d\n", numWorkingSM);
#endif /* OPTIMIZE_LKWAIT */
  
#ifdef OPTIMIZE_LKWAIT
  // to perform test 'res.1'
  if(numWorkingSM != blknum.x)
    return;
#endif /* OPTIMIZE_LKWAIT */
  
  GETTIME_TIC;
  char allFinished = 1;
  do
  {
    allFinished = 1;
    lkMailboxFlush(false);
    for(unsigned sm = 0; sm<blknum.x; sm++)
    {
      if(!lkProfiling && lkHFromDevice(sm) != THREAD_FINISHED)
      {
        allFinished = 0;
        break;
      }
      else
      {
        lkHToDevice(sm) = THREAD_NOP;
      }
    } // for
  }
  while (!allFinished);
  GETTIME_TOC;
  if(lkProfiling) lkWaitTime2 += clock_getdiff_nsec(spec_start, spec_stop);
  
  GETTIME_TIC;
  lkMailboxFlushAsync(true);
  GETTIME_TOC;
  if(lkProfiling) lkWaitTime3 += clock_getdiff_nsec(spec_start, spec_stop);
  
#ifdef OPTIMIZE_LKWAIT
  numWorkingSM -= blknum.x;  // Remove to perform test 'res.1'
  for(int sm = 0; sm < blknum.x; sm++)
    workingSM[sm] = 0;
#endif /* OPTIMIZE_LKWAIT */
  
  log("Done.\n");  
}

/** DISPOSE phase */

/*
 * lkDispose - Order to the kernel to exit and wait for its termination.
 *             Formerly known as 'dispose'
 */
void lkDispose()
{
  log("There are %d SMs\n", blknum.x);

  for (int sm = 0; sm < blknum.x; sm++)
  {
//     lkMailboxPrint("lkDispose", sm);
//     log("Stopping SM #%d\n", sm);
    lkHToDevice(sm) = THREAD_EXIT;
  }
  
  log("Writing mailbox to device..\n");
  lkMailboxFlush(true);
  
  log("Halting syncThread..\n");
  syncThread_run = 0;
  
  log("Waiting for persistent LK thread to shut down...\n");
  checkCudaErrors(cudaDeviceSynchronize());
  log("Persistent LK thread stopped.\n");
  
  checkCudaErrors(cudaGetLastError());

//   log("Resetting Mailbox..\n");
  lkMailboxFree();
  
//   log("Resetting device..\n");
  checkCudaErrors(cudaDeviceReset());
  
//   log("Other deallocations..\n");
  free(workingSM);
  
//   pthread_exit(NULL);
  
  log("Done.\n");
} // lkDispose

void lkHostSleep(long time_ns)
{
    struct timespec ts;
    ts.tv_sec  = 0;
    ts.tv_nsec = time_ns;
    if(nanosleep(&ts, NULL) < 0)
    {
      printf("Error in sleep!\n");
    }
} // lkHostSleep