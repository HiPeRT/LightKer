#include <stdio.h>

/* LK internal headers */
#include "lk_head.h"
#include "lk_utils.h"

/* App-specific data structures */
#include "data.h"

/** App-specific functions: defined by user */
/* Formerly known as 'work_cuda' */
__device__ int lkWorkCuda(volatile data_t *data, volatile res_t *res);
/* Formerly known as 'work_nocuda' */
__device__ int lkWorkNoCuda(volatile data_t data, volatile res_t *res);

#define DeviceWriteMyMailboxFrom(_val)  _vcast(from_device[blockIdx.x]) = (_val)
#define DeviceReadMyMailboxFrom()       _vcast(from_device[blockIdx.x])
#define DeviceReadMyMailboxTo()         _vcast(to_device[blockIdx.x])

/* Main kernel function, for writing cuda aware work function.
 * Busy waits on the GPU until the CPU notifies new work, then
 * - t acknowledges the CPU and starts the real work. When finished
 * - it acknowledges the CPU through the trigger "from_device"
 * Formerly known as 'uniform_polling_cuda'
 */
__global__ void lkUniformPollingCuda(volatile mailbox_elem_t * to_device,
                                     volatile mailbox_elem_t * from_device,
                                     volatile data_t *data, volatile res_t * res,
                                     lk_result_t *lk_results)
{
  __shared__ int res_shared;
  int blkid = blockIdx.x;
  int tid = threadIdx.x;
  
  log("I am thread %d block %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n",
      tid, blkid, __mysmid(), __mywarpid(), __mylaneid());
//   log("data ptr @0x%x res ptr @0x%x lk_results @0x%x\n", _mycast_ data, _mycast_ res, _mycast_ lk_results);

  if(blkid != __mysmid())
  {
    /* Error! */
  }
  
  __syncthreads();

  if (tid == 0)
  {
//     log("mailbox TO @ 0x%x FROM 0x%X\n", _mycast_ &to_device[blkid], _mycast_ &from_device[blkid]);
//     log("Writing THREAD_NOP (%d) in from_device mailbox @0x%x\n", THREAD_NOP, (unsigned int) &from_device[blkid]);
    DeviceWriteMyMailboxFrom(THREAD_NOP);
    res_shared = 0;
//     log("Written THREAD_NOP (%d) in from_device mailbox @0x%x\n", from_device[blkid], (unsigned int) &from_device[blkid]);
  }  
  __syncthreads();
  
  while (1)
  {
    // Shut down
    if (DeviceReadMyMailboxTo() == THREAD_EXIT)
        break;

    // Time to work!
    else if (DeviceReadMyMailboxTo() == THREAD_WORK && DeviceReadMyMailboxFrom() != THREAD_FINISHED)
    {
      if (tid == 0)
      {
        DeviceWriteMyMailboxFrom(THREAD_WORKING);
//         log("Hi, I'm block %d and I received sth to do.\n", blkid);
        res_shared = 0;
      }
      __syncthreads();

      /* if res_shared == 0 => threads finished correctly */
      int lk_res = lkWorkCuda(&data[blkid], &res[blkid]);
      atomicAdd(&res_shared, lk_res);

      __syncthreads();

      if (tid == 0)
      {
        lk_results[blkid] = res_shared == 0 ? LK_EXEC_OK : LK_EXEC_APP_ERR;
        
//         log("Work finished! Set Mailbox from_device to THREAD_FINISHED (%d), lk_results[%d] is %d\n", THREAD_FINISHED, blkid, lk_results[blkid]);
        DeviceWriteMyMailboxFrom(THREAD_FINISHED);
//         log("Now Mailbox from_device is %d\n", DeviceReadMyMailboxFrom());
      }
//       break;
    } // if

    // Host got results
    else if (DeviceReadMyMailboxTo() == THREAD_NOP)
        DeviceWriteMyMailboxFrom(THREAD_NOP);
    
    else
    {}
    
  } // while(1)
  
  __syncthreads();
  if(tid == 0)
    log("SM %d. Shutdown complete.\n", blkid);
  
} // lkUniformPollingCuda



/* Main kernel function, for writing "non-cuda" work function.
 * Busy waits on the GPU until the CPU notifies new work, then
 * - it acknowledges the CPU and starts the real work. When finished
 * - it acknowledges the CPU through the trigger "from_device"
 * Formerly known as 'uniform_polling'
 */
__global__ void lkUniformPollingNoCuda(volatile mailbox_elem_t * to_device,
                                       volatile mailbox_elem_t * from_device,
                                       volatile data_t *data, volatile res_t * res,
                                       lk_result_t *lk_results)
{
  LK_WARN_NOT_SUPPORTED();
}