#include <stdio.h>

/* LK internal headers */
#include "lk_head.h"
#include "lk_utils.h"

/* App-specific data structures */
#include "data.h"

/** App-specific functions: defined by user */
/* Formerly known as 'work_cuda' */
__device__ int lkWorkCuda(volatile data_t data);
/* Formerly known as 'work_nocuda' */
__device__ int lkWorkNoCuda(volatile data_t data);

#define DeviceWriteMyMailboxFrom(_val)  _vcast(from_device[blockIdx.x]) = (_val)
#define DeviceReadMyMailboxFrom()       _vcast(from_device[blockIdx.x])
#define DeviceReadMyMailboxTo()         _vcast(to_device[blockIdx.x])

/* Main kernel function, for writing "non-cuda" work function.
 * Busy waits on the GPU until the CPU notifies new work, then
 * - it acknowledges the CPU and starts the real work. When finished
 * - it acknowledges the CPU through the trigger "from_device"
 * Formerly known as 'uniform_polling'
 */
__global__ void lkUniformPollingNoCuda(volatile mailbox_elem_t * to_device, volatile mailbox_elem_t * from_device,
                                       volatile data_t *data, int *lk_results)
{
  int blkid = blockIdx.x;
  int tid = threadIdx.x;
  
  log("I am thread %d block %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n",
      tid, blkid, __mysmid(), __mywarpid(), __mylaneid());
  return;
#if 0
  if (tid == 0)
  {
    while (1)
    {
//       volatile int to_device = _vcast(trig[blkid].to_device);
      volatile int to_device = lkToDevice(trig, blkid);

//       if (to_device == THREAD_EXIT)
      if (lkToDevice(trig, blkid) == THREAD_EXIT)
        break;

//       if (to_device == THREAD_WORK && _vcast(trig[blkid].from_device) != THREAD_FINISHED)
      if (lkToDevice(trig, blkid) == THREAD_WORK && lkFromDevice(trig, blkid) != THREAD_FINISHED)
      {
//         _vcast(trig[blkid].from_device) = THREAD_WORKING;
        lkFromDevice(trig, blkid) = THREAD_WORKING;
      
        log("Hi, I'm block %d and I received sth to do!\n clock(): %d\n", blkid, clock());
          
        lkWorkNoCuda(data[blkid]);
          
//         log("Work finished! Set data[%d].from_device to THREAD_FINISHED, lk_results[%d] is %d\n", blkid, blkid, lk_results[blkid]);
          
//         _vcast(trig[blkid].from_device) = THREAD_FINISHED;
        lkFromDevice(trig, blkid) = THREAD_FINISHED;
      }
    }
    log("I'm out of the while\n");
  }
#endif
}

/* Main kernel function, for writing cuda aware work function.
 * Busy waits on the GPU until the CPU notifies new work, then
 * - t acknowledges the CPU and starts the real work. When finished
 * - it acknowledges the CPU through the trigger "from_device"
 * Formerly known as 'uniform_polling_cuda'
 */
__global__ void lkUniformPollingCuda(volatile mailbox_elem_t * to_device, volatile mailbox_elem_t * from_device,
                                     volatile data_t *data, int *lk_results)
{
  int blkid = blockIdx.x;
  int tid = threadIdx.x;
  
  log("I am thread %d block %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n",
      tid, blkid, __mysmid(), __mywarpid(), __mylaneid());
//   log("mailbox TO @ 0x%x FROM 0x%X\n", _mycast_ to_device, _mycast_ from_device);
  
  if (tid == 0)
  {
//     log("Writing THREAD_NOP (%d) in from_device mailbox @0x%x\n", THREAD_NOP, (unsigned int) &from_device[blkid]);
    DeviceWriteMyMailboxFrom(THREAD_NOP);
//     log("Written %d in from_device mailbox @0x%x\n", from_device[blkid], (unsigned int) &from_device[blkid]);
  }
  
  
  while (1)
  {
    // Shut down
    if (DeviceReadMyMailboxTo() == THREAD_EXIT)
        break;

    // Time to work!
    else if (DeviceReadMyMailboxTo() == THREAD_WORK && DeviceReadMyMailboxFrom() != THREAD_FINISHED)
    {
      if (tid == 0)
        DeviceWriteMyMailboxFrom(THREAD_WORKING);
      
      log("Hi, I'm block %d and I received sth to do.\n", blkid);

      lk_results[blkid] = lkWorkCuda(data[blkid]);
      
      log("work finished! Set Mailbox from_device to THREAD_FINISHED, lk_results[%d] is %d\n",
          blkid, blkid, lk_results[blkid]);
      if (tid == 0)
      {
        DeviceWriteMyMailboxFrom(THREAD_FINISHED);
        log("Now Mailbox from_device is %d\n", DeviceReadMyMailboxFrom());
      }
//       break;
    } // if

    // Host got results
    else if (DeviceReadMyMailboxTo() == THREAD_NOP)
        DeviceWriteMyMailboxFrom(THREAD_NOP);
    
  } // while(1)
  
  log("I'm a thread and I'm out of the while\n");
} // lkUniformPollingCuda
