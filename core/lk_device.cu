#include <stdio.h>

/* LK internal headers */
#include "head.h"
#include "utils.h"

/* App-specific data structures */
#include "data.h"

/** App-specific functions: defined by user */
/* Formerly known as 'work_cuda' */
__device__ int lkWorkCuda(volatile data_t data);
/* Formerly known as 'work_nocuda' */
__device__ int lkWorkNoCuda(volatile data_t data);


/* Main kernel function, for writing "non-cuda" work function.
 * Busy waits on the GPU until the CPU notifies new work, then
 * - it acknowledges the CPU and starts the real work. When finished
 * - it acknowledges the CPU through the trigger "from_device"
 * Formerly known as 'uniform_polling'
 */
__global__ void lkUniformPollingNoCuda(volatile trig_t *trig, volatile data_t *data, int *lk_results)
{
    int blkid = blockIdx.x;
    int tid = threadIdx.x;
    
    log("I am thread %d block %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n",
        tid, blkid, __mysmid(), __mywarpid(), __mylaneid());

    if (tid == 0)
    {
        while (1)
        {
            volatile int to_device = _vcast(trig[blkid].to_device);

            //dispose
            if (to_device == THREAD_EXIT)
                break;

            if (to_device == THREAD_WORK && _vcast(trig[blkid].from_device) != THREAD_FINISHED)
            {
                _vcast(trig[blkid].from_device) = THREAD_WORKING;
            
//                 log("Hi, I'm block %d and I received sth to do!\n clock(): %d\n", blkid, clock());
                
                /*lk_results[blkid] = */ lkWorkNoCuda(data[blkid]);
                
//                 log("Work finished! Set data[%d].from_device to THREAD_FINISHED, lk_results[%d] is %d\n", blkid, blkid, lk_results[blkid]);
                
                _vcast(trig[blkid].from_device) = THREAD_FINISHED;
            }
        }
        log("I'm out of the while\n");
    }
}

/* Main kernel function, for writing cuda aware work function.
 * Busy waits on the GPU until the CPU notifies new work, then
 * - t acknowledges the CPU and starts the real work. When finished
 * - it acknowledges the CPU through the trigger "from_device"
 * Formerly known as 'uniform_polling_cuda'
 */
__global__ void lkUniformPollingCuda(volatile trig_t *trig, volatile data_t *data, int *lk_results)
{
	int blkid = blockIdx.x;
	int tid = threadIdx.x;
    
    log("I am thread %d block %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n",
        tid, blkid, __mysmid(), __mywarpid(), __mylaneid());

    if (tid == 0)
      _vcast(trig[blkid].from_device) = THREAD_NOP;

    while (1)
    {
        if (_vcast(trig[blkid].to_device) == THREAD_EXIT)
            break;

        if (_vcast(trig[blkid].to_device) == THREAD_WORK && _vcast(trig[blkid].from_device) != THREAD_FINISHED)
        {
            if (tid == 0)
            {
                _vcast(trig[blkid].from_device) = THREAD_WORKING;
            }
            
            log("Hi, I'm block %d and I received sth to do!\n clock(): %d\n", blkid, clock());

            /*lk_results[blkid] =*/ lkWorkCuda(data[blkid]);
            
            log("work finished! Set data[%d].from_device to THREAD_FINISHED, lk_results[%d] is %d\n",
                blkid, blkid, lk_results[blkid]);
            
            if (tid == 0)
            {
                _vcast(trig[blkid].from_device) = THREAD_FINISHED;
            }
        }

        if (_vcast(trig[blkid].to_device) == THREAD_NOP)
            _vcast(trig[blkid].from_device) = THREAD_NOP;
    }
    log("I'm a thread and I'm out of the while\n");
}

/* These are for testing */

__device__ void sleep()
{
    clock_t start = clock();
    clock_t now;
    for (;;)
    {
        now = clock();
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= 15000000)
            break;
    }
}

__global__ void simple_kernel(volatile data_t *data, int *lk_results)
{
    clock_t clock_count = 200000;
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    
    while (clock_offset < clock_count)
        clock_offset = clock() - start_clock;
	lk_results[blockIdx.x] = 1;
}
