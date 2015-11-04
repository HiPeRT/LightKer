#include <stdio.h>

__device__ int work_nocuda(volatile data_t data);
__device__ int work_cuda(volatile data_t data);

__device__ void sleep()
{
    clock_t start = clock();
    clock_t now;
    for (;;) {
        now = clock();
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= 15000000) {
            break;
        }
    }
}

/* Main kernel function, for writing "non-cuda" work function.
   Busy waits on the GPU until the CPU notifies new work, then
   it acknowledges the CPU and starts the real work. When finished
   it acknowledges the CPU through the trigger "from_device"
 */
__global__ void uniform_polling(volatile trig_t *trig, volatile data_t *data, int *results)
{
	int blkid = blockIdx.x;
	int tid = threadIdx.x;

	if (tid == 0) {
		while (1) {
			volatile int to_device = _vcast(trig[blkid].to_device);

			//dispose
			if (to_device == THREAD_EXIT)
				break;

			if (to_device == THREAD_WORK && _vcast(trig[blkid].from_device) != THREAD_FINISHED) {
				_vcast(trig[blkid].from_device) = THREAD_WORKING;
				log("Hi, I'm block %d and I received sth to do!\n clock(): %d\n", blkid, clock());
				results[blkid] = work_nocuda(data[blkid]);
				log("Work finished! Set data[%d].from_device to THREAD_FINISHED, results[%d] is %d\n",
				     blkid, blkid, results[blkid]);
				_vcast(trig[blkid].from_device) = THREAD_FINISHED;
			}
		}
        	log("I'm a thread and I'm out of the while\n");
	}
}

/* Main kernel function, for writing cuda aware work function.
   Busy waits on the GPU until the CPU notifies new work, then
   it acknowledges the CPU and starts the real work. When finished
   it acknowledges the CPU through the trigger "from_device"
 */
__global__ void uniform_polling_cuda(volatile trig_t *trig, volatile data_t *data, int *results)
{
	int blkid = blockIdx.x;
	int tid = threadIdx.x;
	clock_t clock_count = 60000;
	clock_t start_clock = clock();
	clock_t clock_offset = 0;

        if (tid == 0)
		_vcast(trig[blkid].from_device) = THREAD_NOP;

        while (1) {
		if (_vcast(trig[blkid].to_device) == THREAD_EXIT)
			break;

		if (_vcast(trig[blkid].to_device) == THREAD_WORK && _vcast(trig[blkid].from_device) != THREAD_FINISHED) {
			if (tid == 0) {
				_vcast(trig[blkid].from_device) = THREAD_WORKING;
			}
			log("Hi, I'm block %d and I received sth to do!\n clock(): %d\n", blkid, clock());
			while (clock_offset < clock_count) {
				clock_offset = clock() - start_clock;
			}
			//results[blkid] = work_cuda(data[blkid]);
			log("work finished! Set data[%d].from_device to THREAD_FINISHED, results[%d] is %d\n",
				blkid, blkid, results[blkid]);
			if (tid == 0) {
				_vcast(trig[blkid].from_device) = THREAD_FINISHED;
			}
		}

		if (_vcast(trig[blkid].to_device) == THREAD_NOP)
			_vcast(trig[blkid].from_device) = THREAD_NOP;
	}
	log("I'm a thread and I'm out of the while\n");
}

__global__ void simple_kernel(volatile data_t *data, int *results) {
    clock_t clock_count = 200000;
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count) {
        clock_offset = clock() - start_clock;
    }
	results[blockIdx.x] = 1;
}
