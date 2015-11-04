// #include <stdio.h>
#include <assert.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <unistd.h>
// #include <unistd.h>
// #include <time.h>
// #include <math.h>
// #include <inttypes.h>
// #include <getopt.h>
// #include <stdlib.h>

/* LK internal headers */
#include "head.h"
#include "utils.h"
/* To treat APP specific data_t type */
#include "data.h"
// #include "app.cu"
// 
// #include "lk_host.h"

/* Low-level functionalities */

/* Initialize the triggers and start the kernel.
 * formerly known as 'init'
 */
void lkLaunch(void (*kernel) (volatile trig_t *, volatile data_t *, int *),
			  trig_t *trig, trig_t *d_trig, data_t *data, int *results,
			  dim3 blkdim, dim3 blknum, int shmem,
			  cudaStream_t *stream_kernel, cudaStream_t *backbone_stream)
{
	int wg = blknum.x;

	// trigger initialization
	for (int i = 0; i < wg; i++) {
		_vcast(trig[i].to_device) = THREAD_NOP;
	}
	checkCudaErrors(cudaMemcpyAsync(d_trig, trig, sizeof(trig_t) * wg,
			cudaMemcpyHostToDevice, *backbone_stream));

	kernel <<< blknum, blkdim, shmem, *stream_kernel >>> (d_trig, data, results);
}

/* Order the given sm to start working.
 * Formerly known as 'work'
 */
void lkTriggerSM(trig_t *trig, trig_t *d_trig, int sm, dim3 blknum, cudaStream_t *backbone_stream)
{
    log("SM %d blknum %d\n", sm, blknum.x);
	assert(sm <= blknum.x);
	assert(_vcast(trig[sm].from_device) != THREAD_WORK);

	_vcast(trig[sm].to_device) = THREAD_WORK;
	checkCudaErrors(cudaMemcpyAsync(&d_trig[sm], &trig[sm], sizeof(trig_t), cudaMemcpyHostToDevice, *backbone_stream));

	log("sm %d from_device %s to_device %s\n", sm,
        getFlagName(_vcast(trig[sm].from_device)), getFlagName(_vcast(trig[sm].to_device)));
}


/* Order the given sm to start working.
 */
void lkTriggerMultiple(trig_t *trig, trig_t *d_trig, dim3 blknum, cudaStream_t *backbone_stream)
{
    log("blknum=%d\n", blknum.x);

    for(int i=0; i<blknum.x; i++)
    {
      log("Triggering SM #%d\n", i);
      _vcast(trig[i].to_device) = THREAD_WORK;
    }
    
    log("Transfering %d mailboxes to Device..\n", blknum.x);
    checkCudaErrors(cudaMemcpyAsync(&d_trig[0], &trig[0], sizeof(trig_t) * blknum.x, cudaMemcpyHostToDevice, *backbone_stream));
    
//     log("sm %d from_device %s to_device %s\n", sm,
//         getFlagName(_vcast(trig[sm].from_device)), getFlagName(_vcast(trig[sm].to_device)));
}

/* Busy wait until the given sm is working. Trigger to_device is restored to state "THREAD_NOP".
 * Formerly known as 'sm_wait'
 */
void lkWaitSM(trig_t *trig, trig_t *d_trig, int sm, dim3 blknum, cudaStream_t *backbone_stream)
{
    log("SM #%d\n", sm);
    
    if(_vcast(trig[sm].to_device) != THREAD_WORK)
    {
      printf("SM #%d was not triggered! %d", sm, _vcast(trig[sm].to_device));
      return;
    }
    
    log("waiting for SM #%d to start working\n",sm);

	do {

//         checkCudaErrors(cudaMemcpyAsync(trig, d_trig, sizeof(trig_t)*blknum.x,
//               cudaMemcpyDeviceToHost, *backbone_stream));
		checkCudaErrors(cudaMemcpyAsync(&trig[sm], &d_trig[sm], sizeof(trig_t), cudaMemcpyDeviceToHost, *backbone_stream));
//         log("waiting for SM #%d to start working (to_device flag: %s, from_device flag: %s)\n",
//             sm, getFlagName(_vcast(trig[sm].to_device)), getFlagName(_vcast(trig[sm].from_device)));
        
	} while (_vcast(trig[sm].from_device) != THREAD_WORKING && _vcast(trig[sm].from_device) != THREAD_FINISHED);

    log("SM #%d is working: waiting for it to end\n", sm);
	do {

		checkCudaErrors(cudaMemcpyAsync(&trig[sm], &d_trig[sm], sizeof(trig_t), cudaMemcpyDeviceToHost, *backbone_stream));
// 		log("waiting for SM #%d to end working (to_device flag: %s, from_device flag: %s)\n",
//             sm, getFlagName(_vcast(trig[sm].to_device)), getFlagName(_vcast(trig[sm].from_device)));
	} while (_vcast(trig[sm].from_device) == THREAD_WORKING);
    
    log("SM #%d ended its work\n", sm);
}


/* Order to the kernel to exit and wait for its termination.
 * Formerly known as 'dispose'
 */
void lkDispose(trig_t *trig, trig_t *d_trig, dim3 blknum, cudaStream_t *backbone_stream)
{
    int wg = blknum.x;
    log("Stop 'em!\n");

    for (int i = 0; i < wg; i++)
        _vcast(trig[i].to_device) = THREAD_EXIT;
    
    checkCudaErrors(cudaMemcpyAsync(d_trig, trig, sizeof(trig_t) * wg, cudaMemcpyHostToDevice, *backbone_stream));
    
    cudaStreamSynchronize(*backbone_stream);

    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_trig));
    checkCudaErrors(cudaFreeHost(trig));
    
    checkCudaErrors(cudaDeviceReset());
    log("Done.\n");
}

/* User-level API to embed LK into application */

void LKInit()
{
#if 0
  dim3 blknum = 1;
  dim3 blkdim = (1);
  int shmem = 0;
  char s[10000];
  long wait_total = 0, work_total = 0, assign_total = 0, retrieve_total = 0;
  
  verb("Warning: with VERBOSE flag on, time measures will be unreliable\n");

  /** BOOT (INIT) **/
  
  cudaDeviceReset();
  
  log("LIGHTKERNEL START\n");

  int deviceCount;
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

  cudaStream_t stream_kernel, backbone_stream;
  checkCudaErrors(cudaStreamCreate(&stream_kernel));
  checkCudaErrors(cudaStreamCreate(&backbone_stream));
  
  log("[boot] Number of Blocks: %d number of threads per block: %d, shared memory dim: %d\n", blknum.x, blkdim.x, shmem);

  int wg = blknum.x;

  struct timespec spec_start, spec_stop;
  
  trig_t *trig, *d_trig;
  data_t *data;
  int *lk_results;

  /** ALLOC (INIT) **/
  /* cudaHostAlloc: shared between host and GPU */
  checkCudaErrors(cudaHostAlloc((void **)&trig, wg * sizeof(trig_t), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc((void **)&d_trig, wg * sizeof(trig_t)));

  /* Call application-specific initialization of data
   * 'Big offload' is performed here */
  lkInitAppData(&data, wg);
  checkCudaErrors(cudaHostAlloc((void **)&lk_results, wg * sizeof(int), cudaHostAllocDefault));
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("alloc(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

  /** LAUNCH (INIT) **/
  GETTIME_TIC;
  if(cudaMode)
    lkLaunch(lkUniformPollingCuda, trig, d_trig, data, lk_results, blkdim, blknum, shmem, &stream_kernel, &backbone_stream);
  else
    lkLaunch(lkUniformPollingNoCuda, trig, d_trig, data, lk_results, blkdim, blknum, shmem, &stream_kernel, &backbone_stream);
  
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("launch(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
  //print_trigger("after init", trig);
#endif
}
