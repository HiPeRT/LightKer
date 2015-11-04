#ifndef __TIMER_H__
#define __TIMER_H__
// 
// #include <cuda_runtime.h>
// 
// struct GpuTimer
// {
//   cudaEvent_t start;
//   cudaEvent_t stop;
// 
//   GpuTimer()
//   {
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//   }
// 
//   ~GpuTimer()
//   {
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//   }
// 
//   void Start()
//   {
//     cudaEventRecord(start, 0);
//   }
// 
//   void Stop()
//   {
//     cudaEventRecord(stop, 0);
//   }
// 
//   float Elapsed()
//   {
//     float elapsed;
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&elapsed, start, stop);
//     return elapsed;
//   }
// };
// 
// Take times inside a CUDA kernel
//#define USETIMERS
#ifdef USETIMERS
__device__ unsigned long long int cuda_timers[ 1024*1024 ];
 #define TIMER_TIC unsigned long long int tic; if ( threadIdx.x == 0 ) tic = clock64();
 #define TIMER_TOC(tid) unsigned long long int toc = clock64(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc > tic ) ? (toc - tic) : ( toc + (0xffffffffffffffff - tic) ) );
#else
 #define TIMER_TIC
 #define TIMER_TOC(tid)
#endif

// cpu timer
#define USEGETTIME
#ifdef USEGETTIME
#define GETTIME_TIC clock_gettime(CLOCK_MONOTONIC, &spec_start)
#define GETTIME_TOC clock_gettime(CLOCK_MONOTONIC, &spec_stop)
#define clock_getdiff_nsec(start, stop) ((stop.tv_sec - start.tv_sec)*1000000000 + (stop.tv_nsec - start.tv_nsec))
#else
#define GETTIME_TIC
#define GETTIME_TOC
#define clock_getdiff_nsec(start, stop) 0
#endif

#endif  /* __TIMER_H__ */
