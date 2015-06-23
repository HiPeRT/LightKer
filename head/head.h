#ifndef head
#define head

#define L_MAX_LENGTH 1024

#define MAX_NUM_BLOCK 1024
#define MAX_SHMEM 16000

//#define DEBUG
#ifdef DEBUG
    #define log(...)  printf(__VA_ARGS__)
#else
    #define log(...)
#endif

#ifdef VERBOSE
#define verb(...) printf(__VA_ARGS__)
#else
#define verb(...)
#endif

// make an int volatile
#define _vcast(_arr) \
    *(volatile int *)&_arr

struct trig_t {
        int to_device;
        int from_device;
};

struct data_t {
    char str[256];
};

// for from_device:
#define THREAD_INIT 0
#define THREAD_FINISHED 1
#define THREAD_WORKING 2

// for to_device:
#define THREAD_NOP 4
#define THREAD_EXIT 8
#define THREAD_WORK 16

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

extern __device__ int work_nocuda(volatile data_t data);
extern __device__ int work_cuda(volatile data_t data);

#endif /*head*/
