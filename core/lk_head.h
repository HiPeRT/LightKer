#ifndef __LK_HEAD_H__
#define __LK_HEAD_H__

/* Include CUDA (Actually not needed when compiling with nvcc) */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

/** A few utils */
#include <stdio.h>
#ifdef DEBUG
#   define log(...) \
    { \
      printf("[LK] [%s] ", __func__); \
      printf(__VA_ARGS__);\
    }
#else
#   define log(...)     ;
#endif

extern void lkDispose(dim3);
/* TODO do better, add dispose */
#define die(...) \
{ \
  printf("[LK] [%s] FATAL ERROR. ", __func__); \
  printf(__VA_ARGS__);\
  lkDispose(blknum); \
  exit(1);\
}

#ifdef VERBOSE
#   define verb(...)    printf(__VA_ARGS__)
#else
#   define verb(...)    ;
#endif

// Unsupported features
#define LK_WARN_NOT_SUPPORTED(_what) \
    log("[WARNING] " _what " is not supported yet.\n");
// To define always inline functions
#define ALWAYS_INLINE   __attribute__((always_inline))
// make an int volatile
#define _vcast(_arr)    * (volatile int *) &_arr
// For printing ptrs
#define _mycast_        (unsigned int) (uintptr_t)


/** Global definitions */
#define MAX_NUM_BLOCKS          64
#define MIN_NUM_BLOCKS          1
#define MAX_SHMEM               (16 * 1024)
#define MAX_BLOCK_DIM 192

/** LK exec support */
#define LK_EXEC_OK              0
#define LK_EXEC_APP_ERR         1
#define LK_EXEC_INT_ERR         2
#define LK_NOT_IMPLEMENTED      3

/** Mailbox flags */

// for from_device:
#define THREAD_INIT 0
#define THREAD_FINISHED 1
#define THREAD_WORKING 2

// for to_device:
#define THREAD_NOP 4
#define THREAD_EXIT 8
#define THREAD_WORK 16

static const char*
getFlagName(int flag)
{
  switch(flag)
  {
    /* to_device */
    case THREAD_INIT:
      return "THREAD_INIT";
    case THREAD_FINISHED:
      return "THREAD_FINISHED";
    case THREAD_WORKING:
      return "THREAD_WORKING";
      
    /* from_device */
    case THREAD_NOP:
      return "THREAD_NOP";
    case THREAD_EXIT:
      return "THREAD_EXIT";
    case THREAD_WORK:
      return "THREAD_WORK";
      
    default:
      return "Unknown";
  }
}

#endif /* __LK_HEAD_H__ */
