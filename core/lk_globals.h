#ifndef __LK_HEAD_H__
#define __LK_HEAD_H__

/* Include CUDA (Actually not needed when compiling with nvcc) */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

/** A few utils */
#include <stdio.h>
#ifdef LK_DEBUG
#   define log(_s, ...)                                                 \
    {                                                                   \
      printf("[LK] [%s] " _s, __func__, ##__VA_ARGS__);                 \
    }
#else /* LK_DEBUG */
#   define log(...)                                                     \
    {                                                                   \
    }
#endif /* LK_DEBUG */

#ifdef LK_VERBOSE
#   define verb(...)            printf(__VA_ARGS__)
#else
#   define verb(...)            ;
#endif

void lkDispose();
#define die(_s, ...)                                                    \
{                                                                       \
  printf("[LK] [%s] FATAL ERROR. " _s, __func__, ##__VA_ARGS__);        \
  lkDispose();                                                          \
  exit(1);                                                              \
}

#define warning(_s, ...) \
  printf("[LK] Warning. " _s, ##__VA_ARGS__);        \

// Unsupported features
#define LK_WARN_NOT_SUPPORTED() \
    log("[WARNING] %s is not supported yet.\n", __func__ );
// To define always inline functions
#define ALWAYS_INLINE           __attribute__((always_inline))

/** Global definitions */
#define MAX_NUM_BLOCKS          16
#define MIN_NUM_BLOCKS          1
#define MAX_SHMEM               (16 * 1024)
#define MAX_BLOCK_DIM           192

typedef unsigned int lk_result_t;
/** LK exec support */
/* This must be 0 */
#define LK_EXEC_OK              0
#define LK_EXEC_APP_ERR         1
#define LK_EXEC_INT_ERR         2
#define LK_NOT_IMPLEMENTED      3

/** Mailbox flags */

// for from_device:
#define THREAD_INIT             0
#define THREAD_FINISHED         1
#define THREAD_WORKING          2

// for to_device:
#define THREAD_NOP              4
#define THREAD_EXIT             8
#define THREAD_WORK             16

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

/** CUDA streams */
extern cudaStream_t kernel_stream, backbone_stream;

/* Puts some functions (lkWaitSM) in a "safe" mode just to profile them */
extern unsigned int lkProfiling;

#endif /* __LK_HEAD_H__ */
