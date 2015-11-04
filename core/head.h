#ifndef __HEAD_H__
#define __HEAD_H__

#define MAX_NUM_BLOCK 1024
#define MAX_SHMEM 16000

#include <stdio.h>
#ifdef DEBUG
#define MYFUN __func__
#   define log(...) \
    { \
      printf("[LK] [%s] ", __func__); \
      printf(__VA_ARGS__);\
    }
#else
#   define log(...)     ;
#endif

#ifdef VERBOSE
#   define verb(...)    printf(__VA_ARGS__)
#else
#   define verb(...)    ;
#endif

// make an int volatile
#define _vcast(_arr) * (volatile int *) &_arr

/*** Mailbox flags */

struct trig_t {
  int to_device;
  int from_device;
};

// for from_device:
#define THREAD_INIT 0
#define THREAD_FINISHED 1
#define THREAD_WORKING 2

// for to_device:
#define THREAD_NOP 4
#define THREAD_EXIT 8
#define THREAD_WORK 16

static inline const char*
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

static inline void
print_trigger(const char *fun, trig_t * trig)
{
    log("[%s] to_device %s (%d), from_device %s (d)\n", fun, getFlagName(_vcast(trig[0].to_device)), _vcast(trig[0].to_device),
    getFlagName(_vcast(trig[0].from_device)), _vcast(trig[0].from_device));
}

/** LK exec support */
#define LK_EXEC_OK          0
#define LK_EXEC_APP_ERR     1
#define LK_EXEC_INT_ERR     2
#define LK_NOT_IMPLEMENTED  3

#endif /* __HEAD_H__ */
