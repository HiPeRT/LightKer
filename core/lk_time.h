#ifndef __LK_TIME_H__
#define __LK_TIME_H__

/* Host-side timer */
#define USE_GETTIME

#ifdef USE_GETTIME
#  include <time.h>
#  define GETTIME_TIC                           clock_gettime(CLOCK_MONOTONIC, &spec_start)
#  define GETTIME_TOC                           clock_gettime(CLOCK_MONOTONIC, &spec_stop)
#  define clock_getdiff_nsec(_start, _stop)     ((_stop.tv_sec - _start.tv_sec)*1000000000 + (_stop.tv_nsec - _start.tv_nsec))
#  define GETTIME_LOG(_s, ...)                  printf("[LK] [TIME] " _s, ## __VA_ARGS__)
#else /* USE_GETTIME */
#  define GETTIME_TIC
#  define GETTIME_TOC
#  define clock_getdiff_nsec(start, stop)       (0)
#  define GETTIME_LOG(...)
#endif /* USE_GETTIME */

static long boot_total, alloc_total, appalloc_total, launch_total,
            wait_total, trigger_total, assign_total, retrieve_total, wait2_total,
            gettime_total, init_total, dispose_total, app_total;
            
extern long lkTriggerMultipleTime1, lkTriggerMultipleTime2, lkTriggerMultipleTime3;
extern long lkWaitTime1, lkWaitTime2, lkWaitTime3;
#endif  /* __LK_TIME_H__ */
