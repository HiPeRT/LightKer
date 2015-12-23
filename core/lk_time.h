#ifndef __LK_TIME_H__
#define __LK_TIME_H__

/* CPU-side timer */
#define USE_GETTIME

#ifdef USE_GETTIME
#  define GETTIME_TIC                           clock_gettime(CLOCK_MONOTONIC, &spec_start)
#  define GETTIME_TOC                           clock_gettime(CLOCK_MONOTONIC, &spec_stop)
#  define clock_getdiff_nsec(start, stop)       ((stop.tv_sec - start.tv_sec)*1000000000 + (stop.tv_nsec - start.tv_nsec))
#  define GETTIME_LOG(_s, ...)                  printf("[LK] [TSTAMP] " _s, __VA_ARGS__)
#else
#  define GETTIME_TIC
#  define GETTIME_TOC
#  define clock_getdiff_nsec(start, stop)       (0)
#  define GETTIME_LOG(...)
#endif

#endif  /* __LK_TIME_H__ */
