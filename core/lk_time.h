/*
 *  LightKer - Light and flexible GPU persistent threads library
 *  Copyright (C) 2016  Paolo Burgio
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
