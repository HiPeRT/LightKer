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


/* LK internal headers */
#include "lk_time.h"
#include "lk_globals.h"

// (Probably) needed in app.cu
#include "lk_memutils.cu"

/* APP file */
#include "data.h"
#include "app.cu"

/* LK internals */
#include "lk_mailbox.cu"
#include "lk_device.cu"
#include "lk_host.cu"

/* To debug */
#ifndef USE_APP_MAIN

/* Main */
int main(int argc, char **argv)
{  
  /* Global vars */
  dim3 blkdim = (1);
  dim3 blknum = (1);
  bool cudaMode = true;
  int shmem = 0;
  
  /* Timers for profing */
  struct timespec spec_start, spec_stop, app_start, app_stop;
  
  /* GETTIME_INIT */
  boot_total = alloc_total = appalloc_total = launch_total =
    wait_total = trigger_total = assign_total = retrieve_total =
    wait2_total = gettime_total = init_total = app_total = 0;
  
  /* Todo ing a bit... */
  data_t *data = 0;
  res_t *res = 0;
  
  lkParseCmdLine(argc, argv, &blknum, &blkdim, &shmem, &cudaMode);
  
  /* Used in profiling (example1 app) */
#if defined(L_MAX_LENGTH) && defined(WORK_TIME)
  printf("L_MAX_LENGTH %d WORK_TIME %d\n", L_MAX_LENGTH, WORK_TIME);
#endif
  
  /* Gettime overhead (1st round to warm cache) */
#if 0
  GETTIME_TIC;
  GETTIME_TOC;
  GETTIME_TIC;
  GETTIME_TOC;
  gettime_total =  clock_getdiff_nsec(spec_start, spec_stop);
#endif
  
  /** INIT */
//   log("before lkInit, data 0x%x\n", _mycast_ data);
  GETTIME_TIC;
  lkInit(blknum.x, blkdim.x, shmem, cudaMode, 0x0, 0x0, &data, &res);
  GETTIME_TOC;
  init_total =  clock_getdiff_nsec(spec_start, spec_stop);
//   log("after lkInit, data 0x%x\n", _mycast_ data);
  
  /** WORK */
  int more = 1;
  unsigned int num_loops = 0;  
 
  log("Device is ready to go!\n");
  
  lkWaitTime1 = lkWaitTime2 = lkWaitTime3 = 0;
  while (more)
  {
    /** SMALL OFFLOAD (WORK) **/
    GETTIME_TIC;
    more = lkSmallOffloadMultiple(0x0, data, blknum.x);
    GETTIME_TOC;
    assign_total += clock_getdiff_nsec(spec_start, spec_stop);
    
    clock_gettime(CLOCK_MONOTONIC, &app_start);
    /** TRIGGER (WORK) **/
    GETTIME_TIC;
    lkTriggerMultiple();
    GETTIME_TOC;
    trigger_total += clock_getdiff_nsec(spec_start, spec_stop);
//     GETTIME_LOG("Partial launch time %lu\n", clock_getdiff_nsec(spec_start, spec_stop));
//     GETTIME_LOG("lkTriggerMultipleTime1 %lu\n", lkTriggerMultipleTime1);
//     GETTIME_LOG("lkTriggerMultipleTime2 %lu\n", lkTriggerMultipleTime2);
//     GETTIME_LOG("lkTriggerMultipleTime3 %lu\n", lkTriggerMultipleTime3);
    
    /** WAIT (WORK) **/
    GETTIME_TIC;
    lkWaitMultiple();
    GETTIME_TOC;
    wait_total += clock_getdiff_nsec(spec_start, spec_stop);
    clock_gettime(CLOCK_MONOTONIC, &app_stop);
//     printf("Joined %d SMs. Fetching results\n", blknum.x);
    
    /** RETRIEVE DATA (WORK) **/
    GETTIME_TIC;
    lkRetrieveDataMultiple(0x0, res, blknum.x);
    GETTIME_TOC;
    retrieve_total += clock_getdiff_nsec(spec_start, spec_stop);
    
    app_total += clock_getdiff_nsec(app_start, app_stop);
    
    /** 'Empty' wait */
#if 0
    lkProfiling = 1;
    GETTIME_TIC;
    lkWaitMultiple();
    GETTIME_TOC;
    wait2_total += clock_getdiff_nsec(spec_start, spec_stop);
    lkProfiling = 0;
#endif

    struct timespec ts;
    ts.tv_sec  = 0;
    ts.tv_nsec = 6000L;
    if(nanosleep(&ts, NULL) < 0)
    {
      printf("Error in sleep!\n");
    }
      
    num_loops++;
  } // work loop
  
    
  /** DISPOSE (DISPOSE) **/
  GETTIME_TIC;
  lkDispose();
  GETTIME_TOC;
  dispose_total = clock_getdiff_nsec(spec_start, spec_stop);
  
  /* Print timing measurements */
#if 0
  GETTIME_LOG("Gettime %lu\n", gettime_total);
  GETTIME_LOG("Init Tot %lu\n", init_total);
  GETTIME_LOG("These (AVG) numbers are performed on %d measurements\n", num_loops);
//   GETTIME_LOG("TOT copy_data(work) %lu\n", assign_total);
  GETTIME_LOG("AVG copy_data(work) %lu\n", assign_total / num_loops);
  GETTIME_LOG("AVG trigger(work) %lu\n", trigger_total / num_loops);
//   GETTIME_LOG("AVG wait %lu\n", wait_total / (blknum.x * num_loops));
//   GETTIME_LOG("AVG empty wait %lu\n", wait2_total / (blknum.x * num_loops));
  GETTIME_LOG("AVG wait %lu\n", wait_total / num_loops);
  GETTIME_LOG("AVG empty wait %lu\n", wait2_total / num_loops);
  GETTIME_LOG("AVG retrieve data %lu\n", retrieve_total / num_loops);
//   GETTIME_LOG("AVG wait #1 %lu\n", lkWaitTime1 / num_loops);
//   GETTIME_LOG("AVG wait #2 %lu\n", lkWaitTime2 / num_loops);
//   GETTIME_LOG("AVG wait #3 %lu\n", lkWaitTime3 / num_loops);
  GETTIME_LOG("dispose %lu\n", dispose_total);
#endif
  
  GETTIME_LOG("[HEADER] Gettime;Boot (Init);Alloc (Init);App alloc (Init);Launch (Init);Init (Tot);Copy data (Work);Trigger (Work);Wait (Work);Empty wait (Work);Retrieve data (Work);Dispose (Tot);\n");
  GETTIME_LOG("[PROFILE] %lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu;\n", gettime_total, boot_total, alloc_total, appalloc_total, launch_total, init_total, assign_total/num_loops, trigger_total/num_loops, wait_total/num_loops, wait2_total/num_loops, retrieve_total/num_loops, dispose_total);  
  GETTIME_LOG("[TOTAL] %lu\n", app_total / num_loops);
  GETTIME_LOG("[COPYIN] %lu\n", assign_total / num_loops);
  
  //printf("num_loops %d total %lu avg %lu\n", num_loops, assign_total + wait_total, (assign_total + wait_total) / num_loops);
}
#endif


#include <getopt.h>
void lkParseCmdLine(int argc, char **argv, dim3 * blknum, dim3 * blkdim, int *shmem, bool *cudaMode)
{
  static struct option long_options[] =
  {
    {"numblock",  required_argument,  0, 'b'},
    {"cuda-mode", no_argument,        0, 'c'},
    {"help",      no_argument,        0, 'h'},
    {"shmem",     required_argument,  0, 's'},
    {"dimblock",  required_argument,  0, 't'},
  };

  int ret, opt_index = 0, o;

  while (1)
  {
    ret = getopt_long(argc, argv, "b:cf:hs:t:", long_options, &opt_index);

    if (ret == -1)
      break;

    switch (ret)
    {
      case 'b':
        o = atoi(optarg);
        if (o <= MAX_NUM_BLOCKS)
        {
          if (o >= MIN_NUM_BLOCKS)
            blknum->x = o;
          else
            blknum->x = MIN_NUM_BLOCKS;
        }
        log("Number of Blocks set to %d\n", blknum->x);
        break;
        
      case 'c':
        *cudaMode = !*cudaMode;
        log("Setting CUDA mode to '%s'\n", cudaMode ? "on" : "off");
        break;
      
      case 'h':
        printf("Available options:\n\n");
        printf("-b x    Number of CUDA threads blocks (default %d)\n", blknum->x);
        printf("-c      Toggles CUDA mode (default %s)\n", cudaMode ? "yes" : "no");
        printf("-s x    Device per-SM dynamic Shared Memory size in bytes (default %d)\n", *shmem);
        printf("-t x    Number of CUDA threads per block (default %d)\n", blkdim->x);
        
        printf("\nReport bugs to: mailing-address\n");
        printf("pkg home page: <http://www.gnu.org/software/pkg/>\n");
        printf("General help using GNU software: <http://www.gnu.org/gethelp/>\n");
        exit(0);
        break;        
          
      case 's':
        o = atoi(optarg);
        if (o > 0)
          *shmem = o;
        log("SHMEM dim set to %d\n", *shmem);
        break;
        
      case 't':
        o = atoi(optarg);
        if (o >= 1 && o <= MAX_BLOCK_DIM)
          blkdim->x = o;
        log("Number of Threads per Block set to %d\n", blkdim->x);
        break;
          
      default:
        printf("Unknown switch '%c'. Ignoring..\n", ret);
        break;
    } // switch
    
  } // while
  
} // lkParseCmdLines