#include <getopt.h>

/* LK internal headers */
#include "lk_time.h"
#include "lk_head.h"

// (Probably) needed in app.cu
#include "lk_memutils.cu"

/* APP file */
#include "data.h"
#include "app.cu"

/* LK internals */
#include "lk_mailbox.cu"
#include "lk_device.cu"
#include "lk_host.cu"

/* Parse ARGC */
void parse_cmdline(int, char**, dim3*, dim3*, int*,  bool *);

/* Global vars */
dim3 blkdim = (1);
dim3 blknum = (1);
bool cudaMode = true;
int shmem = 0;

/* Main */
int main(int argc, char **argv)
{  
  long wait_total = 0, work_total = 0, assign_total = 0, retrieve_total = 0;
  /* Timers for profing */
  struct timespec spec_start, spec_stop;
  
  /* Todo ing a bit... */
  data_t *data = 0;
  res_t *res = 0;
  
  parse_cmdline(argc, argv, &blknum, &blkdim, &shmem, &cudaMode);

  /** INIT */
  lkInit(blknum, blkdim, shmem, cudaMode, &data, &res);
  
  /** WORK */
  int more = 1;
  unsigned int num_loops = 0;  
 
  log("Device is ready to go!\n");
  
  while (more)
  {
    /** SMALL OFFLOAD (WORK) **/
    GETTIME_TIC;
    more = lkSmallOffloadMultiple(data, blknum.x);
    verb("more is %d\n", more);
    GETTIME_TOC;
    assign_total += clock_getdiff_nsec(spec_start, spec_stop);
    
    /** TRIGGER (WORK) **/
    GETTIME_TIC;
    lkTriggerMultiple(blknum);
    GETTIME_TOC;
    work_total += clock_getdiff_nsec(spec_start, spec_stop);
    
    log("Waiting for %d SMs\n", blknum.x);
    for (unsigned int sm = 0 ; sm < blknum.x; sm++)
    {
      /* Profile lkWaitSM with the possibility to need to wait the GPU. */
      /** WAIT (WORK) **/
      GETTIME_TIC;
      lkWaitSM(sm, blknum);
      GETTIME_TOC;
      wait_total += clock_getdiff_nsec(spec_start, spec_stop);
    }
    log("Joined %d SMs. Fetching results\n", blknum.x);

    for (unsigned int sm = 0 ; sm < blknum.x; sm++)
    {
      /** RETRIEVE DATA (WORK) **/
      GETTIME_TIC;
      int result = lkRetrieveData(res, sm);
      GETTIME_TOC;
      verb("lk_h_results[%d] is %d, res is %d\n", sm, lk_h_results[sm], result);
      retrieve_total += clock_getdiff_nsec(spec_start, spec_stop);
    }

    num_loops++;
  } // work loop
    
  /** DISPOSE (DISPOSE) **/
  GETTIME_TIC;
  lkDispose(blknum);
  GETTIME_TOC;
  GETTIME_LOG("dispose %lu\n", clock_getdiff_nsec(spec_start, spec_stop));
  
  /* Print timing measurements */
  GETTIME_LOG("copy_data(work) %lu\n", assign_total);
  GETTIME_LOG("AVG copy_data(work) %lu\n", assign_total / num_loops);
  GETTIME_LOG("trigger(work) %lu\n", work_total);
  GETTIME_LOG("wait %lu\n", wait_total);
  GETTIME_LOG("retrieve data %lu\n", retrieve_total);
  
  //printf("num_loops %d total %lu avg %lu\n", num_loops, assign_total + wait_total, (assign_total + wait_total) / num_loops);
}

void parse_cmdline(int argc, char **argv, dim3 * blknum, dim3 * blkdim, int *shmem, bool *cudaMode)
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
  
} // parse_cmdlines
