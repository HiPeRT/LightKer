#include <getopt.h>

/* LK internal headers */
#include "lk_time.h"
#include "lk_head.h"

/* LK internals */
#include "lk_mailbox.cu"
#include "lk_host.cu"
#include "lk_device.cu"

// (Probably) needed in app.cu
#include "lk_memutils.cu"

/* APP file */
#include "app.cu"

/* Launch in CUDA mode? */
bool cudaMode = true;
dim3 blknum = 1;
dim3 blkdim = (1);
int shmem = 0;
/* Number of actual CUDA blocks => SMs */
int wg;

cudaStream_t backbone_stream, stream_kernel;
void parse_cmdline(int, char**, dim3*, dim3*, int*);

/* Main */
int main(int argc, char **argv)
{  
  char s[10000];
  long wait_total = 0, work_total = 0, assign_total = 0, retrieve_total = 0;
  
  parse_cmdline(argc, argv, &blknum, &blkdim, &shmem);
  
  verb("Warning: with VERBOSE flag on, time measures will be unreliable\n");

  /** BOOT (INIT) **/
  
  cudaDeviceReset();
  
  log("LIGHTKERNEL START\n");

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  /* Get device properties */
  int device;
  for (device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    log("[boot] Device canMapHostMemory: %s.\n", deviceProp.canMapHostMemory ? "yes" : "no");
    log("[boot] Device %d has async engine count %d.\n", device, deviceProp.asyncEngineCount);
  }
  
  /* Create kernel and backbone (mailbox) streams */
  checkCudaErrors(cudaStreamCreate(&stream_kernel));
  checkCudaErrors(cudaStreamCreate(&backbone_stream));
  
  log("[boot] Number of Blocks: %d number of threads per block: %d, shared memory dim: %d\n", blknum.x, blkdim.x, shmem);

  wg = blknum.x;

  struct timespec spec_start, spec_stop;
  
  data_t *data;
  int *lk_results;

  /** ALLOC (INIT) **/
  GETTIME_TIC;
  
  log("[alloc] sizeof(mailbox_t) is %d sizeof(mailbox_elem_t) is %d\n",
      sizeof(mailbox_t), sizeof(mailbox_elem_t));
  
  /* Mailboxes */
  
  if(lkMailboxInit())
    die("Mailbox initialization failed\n");
  
  /* Results from Device */
  checkCudaErrors(cudaHostAlloc((void **)&lk_results, wg * sizeof(int), cudaHostAllocDefault));

  /* Call application-specific initialization of data
   * 'Big offload' is performed here */
  log("Invoke app-specific data initialization data ptr @0x%x it is 0x%x\n", _mycast_ &data, _mycast_ data);
  lkInitAppData(&data, wg);
  
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("alloc(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
  
  /** LAUNCH (INIT) **/
  GETTIME_TIC;
  lkLaunch(cudaMode ? lkUniformPollingCuda : lkUniformPollingNoCuda,
            data, lk_results,
            blkdim, blknum, shmem);
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("launch(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

  
  /** WORK */
  int sm = 0;
  int more = 1;
  int num_loops = 0;  
 
  log("Device is ready to go!\n");
  
  while (more)
  {
    /** SMALL OFFLOAD (WORK) **/
    GETTIME_TIC;
    more = lkSmallOffloadMultiple(data, blknum);
    GETTIME_TOC;
    assign_total += clock_getdiff_nsec(spec_start, spec_stop);
    
    /** TRIGGER (WORK) **/
    GETTIME_TIC;
    lkTriggerMultiple(blknum);
    GETTIME_TOC;
    work_total += clock_getdiff_nsec(spec_start, spec_stop);
    
    log("Waiting for %d SMs\n", wg);
    for (int sm = 0 ; sm < wg/*blknum.x*/; sm++)
    {
//       log("Waiting for SM #%d out of %d\n", sm, blknum.x);
      /* Profile lkWaitSM with the possibility to need to wait the GPU. */
      /** WAIT (WORK) **/
      GETTIME_TIC;
      lkWaitSM(sm, blknum);
      GETTIME_TOC;
      wait_total += clock_getdiff_nsec(spec_start, spec_stop);
    }

    for (int sm = 0 ; sm < wg ; sm++)
    {
      /** RETRIEVE DATA (WORK) **/
      GETTIME_TIC;
      int res = lkRetrieveData(data, sm);
      GETTIME_TOC;
      log("lk_results[%d] is %d, res is %d\n", sm, lk_results[sm], res);
      retrieve_total += clock_getdiff_nsec(spec_start, spec_stop);
    }

    num_loops++;
  } // work loop
  
  /** DISPOSE (DISPOSE) **/
  GETTIME_TIC;
  lkDispose(blknum);
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("dispose %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
  
//   /* Print timing measurements */
//   sprintf(s, "%s t%ld", s, assign_total);
//   verb("copy_data(work) %lld\n", assign_total);
//   sprintf(s, "%s %ld", s, assign_total / num_loops);
//   verb("AVG copy_data(work) %lld\n", assign_total / num_loops);
//   sprintf(s, "%s %ld", s, work_total);
//   verb("trigger(work) %lld\n", work_total);
//   sprintf(s, "%s w%ld", s, wait_total);
//   verb("wait %lld\n", wait_total);
//   sprintf(s, "%s %ld", s, retrieve_total);
//   verb("retrieve data %lld\n", retrieve_total);

  //printf("num_loops %d total %lu avg %lu\n", num_loops, assign_total + wait_total, (assign_total + wait_total) / num_loops);
}

void parse_cmdline(int argc, char **argv, dim3 * blknum, dim3 * blkdim, int *shmem)
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
        cudaMode = !cudaMode;
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
