#include <getopt.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include "core/lk_time.h"
#include "core/lk_utils.h"

#include "work.cu"

#define APPNAME "EXAMPLE1"
// #define applog(_s, ...)                                             \
{                                                                   \
  printf("[" APPNAME "] [%s] " _s, __func__, ##__VA_ARGS__);       \
}
#define applog(_s, ...)    

typedef struct string_s
{
  char s[L_MAX_LENGTH];
} string_t;

__global__ void kernel(string_t str[], unsigned int num[])
{
  applog("[EXAMPLE] block %d will work on data '%s'\n", blockIdx.x, str);
  WORK((const char *) str[blockIdx.x].s, &num[blockIdx.x]);
  applog("[EXAMPLE] block %d returns %d\n", blockIdx.x, num[blockIdx.x]);  
}

void parse_cmdline(int argc, char **argv, dim3 * blknum, dim3 * blkdim);

int main(int argc, char **argv)
{
  dim3 numBlocks = 1, numThreads = 1;
  string_t *h_string, *d_string;
  unsigned int *h_num, *d_num;
  struct timespec spec_start, spec_stop, app_start, app_stop;
  long gettime_total, alloc_total, copyin_total, launch_total, wait1_total, wait2_total, dispose_total, copyout_total;
  
  gettime_total = alloc_total = copyin_total = launch_total = wait1_total = wait2_total = copyout_total = dispose_total = 0;
  
  parse_cmdline(argc, argv, &numBlocks, &numThreads);
  applog("numBlocks %d numThreads %d\n", numBlocks.x, numThreads.x);
  
#if 1
  /* Gettime overhead (1 round to warm cache) */
  GETTIME_TIC;
  GETTIME_TOC;
  GETTIME_TIC;
  GETTIME_TOC;
  gettime_total =  clock_getdiff_nsec(spec_start, spec_stop);
#endif
  
  GETTIME_TIC;
  /* Input (string) */
  applog("Alloc input data (Host)\n");
  checkCudaErrors(cudaHostAlloc((void **)&h_string, numBlocks.x * sizeof(string_t), cudaHostAllocDefault));
  applog("Alloc input data (Device)\n");
  checkCudaErrors(cudaMalloc((void **)&d_string, numBlocks.x * sizeof(string_t)));
  
  /* Output (integer) */
  applog("Alloc output data (Host)\n");
  checkCudaErrors(cudaHostAlloc((void **)&h_num, numBlocks.x * sizeof(unsigned int), cudaHostAllocDefault));
  applog("Alloc output data (Device)\n");
  checkCudaErrors(cudaMalloc((void **)&d_num, numBlocks.x * sizeof(unsigned int)));
  
  applog("Init data\n");
  /* Init app data */
  for(int i=0; i<numBlocks.x; i++)
  {
    applog("[EXAMPLE] Invoking INIT_DATA h_string @0x%x\n", _mycast_ &h_string[i]);
    INIT_DATA(h_string[i].s, i);
  }
  GETTIME_TOC;
  alloc_total =  clock_getdiff_nsec(spec_start, spec_stop);
  unsigned int num_loops = 10;
  
  for(int i=0; i<num_loops; i++)
  {
    GETTIME_TIC;
    applog("Copy data to device\n");
    /* Move data to device. We do the very same way as LK, block by block */
    for(int i=0; i<numBlocks.x; i++)
    {
      checkCudaErrors(cudaMemcpy(
                                &d_string[i],
                                &h_string[i],
                                sizeof(string_t),
                                cudaMemcpyHostToDevice));
    }
    GETTIME_TOC;
    copyin_total += clock_getdiff_nsec(spec_start, spec_stop);
      
    clock_gettime(CLOCK_MONOTONIC, &app_start);
    GETTIME_TIC;
    applog("Invoke CUDA kernel..\n");
    kernel<<<numBlocks, numThreads>>>(d_string, d_num);
    GETTIME_TOC;
    launch_total += clock_getdiff_nsec(spec_start, spec_stop);
    printf("Partial launch time %lu\n", clock_getdiff_nsec(spec_start, spec_stop));
  
    applog("Wait for CUDA kernel..\n");
    GETTIME_TIC;
    cudaDeviceSynchronize();
    GETTIME_TOC;
    wait1_total += clock_getdiff_nsec(spec_start, spec_stop);
    clock_gettime(CLOCK_MONOTONIC, &app_stop);
#if 0
    GETTIME_TIC;
    cudaDeviceSynchronize();
    GETTIME_TOC;
    wait2_total = clock_getdiff_nsec(spec_start, spec_stop);
#endif
    
    GETTIME_TIC;
    applog("Copy data from device\n");
    /* Move data to device. We do the very same way as LK, block by block */
    for(int i=0; i<numBlocks.x; i++)
      checkCudaErrors(cudaMemcpy(&h_num[i], &d_num[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
    GETTIME_TOC;
    copyout_total += clock_getdiff_nsec(spec_start, spec_stop);
    
    app_total += clock_getdiff_nsec(app_start, app_stop);
  } // work loop

  /* Check */
  for(int i=0; i<numBlocks.x; i++)
    CHECK_RESULTS((const char *) h_string[i].s, h_num[i], numThreads.x, i);

  GETTIME_TIC;
  applog("Dispose data\n");
  checkCudaErrors(cudaFree(d_num));
  checkCudaErrors(cudaFreeHost(h_string));
  checkCudaErrors(cudaFree(d_string));
  checkCudaErrors(cudaFreeHost(h_num));
  GETTIME_TOC;
  dispose_total = clock_getdiff_nsec(spec_start, spec_stop);
  
  checkCudaErrors(cudaGetLastError());
  
  GETTIME_LOG("[HEADER] Gettime;Alloc;Copyin;Launch;Wait;Empty wait;Copyout;Dispose;\n");
  GETTIME_LOG("[PROFILE] %lu;%lu;%lu;%lu;%lu;%lu;%lu;%lu;\n", gettime_total, alloc_total, copyin_total / num_loops, launch_total / num_loops, wait1_total / num_loops, wait2_total / num_loops, copyout_total / num_loops, dispose_total);
  GETTIME_LOG("[TOTAL] %lu\n", app_total / num_loops);
  GETTIME_LOG("[COPYIN] %lu\n", copyin_total / num_loops);
  
  return 0;
} // main


void parse_cmdline(int argc, char **argv, dim3 * blknum, dim3 * blkdim)
{
  static struct option long_options[] =
  {
    {"numblock",  required_argument,  0, 'b'},
    {"help",      no_argument,        0, 'h'},
    {"dimblock",  required_argument,  0, 't'},
  };

  int ret, opt_index = 0, o;

  while (1)
  {
    ret = getopt_long(argc, argv, "b:ht:", long_options, &opt_index);

    if (ret == -1)
      break;

    switch (ret)
    {
      case 'b':
        o = atoi(optarg);
//         if (o <= MAX_NUM_BLOCKS)
//         {
//           if (o >= MIN_NUM_BLOCKS)
            blknum->x = o;
//           else
//             blknum->x = MIN_NUM_BLOCKS;
//         }
        applog("Number of Blocks set to %d\n", blknum->x);
        break;
        
      
      case 'h':
        printf("Available options:\n\n");
        printf("-b x    Number of CUDA threads blocks (default %d)\n", blknum->x);
        printf("-t x    Number of CUDA threads per block (default %d)\n", blkdim->x);
        
        printf("\nReport bugs to: mailing-address\n");
        printf("pkg home page: <http://www.gnu.org/software/pkg/>\n");
        printf("General help using GNU software: <http://www.gnu.org/gethelp/>\n");
        exit(0);
        break;        
          
      case 't':
        o = atoi(optarg);
//         if (o >= 1 && o <= MAX_BLOCK_DIM)
          blkdim->x = o;
        applog("Number of Threads per Block set to %d\n", blkdim->x);
        break;
          
      default:
        printf("Unknown switch '%c'. Ignoring..\n", ret);
        break;
    } // switch
    
  } // while
  
} // parse_cmdlines