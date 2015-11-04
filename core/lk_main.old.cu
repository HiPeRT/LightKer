extern void parse_cmdline(int, char**, dim3*, dim3*, int*);

// To debug isKindOf: remove
extern int TEST_IDX;
extern int **g_results;

/* Main */

/* Launch in CUDA mode? */
bool cudaMode = true;
int main(int argc, char **argv)
{
  dim3 blknum = 1;
  dim3 blkdim = (1);
  int shmem = 0;
  FILE *file = NULL;
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

  cudaStream_t stream_kernel, backbone_stream;
  checkCudaErrors(cudaStreamCreate(&stream_kernel));
  checkCudaErrors(cudaStreamCreate(&backbone_stream));
  
  log("[boot] Number of Blocks: %d number of threads per block: %d, shared memory dim: %d\n", blknum.x, blkdim.x, shmem);

  if (filename)
      file = fopen(filename, "a");

  int wg = blknum.x;

  struct timespec spec_start, spec_stop;
  
  trig_t *trig, *d_trig;
  data_t *data;
  int *lk_results;

#if 0
  /** OVERHEAD **/
  GETTIME_TIC;
  GETTIME_TOC;
  GETTIME_TIC;
  GETTIME_TOC;
  sprintf(s, "%ld", clock_getdiff_nsec(spec_start, spec_stop));
  verb("overhead %lld\n", clock_getdiff_nsec(spec_start, spec_stop));


  int *arr;
  GETTIME_TIC;
  checkCudaErrors(cudaMalloc(&arr, 1024 * sizeof(int)));
  GETTIME_TOC;
  long long int t1_boot = clock_getdiff_nsec(spec_start, spec_stop);
  log("boot(init): first %lld\n", t1_boot);

  GETTIME_TIC;
  checkCudaErrors(cudaMalloc(&arr, 1024 * sizeof(int)));
  GETTIME_TOC;
  long long int t_boot = t1_boot - clock_getdiff_nsec(spec_start, spec_stop);
  log("boot(init): second %ld\n", clock_getdiff_nsec(spec_start, spec_stop));
  sprintf(s, "%s %lld", s, t_boot);
  verb("boot(init) %lld\n", t_boot);
#endif

  /** ALLOC (INIT) **/
  GETTIME_TIC;
  /* cudaHostAlloc: shared between host and GPU */
  checkCudaErrors(cudaHostAlloc((void **)&trig, wg * sizeof(trig_t), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc((void **)&d_trig, wg * sizeof(trig_t)));

  /* Call application-specific initialization of data
   * 'Big offload' is performed here */
  lkInitAppData(&data, wg);
  checkCudaErrors(cudaHostAlloc((void **)&lk_results, wg * sizeof(int), cudaHostAllocDefault));
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("alloc(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

  /** LAUNCH (INIT) **/
  GETTIME_TIC;
  if(cudaMode)
    lkLaunch(lkUniformPollingCuda, trig, d_trig, data, lk_results, blkdim, blknum, shmem, &stream_kernel, &backbone_stream);
  else
    lkLaunch(lkUniformPollingNoCuda, trig, d_trig, data, lk_results, blkdim, blknum, shmem, &stream_kernel, &backbone_stream);
  
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("launch(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
  //print_trigger("after init", trig);

  /** WORK */
  int sm = 0;
  int more = 1;
  int num_loops = 0;
  

  while (more)
  {
    printf("-----TEST_IDX is %d------\n", TEST_IDX);
    
#if 1
    /** SMALL OFFLOAD (WORK) **/
    GETTIME_TIC;
    more = lkSmallOffloadMultiple(data, blknum, &backbone_stream);
    GETTIME_TOC;
    assign_total += clock_getdiff_nsec(spec_start, spec_stop);
    
    //printf("assign data done\n");
    
    /** TRIGGER (WORK) **/
    GETTIME_TIC;
    lkTriggerMultiple(trig, d_trig, blknum, &backbone_stream);
    GETTIME_TOC;
    work_total += clock_getdiff_nsec(spec_start, spec_stop);
#endif

//              for (sm = 0 ; sm < blknum.x ; sm++) {
// #if 0
//             /** COPY_DATA (WORK) **/
//             GETTIME_TIC;
//             lkSmallOffload(data, sm, &backbone_stream);
//             GETTIME_TOC;
//             assign_total += clock_getdiff_nsec(spec_start, spec_stop);
//             
//                      /** TRIGGER (WORK) **/
//                      GETTIME_TIC;
//                      lkTriggerSM(trig, d_trig, sm, blknum, &backbone_stream);
//                      GETTIME_TOC;
//                      work_total += clock_getdiff_nsec(spec_start, spec_stop);
// #endif
//              }
    //TEST_IDX++;
    

    for (sm = 0 ; sm < blknum.x ; sm++)
    {
      /* Profile lkWaitSM with the possibility to need to wait the GPU. */
      /** WAIT (WORK) **/
      GETTIME_TIC;
      lkWaitSM(trig, d_trig, sm, blknum, &backbone_stream);
      GETTIME_TOC;
      wait_total += clock_getdiff_nsec(spec_start, spec_stop);
    }

#if 0
  /* NOTE: now that lkWaitSM() waits for stable state this is unfeasible */
    /* Profile lkWaitSM when it's useless (no need to wait the GPU). */
    /* Wait uselessly to get overhead of calling lkWaitSM() */
    /** WAIT (USELESS) **/
    GETTIME_TIC;
    lkWaitSM(trig, sm, blknum);
    GETTIME_TOC;
    sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
    verb("useless wait %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
#endif

    for (sm = 0 ; sm < blknum.x ; sm++)
    {
      /** RETRIEVE DATA (WORK) **/
      GETTIME_TIC;
      int res = lkRetrieveData(data, sm, &backbone_stream);
      GETTIME_TOC;
      log("lk_results[%d] is %d, res is %d\n", sm, lk_results[sm], res);
      printf("--> SM %d returned %d\n", sm, g_results[TEST_IDX-1][sm]);
      retrieve_total += clock_getdiff_nsec(spec_start, spec_stop);
    }

    num_loops++;
  
  } // work loop
  
  /** DISPOSE **/
  GETTIME_TIC;
  lkDispose(trig, d_trig, blknum, &backbone_stream);
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("dispose %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
  
  /* Print timing measurements */
  sprintf(s, "%s t%ld", s, assign_total);
  verb("copy_data(work) %lld\n", assign_total);
  sprintf(s, "%s %ld", s, assign_total / num_loops);
  verb("AVG copy_data(work) %lld\n", assign_total / num_loops);
  sprintf(s, "%s %ld", s, work_total);
  verb("trigger(work) %lld\n", work_total);
  sprintf(s, "%s w%ld", s, wait_total);
  verb("wait %lld\n", wait_total);
  sprintf(s, "%s %ld", s, retrieve_total);
  verb("retrieve data %lld\n", retrieve_total);

  //printf("num_loops %d total %lu avg %lu\n", num_loops, assign_total + wait_total, (assign_total + wait_total) / num_loops);

#if 0
  /**** DEFAULT KERNEL ****/
  data_t *d;
  int *d_resu;
  verb("\n\nDefault kernel:\n");

  verb("boot(init): it's independent of the kernel type. Needed only on"
          "the first execution of a kernel in the program.\n");

  /** AlLOC (INIT) **/
  GETTIME_TIC;
  checkCudaErrors(cudaMalloc(&d, wg * sizeof(data_t)));
  checkCudaErrors(cudaMalloc(&d_resu, wg * sizeof(int)));
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("alloc(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));


  /** COPY DATA **/
  // non prendo i tempi della memorizzazione su host.
  data_t *h_data;
  h_data = (data_t *)malloc(wg * sizeof(data_t));
  for (int i = 0; i < blknum.x; i++) {
      sprintf(h_data[i].str, "Ciao mondo, %d", i);
  }
  int *h_res;
  h_res = (int *)malloc(wg * sizeof(int));
  GETTIME_TIC;
  checkCudaErrors(cudaMemcpy(d, h_data, wg * sizeof(data_t), cudaMemcpyHostToDevice));
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("copy_data(work) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));


  /** TRIGGER **/
  GETTIME_TIC;
  simple_kernel <<< blknum, blkdim, shmem >>> (d, d_resu);
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("trigger(work) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));


  /** WAIT **/
  GETTIME_TIC;
  cudaDeviceSynchronize();
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("wait %lld\n", clock_getdiff_nsec(spec_start, spec_stop));


  /** WAIT INUTILE **/
  GETTIME_TIC;
  cudaDeviceSynchronize();
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("second wait %lld\n", clock_getdiff_nsec(spec_start, spec_stop));


  /** DISPOSE **/
  GETTIME_TIC;
  checkCudaErrors(cudaGetLastError());
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("dispose %lld\n", clock_getdiff_nsec(spec_start, spec_stop));


  /** RETRIEVE_DATA **/
  GETTIME_TIC;
  checkCudaErrors(cudaMemcpy(h_res, d_resu, wg * sizeof(int), cudaMemcpyDeviceToHost));
  GETTIME_TOC;
  sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
  verb("retrieve_data %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
#endif

  if (file)
    fprintf(file, "%d %s\n", blknum.x, s);

}

void parse_cmdline(int argc, char **argv, dim3 * blknum, dim3 * blkdim, int *shmem)
{
  static struct option long_options[] =
  {
      {"cuda-mode", no_argument,        0, 'c'},
      {"dimblock",  required_argument,  0, 'd'},
      {"filename",  required_argument,  0, 'f'},
      {"help",      no_argument,        0, 'h'},
      {"numblock",  required_argument,  0, 'n'},
      {"shmem",     required_argument,  0, 's'},
  };

  int ret, opt_index = 0, o;

  while (1)
  {
    ret = getopt_long(argc, argv, "cd:f:hn:s:", long_options, &opt_index);

    if (ret == -1)
        break;


    switch (ret)
    {
      case 'c':
        cudaMode = !cudaMode;
        log("Setting CUDA mode to '%s'\n", cudaMode ? "on" : "off");
        break;
        
      case 'd':
        o = atoi(optarg);
        if (o >= 1 && o <= MAX_BLK_DIM)
            blkdim->x = o;
        log("Number of Threads per Block set to %d\n", blkdim->x);
        break;
        
      case 'f':
        filename = optarg;
        log("filename set to %s\n", filename);
        break;
        
      case 'h':
        printf("Available options:\n\n");
        printf("-c      Toggles CUDA mode (default %s)\n", cudaMode ? "yes" : "no");
        printf("-d x    Number of CUDA threads per block (default %d)\n", blkdim->x);
        printf("-n x    Number of CUDA threads blocks (default %d)\n", blknum->x);
        printf("-s x    Device per-SM dynamic Shared Memory size in bytes (default %d)\n", *shmem);
        
        printf("\nReport bugs to: mailing-address\n");
        printf("pkg home page: <http://www.gnu.org/software/pkg/>\n");
        printf("General help using GNU software: <http://www.gnu.org/gethelp/>\n");
        exit(0);
        break;
        
      case 'n':
        o = atoi(optarg);
        if (o <= MAX_BLOCKS)
        {
            if (o >= MINBLOCK)
                blknum->x = o;
            else
                blknum->x = MINBLOCK;
        }
        log("Number of Blocks set to %d\n", blknum->x);
        break;
          
          
      case 's':
        o = atoi(optarg);
        if (o > 0)
            *shmem = o;
        log("SHMEM dim set to %d\n", *shmem);
        break;
          
      default:
        printf("How comes?\n");
        break;
    } // switch
    
  } // while
  
} // parse_cmdlines
