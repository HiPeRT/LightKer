#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <getopt.h>
#include <stdlib.h>

#include "utils.h"
#include "head.h"

#include "data.h"
#include "app.cu"

#include "light_host.h"
#include "light_kernel.cu"

#define MAX_BLOCKS 128
#define MAX_BLK_DIM 32

#ifndef MINBLOCK
#define MINBLOCK 1
#endif

/* Allocate space on host for data */
void init_data(data_t **data, int wg);
/* Assign the given element to the given sm. Doesn't modify any trigger. */
void assign_data(data_t *data, void *payload, int sm);

char *filename = NULL;

inline void print_trigger(const char *fun, trig_t * trig)
{
	log("[%s] to_device %d, from_device %d\n", fun, _vcast(trig[0].to_device), _vcast(trig[0].from_device));
}

/* Initialize the triggers and start the kernel. */
void init(void (*kernel) (volatile trig_t *, volatile data_t *, int *), trig_t *trig, data_t *data, int *results, dim3 blkdim, dim3 blknum, int shmem)
{
	int wg = blknum.x;

	// trigger initialization
	for (int i = 0; i < wg; i++) {
		_vcast(trig[i].from_device) = THREAD_NOP;
		_vcast(trig[i].to_device) = THREAD_NOP;
	}

	kernel <<< blknum, blkdim, shmem >>> (trig, data, results);
}

/* Order the given sm to start working. */
void work(trig_t * trig, int sm, dim3 blknum)
{
	assert(sm <= blknum.x);

	_vcast(trig[sm].to_device) = THREAD_WORK;
}

/* Busy wait until the given sm is working.
 * Trigger to_device is restored to state "THREAD_NOP".
 */
void sm_wait(trig_t *trig, int sm, dim3 blknum)
{
	assert(_vcast(trig[sm].to_device) == THREAD_WORK);

	while (_vcast(trig[sm].from_device) != THREAD_WORKING); //print_trigger("wait", trig);
	while (_vcast(trig[sm].from_device) == THREAD_WORKING); //print_trigger("wait", trig);

	_vcast(trig[sm].to_device) = THREAD_NOP;
}

int retrieve_data(trig_t * trig, int *results, int sm)
{
	assert(_vcast(trig[sm].from_device) == THREAD_FINISHED);

	_vcast(trig[sm].to_device) = THREAD_NOP;

	return _vcast(results[sm]);
}

/* Order to the kernel to exit and wait for its termination. */
void dispose(trig_t * trig, dim3 blknum)
{
	int wg = blknum.x;
	log("Stop 'em!\n");
	for (int i = 0; i < wg; i++) {
		trig[i].to_device = THREAD_EXIT;
	}

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

//      cudaFreeHost(trig);
//      cudaFreeHost(results);
//      cudaDeviceReset();
}

int main(int argc, char **argv)
{
	dim3 blknum = 2;
	dim3 blkdim = (32);
	int shmem = 0;
	FILE *file = NULL;
	char s[10000];

	verb("Warning: with VERBOSE flag on, time measures will be unreliable\n");

	parse_cmdline(argc, argv, &blknum, &blkdim, &shmem);
	assert(blkdim.x <= 32);

	if (filename)
		file = fopen(filename, "a");

	int wg = blknum.x;

	struct timespec spec_start, spec_stop;
	verb("blknum.x : %d\n", blknum.x);

	/** OVERHEAD **/
	GETTIME_TIC;
	GETTIME_TOC;
	GETTIME_TIC;
	GETTIME_TOC;
	sprintf(s, "%ld", clock_getdiff_nsec(spec_start, spec_stop));
	verb("overhead %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

	trig_t *trig;
	data_t *data;
	int *results;

	/** BOOT (INIT) **/
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

	log ("1\n");

	verb("\n\nLight kernel:\n");

	/** ALLOC (INIT) **/
	GETTIME_TIC;
	/* cudaHostAlloc: shared between host and GPU */
	checkCudaErrors(cudaHostAlloc((void **)&trig, wg * sizeof(trig_t), cudaHostAllocDefault));
	init_data(&data, wg);
	checkCudaErrors(cudaHostAlloc((void **)&results, wg * sizeof(int), cudaHostAllocDefault));
	GETTIME_TOC;
	sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
	verb("alloc(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

	log ("2\n");
	/** SPAWN (INIT) **/
	GETTIME_TIC;
	init(uniform_polling_cuda, trig, data, results, blkdim, blknum, shmem);
	GETTIME_TOC;
	sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
	verb("spawn(init) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
	//print_trigger("after init", trig);

	int sm = 0;

	/** COPY_DATA (WORK) **/
	GETTIME_TIC;
	assign_data(data, (void *)"prova", sm);
	GETTIME_TOC;
	sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
	verb("copy_data(work) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

	/** TRIGGER (WORK) **/
	GETTIME_TIC;
	work(trig, sm, blknum);
	GETTIME_TOC;
	sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
	verb("trigger(work) %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

	/* Profile sm_wait with the possibility to need to wait the GPU. */
	/** WAIT **/
	GETTIME_TIC;
	sm_wait(trig, sm, blknum);
	GETTIME_TOC;
	sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
	verb("wait %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

#if 0
	/* Profile sm_wait when it's useless (no need to wait the GPU). */
	/* Wait uselessly to get overhead of calling sm_wait() */
	/** WAIT (USELESS) **/
	GETTIME_TIC;
	sm_wait(trig, sm, blknum);
	GETTIME_TOC;
	sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
	verb("useless wait %lld\n", clock_getdiff_nsec(spec_start, spec_stop));
#endif

	/** RETRIEVE DATA **/
	GETTIME_TIC;
	int res = retrieve_data(trig, results, sm);
	GETTIME_TOC;
	sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
	verb("retrieve_data %lld\n", clock_getdiff_nsec(spec_start, spec_stop));

// test if light_kernel works also with subsequent calls of work
#if 0
	/** WORK **/
	GETTIME_TIC;
	work(trig, data, sm, "provaa", blknum);
	/** WAIT **/
	sm_wait(trig, sm, blknum);
	/** WORK **/
	work(trig, data, sm, "provaaaaa", blknum);
	/** WAIT **/
	sm_wait(trig, sm, blknum);
#endif

	/** DISPOSE **/
	GETTIME_TIC;
	dispose(trig, blknum);
	GETTIME_TOC;
	sprintf(s, "%s %ld", s, clock_getdiff_nsec(spec_start, spec_stop));
	verb("dispose %lld\n", clock_getdiff_nsec(spec_start, spec_stop));


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
	static struct option long_options[] = {
		{"numblock", required_argument, 0, 'n'},
		{"dimblock", required_argument, 0, 'd'},
		{"shmem", required_argument, 0, 's'},
		{"filename", required_argument, 0, 'f'},
	};

	int ret;
	int opt_index = 0;
	int o;

	while (1) {
		ret = getopt_long(argc, argv, "n:d:s:f:", long_options, &opt_index);

		if (ret == -1)
			break;

		o = atoi(optarg);

		switch (ret) {
		case 'n':
			if (o <= MAX_BLOCKS) {
                if (o >= MINBLOCK)
    				blknum->x = o;
                else
                    blknum->x = MINBLOCK;
            }
			break;
		case 'd':
			if (o >= 1 && o <= MAX_BLK_DIM)
				blkdim->x = o;
			break;
		case 's':
			if (o > 0)
				*shmem = o;
			break;
		case 'f':
			filename = optarg;
			break;
		default:
			printf("How come?\n");
		}
	}
}
