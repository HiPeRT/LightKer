#include "../../head/head.h"
#include "../../head/utils.h"

__device__ data_t data_dereference(volatile char *data, int blkid)
{
	data_t *d = (data_t *)data;

	return d[blkid];
}

void init_data(void **data, int wg)
{
	checkCudaErrors(cudaHostAlloc(data, wg * sizeof(data_t), cudaHostAllocDefault));
}

void assign_data(void *data, int sm)
{
	data_t *d = (data_t *)data;

	strncpy(d[sm].str, "prova", L_MAX_LENGTH);
	log("assigned data \"%s\" to thread %d\n", str, sm);
}

__device__ int work_nocuda(volatile data_t data)
{
	log("Hi! I'm block %d and I'm working on data ''%s'' [NOCUDA]\n", blockIdx.x, data.str);
	clock_t clock_count = 200000;
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	while (clock_offset < clock_count)
		clock_offset = clock() - start_clock;
	return 1;
}

__device__ int work_cuda(volatile data_t data)
{
	log("Hi! I'm block %d and I'm working on data ''%s'' [CUDA]\n", blockIdx.x, data.str);
	clock_t clock_count = 200000;
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	if (threadIdx.x == 0) {
		while (clock_offset < clock_count)
			clock_offset = clock() - start_clock;
	}
	return 1;
}
