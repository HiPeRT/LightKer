#include "../../head/head.h"
#include "../../head/utils.h"

void init_data(data_t **data, int numblocks)
{
	checkCudaErrors(cudaHostAlloc((void **)data, numblocks * sizeof(data_t), cudaHostAllocDefault));
}

void assign_data(data_t *data, void *payload, int sm)
{
	strncpy(data[sm].str, (char *)payload, L_MAX_LENGTH);
	log("assigned data \"%s\" to thread %d\n", (char *)payload, sm);
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
