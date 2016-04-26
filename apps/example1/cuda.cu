#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include "core/lk_time.h"
#include "core/lk_utils.h"
#define L_MAX_LENGTH 20

#include "work.cu"

__global__ void kernel(char *str[], unsigned int num[])
{
  printf("[EXAMPLE] block %d will work on data '%s'\n", blockIdx.x, str);
  WORK((const char *) str + blockIdx.x * L_MAX_LENGTH, &num[blockIdx.x]);
  printf("[EXAMPLE] block %d returns %d\n", blockIdx.x, num[blockIdx.x]);  
}

int main()
{
  int numBlocks = 1, numThreads = 1;
  char *h_string, ** d_string;
  unsigned int *h_num, *d_num;
  
  /* Input string */
  printf("Alloc input data (Host)\n");
  checkCudaErrors(cudaHostAlloc((void **)&h_string, numBlocks * L_MAX_LENGTH, cudaHostAllocDefault));
  printf("Alloc input data (Device)\n");
  checkCudaErrors(cudaMalloc((void **)&d_string, numBlocks * L_MAX_LENGTH));
  
  /* Output integers */
  printf("Alloc output data (Host)\n");
  checkCudaErrors(cudaHostAlloc((void **)&h_num, numBlocks * sizeof(unsigned int), cudaHostAllocDefault));
  printf("Alloc output data (Device)\n");
  checkCudaErrors(cudaMalloc((void **)&d_num, numBlocks * sizeof(unsigned int)));
  
  printf("Init data\n");
  /* Init app data */
  char * ptr = h_string;
  for(int i=0; i<numBlocks; i++, ptr += L_MAX_LENGTH*i)
  {
    printf("[EXAMPLE] Invoking INIT_DATA h_string @0x%x\n", _mycast_ ptr);
    INIT_DATA(ptr, i);
  }
  
  printf("Copy data to device\n");
  /* Move data to device. We do the very same way as LK, block by block */
  for(int i=0; i<numBlocks; i++)
    checkCudaErrors(cudaMemcpy(d_string + L_MAX_LENGTH*i, h_string + L_MAX_LENGTH*i, L_MAX_LENGTH, cudaMemcpyHostToDevice));
    
  printf("Invoke CUDA kernel..\n");
  kernel<<<numBlocks, numThreads>>>(d_string, d_num);
  printf("Wait for CUDA kernel..\n");
  cudaDeviceSynchronize();
  
  printf("Copy data from device\n");
  /* Move data to device. We do the very same way as LK, block by block */
  for(int i=0; i<numBlocks; i++)
    checkCudaErrors(cudaMemcpy(&h_num[i], &d_num[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
  
  for(int i=0; i<numBlocks; i++)
    CHECK_RESULTS((const char *) h_string + L_MAX_LENGTH*i, h_num[i], numThreads, i);
    
  printf("Dispose data\n");
  checkCudaErrors(cudaFree(d_num));
  checkCudaErrors(cudaFreeHost(h_string));
  checkCudaErrors(cudaFree(d_string));
  checkCudaErrors(cudaFreeHost(h_num));
  
  checkCudaErrors(cudaGetLastError());
  
  return 0;
} // main