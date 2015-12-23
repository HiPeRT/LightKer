#include "lk_utils.h"

void lkDeviceAlloc(void** pDev, size_t size)
{
  checkCudaErrors(cudaMalloc((void **) pDev, size));
}

void lkHostAlloc(void **pHost, size_t size)
{
  checkCudaErrors(cudaHostAlloc((void **) pHost, size, cudaHostAllocDefault));
}

void lkMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
  checkCudaErrors(cudaMemcpyAsync(dst, src, count, kind, backbone_stream));
  cudaStreamSynchronize(backbone_stream);
}

void lkMemcpyToDevice(void *dstDev, const void *srcHost, size_t count)
{
  lkMemcpy(dstDev, srcHost, count, cudaMemcpyHostToDevice);
}

void lkMemcpyFromDevice(void *dstHost, const void *srcDev, size_t count)
{
  lkMemcpy(dstHost, srcDev, count, cudaMemcpyDeviceToHost);
}
