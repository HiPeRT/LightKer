void lkDeviceAlloc(void** dataHostPtr, size_t size)
{
  checkCudaErrors(cudaMalloc((void **) dataHostPtr, size));
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

void lkMemcpyToDevice(void *dst, const void *src, size_t count)
{
  lkMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

void lkMemcpyFromDevice(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
  lkMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}
