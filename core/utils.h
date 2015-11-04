#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// #include <unistd.h>
// #include <unistd.h>
// #include <time.h>
// #include <math.h>
#include <inttypes.h>
// #include <getopt.h>
// #include <stdlib.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << ": " << func << " " << err <<std::endl;
    cudaDeviceReset();
    exit(1);
  }
}

/* Retrieve GPU info */

static __device__ __inline__ uint32_t __mysmid(){
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;}

static __device__ __inline__ uint32_t __mywarpid(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;}

static __device__ __inline__ uint32_t __mylaneid(){
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;}

#endif /* __UTILS__ */
