/*
 *  LightKer - Light and flexible GPU persistent threads library
 *  Copyright (C) 2016  Paolo Burgio
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __LK_UTILS_H__
#define __LK_UTILS_H__

// make an int volatile
#define _vcast(_arr)            * (volatile int *) &_arr
// For printing ptrs
#define _mycast_                (unsigned int) (uintptr_t)

#include <iostream>
#include <assert.h>
#include <inttypes.h>

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << ": " << func << " " << err <<std::endl;
    cudaDeviceReset();
    exit(1);
  }
}
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

/* Retrieve GPU info */

static __device__ __inline__ uint32_t __mysmid()
{
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

static __device__ __inline__ uint32_t __mywarpid()
{
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

static __device__ __inline__ uint32_t __mylaneid()
{
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}
  
static __device__ int d_strlen(const char * str)
{
  int i = 0;
  while(str[i++] != '\0');
  return i-1;
}

#endif /* __LK_UTILS_H__ */
