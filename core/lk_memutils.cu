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

#include "lk_utils.h"

void lkDeviceAlloc(void** pDev, size_t size)
{
  checkCudaErrors(cudaMalloc((void **) pDev, size));
}

void lkHostAlloc(void **pHost, size_t size)
{
  checkCudaErrors(cudaHostAlloc((void **) pHost, size, cudaHostAllocDefault));
}

void lkDeviceFree(void **pDev)
{
  checkCudaErrors(cudaFree(pDev));
}

void lkHostFree(void **pHost)
{
  checkCudaErrors(cudaFreeHost(pHost));
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
