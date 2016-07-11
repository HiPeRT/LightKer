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

#ifndef __LK_HOST__
#define __LK_HOST__

/* APIs TODO create LK* removing unnecessary args */

/* Utils */
int lkNumThreadsPerSM();
int lkNumClusters();

/* Memutils */
void lkDeviceAlloc(void **pDev, size_t size);
void lkHostAlloc(void **pHost, size_t size);
void lkDeviceFree(void **pDev);
void lkHostFree(void **pHost);
void lkMemcpyToDevice(void *dstDev, const void *srcHost, size_t count);
void lkMemcpyFromDevice(void *dstHost, const void *srcDev, size_t count);

void lkInit(unsigned int blknum_x, unsigned int blkdim_x, int shmem, bool cudaMode, data_t **hostDataPtr, res_t **hostResPtr, data_t **devDataPtr, res_t **devResPtr);
void lkParseCmdLine(int argc, char **argv, dim3 * blknum, dim3 * blkdim, int *shmem, bool *cudaMode);

void lkTriggerMultiple();
void lkWaitMultiple();
void lkWaitSM(int sm);
void lkTriggerSM(int sm);
// void lkRetrieveDataMultiple(res, blknum.x);


#endif /* __LK_HOST__ */
