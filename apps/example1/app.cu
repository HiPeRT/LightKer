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

#include "include/lk_host.h"
#include "core/lk_utils.h"
#include "data.h"

#define applog(_s, ...)                                             \
{                                                                   \
  printf("[EXAMPLE1] [%s] " _s, __func__, ##__VA_ARGS__);           \
}

unsigned char DATA_ON_DEVICE = 0;
unsigned char RES_ON_DEVICE = 0;

data_t *host_data = 0;
res_t *host_res = 0;
unsigned int count_iter = 0, max_iter = 10;

/* Contains INIT_DATA, WORK and CHECK_RESULTS functions */
#include "work.cu"

/*
 * lkInitAppData - Allocate application-specific data_t using lkDeviceAlloc
 */
void lkInitAppData(data_t **hostDataPtr, data_t **devDataPtr, res_t **hostResPtr, res_t ** devResPtr, int numSm)
{
  applog("\n");
    
  if(DATA_ON_DEVICE)
  {
    lkHostAlloc((void **) &host_data, sizeof(data_t) * numSm);
    lkDeviceAlloc((void **) devDataPtr, sizeof(data_t) * numSm);
  }
  else
  {
    lkHostAlloc((void **) devDataPtr, sizeof(data_t) * numSm);
    host_data = *devDataPtr;
  }
  
  if(RES_ON_DEVICE)
  {
    lkHostAlloc((void **) &host_res, sizeof(res_t) * numSm);
    lkDeviceAlloc((void **) devResPtr, sizeof(res_t) * numSm);
  }
  else
  {
    lkHostAlloc((void **) devResPtr, sizeof(res_t) * numSm);
    host_res = *devResPtr;
  }
  
  for(int sm=0; sm<numSm; sm++)
    INIT_DATA(host_data[sm].str, sm);
	
	// In this example, we do not use hostDataPtr and hostResPtr at all
// 	if(hostDataPtr != 0x0)
// 		*hostDataPtr = host_data;
// 	if(hostResPtr != 0x0)
// 		*hostResPtr = host_res;
} // lkInitAppData

/*
 * lkSmallOffload - Offload all of the SMs
 *                  RETURN 0 if you want the engine to go on and invoke lkWork[No]Cuda, !=0 otherwise - FIXME
 */
int lkSmallOffload(data_t *hostDataPtr, data_t *devDataPtr, int sm, int startRow)
// int lkSmallOffload(data_t *dataPtr, int sm)
{
  applog("Not implemented!\n");
  
  return 0;
}

/*
 * lkSmallOffloadMultiple - Offlad all of the SMs
 *                          RETURN 0 if you want the engine to go on and invoke lkWork(No)Cuda, !=0 otherwise - FIXME
 */
int lkSmallOffloadMultiple(data_t *hostDataPtr, data_t *devDataPtr, int numSm)
{
//   applog(numSm %d dataPtr 0x%x count_iter %u max_iter %u\n", numSm, _mycast_ dataPtr, count_iter, max_iter);
    
  if(DATA_ON_DEVICE)
    lkMemcpyToDevice(&devDataPtr[0], &host_data[0], sizeof(data_t) *numSm);

  return ++count_iter != max_iter;
}
/*
 * lkWorkCuda - Perform your work
 *              RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkCuda(volatile data_t *dataPtr, volatile res_t *res)
{
//   applog(Hi! I'm core %d of SM %d and I'm working on data ''%s''\n", threadIdx.x, blockIdx.x, data->str);

  WORK((const char *) dataPtr->str, (unsigned int *) &res->num);

//   applog(Block %d returns %d \n", blockIdx.x, res->num);
  
  return 0;
}

/*
 * lkRetrieveData - Retrieve app results
 */
void lkRetrieveData(res_t *hostResPtr, res_t *devResPtr, res_t * resPtr, int sm)
{
  applog("Not implemented!\n");
}

/*
 * lkRetrieveDataMultiple - Retrieve app results
 */
void lkRetrieveDataMultiple(res_t *hostResPtr, res_t *devResPtr, unsigned int numSm)
{
//   applog(sm %d \n", sm);

  if(RES_ON_DEVICE)
    lkMemcpyFromDevice(&host_res[0], &devResPtr[0], sizeof(res_t) * numSm);
  
//   for(int sm =0; sm<numSm; sm++)
//     CHECK_RESULTS((const char *) host_data[sm].str, host_res[sm].num, lkNumThreadsPerSM(), sm);
}

/*
 * lkWorkNoCuda - Perform your work
 *                RETURN INT: 0 if everything went fine, !=0 in case of errors
 */
__device__ int lkWorkNoCuda(volatile data_t *data, volatile res_t *res)
{
  applog("Not implemented!\n");
  return 1;
}

