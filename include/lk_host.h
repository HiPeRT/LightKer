#ifndef __LK_HOST__
#define __LK_HOST__

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

/* APIs TODO create LK* removing unnecessary args */
void lkInit(unsigned int blknum_x, unsigned int blkdim_x, int shmem, bool cudaMode, data_t ** data, res_t ** res);
void lkTriggerMultiple();
void lkWaitMultiple();
void lkParseCmdLine(int argc, char **argv, dim3 * blknum, dim3 * blkdim, int *shmem, bool *cudaMode);
// void lkRetrieveDataMultiple(res, blknum.x);

#endif /* __LK_HOST__ */
