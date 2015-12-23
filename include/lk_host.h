#ifndef __LK_HOST__
#define __LK_HOST__

/* Utils */
int lkNumThreadsPerSM();
int lkNumSMs();

/* Memutils */
void lkDeviceAlloc(void** pDev, size_t size);
void lkHostAlloc(void ** pHost, size_t size);
void lkMemcpyToDevice(void *dstDev, const void *srcHost, size_t count);
void lkMemcpyFromDevice(void *dstHost, const void *srcDev, size_t count);

#endif /* __LK_HOST__ */
