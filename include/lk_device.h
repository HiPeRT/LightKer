#ifndef __LK_DEVICE_H__
#include __LK_DEVICE_H__

/*** App-specific APIs */
/* Allocate space on host for data
 * Formerly known as 'init_data' */
void lkInitAppData(data_t **data, int wg);

/* Assign the given element to the given sm. Doesn't modify any trigger. 
 * Formerly known as 'assign_data' */
int lkSmallOffloadMultiple(data_t *data, dim3 blknum, cudaStream_t *backbone_stream);

/* Small offload for a single SM */
int lkSmallOffload(data_t *data, int sm, cudaStream_t *backbone_stream);

/* Retrieve results by a given sm.
 * The parameter results, and the value returned, refer to LK results, which can be 'OK' and 'FAIL'
 * Formerly known as 'retrieve_data'
 */
int lkRetrieveData(data_t *data, int sm, cudaStream_t *backbone_stream);
// int retrieve_data(trig_t *trig, trig_t *d_trig, int *results, int sm, cudaStream_t *backbone_stream);

#endif /* __LK_DEVICE_H__ */