#ifndef __LK_HOST__
#define __LK_HOST__

/* Initialize the triggers and start the kernel.
 * formerly known as 'init'
 */
void lkLaunch(void (*kernel) (volatile trig_t *, volatile data_t *, int *),
                          trig_t *trig, trig_t *d_trig, data_t *data, int *results,
                          dim3 blkdim, dim3 blknum, int shmem,
                          cudaStream_t *stream_kernel, cudaStream_t *backbone_stream);


/* Order the given sm to start working.
 * Formerly known as 'work'
 */
void lkTriggerSM(trig_t *trig, trig_t *d_trig, int sm, dim3 blknum, cudaStream_t *backbone_stream);

/* Order the given sm to start working.
 */
void lkTriggerMultiple(trig_t *trig, trig_t *d_trig, dim3 blknum, cudaStream_t *backbone_stream);

/* Busy wait until the given sm is working. Trigger to_device is restored to state "THREAD_NOP".
 * Formerly known as 'sm_wait'
 */
void lkWaitSM(trig_t *trig, trig_t *d_trig, int sm, dim3 blknum, cudaStream_t *backbone_stream);


/* Order to the kernel to exit and wait for its termination.
 * Formerly known as 'dispose'
 */
void lkDispose(trig_t *trig, trig_t *d_trig, dim3 blknum, cudaStream_t *backbone_stream);

#endif /* __LK_HOST__ */
