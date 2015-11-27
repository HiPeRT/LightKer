#include <assert.h>

/* LK internal headers */
#include "lk_head.h"
#include "lk_utils.h"

/* To treat APP specific data_t type */
#include "data.h"

/* Low-level functionalities */

/* Initialize the triggers and start the kernel.
 * formerly known as 'init'
 */

extern cudaStream_t stream_kernel;
extern cudaStream_t backbone_stream;
void lkLaunch(void (*kernel) (volatile mailbox_elem_t *, volatile mailbox_elem_t *,  volatile data_t *, int *),
              data_t *data, int *results,
              dim3 blkdim, dim3 blknum, int shmem)
{
  int wg = blknum.x;
  log("\n");

  // Trigger initialization
  for (int sm = 0; sm < wg; sm++)
    lkHToDevice(sm) = THREAD_NOP;
  
  lkMailboxFlush(true);

  kernel <<< blknum, blkdim, shmem, stream_kernel >>> (d_to_device, d_from_device, data, results);
  
  // Wait for LK thread(s) to be ready to work
  for (int sm = 0; sm < wg; sm++)
  {
    while(lkHFromDevice(sm) != THREAD_NOP)
    {
      lkMailboxFlushSM(false, sm);
      lkMailboxSync();
    }
  }
  log("Done.");
} // lkLaunch

/* Order the given SM to start working.
 * Formerly known as 'work'
 */
void lkTriggerSM(int sm, dim3 blknum)
{
  LK_WARN_NOT_SUPPORTED("lkTriggerSM");
#if 0
  log("SM %d blknum %d\n", sm, blknum.x);
  assert(sm <= blknum.x);
//   assert(_vcast(trig[sm].from_device) != THREAD_WORK);
  assert(lkFromDevice(h_trig, sm) != THREAD_WORK);

//   _vcast(trig[sm].to_device) = THREAD_WORK;
  lkToDevice(h_trig, sm) = THREAD_WORK;
//   checkCudaErrors(cudaMemcpyAsync(&d_trig[sm], &trig[sm], sizeof(trig_t), cudaMemcpyHostToDevice, *backbone_stream));
  lkFlushMailboxSM(true, sm);
  print_trigger("[lkTriggerSM]", trig, sm);
#endif
}


/* Order all SMs to start working.
 */
void lkTriggerMultiple(dim3 blknum)
{
  log("blknum=%d\n", blknum.x);

  for(int sm=0; sm<blknum.x; sm++)
  {
//     log("Triggering SM #%d\n", sm);
    lkHToDevice(sm) = THREAD_WORK;
  }
//   lkMailboxPrint("lkTriggerMultiple", 0);
  
//   log("Transfering all mailboxes to Device..\n");
  lkMailboxFlush(true);
  lkMailboxSync();
  
  log("Done.\n");  
} // lkTriggerMultiple

/* Busy wait until the given sm is working. Trigger to_device is restored to state "THREAD_NOP".
 * Formerly known as 'sm_wait'
 */
void lkWaitSM(int sm, dim3 blknum)
{
  log("SM #%d\n", sm);
  
  if(lkHToDevice(sm) != THREAD_WORK)
  {
    printf("SM #%d was not triggered!", sm);
    return;
  }
  
//   log("waiting for SM #%d to start working\n",sm);

//   lkMailboxPrint("lkWaitSM", sm);
  do
  {
    lkMailboxFlushSM(false, sm);
    lkMailboxSync();
  }
  while (lkHFromDevice(sm) != THREAD_WORKING && lkHFromDevice(sm) != THREAD_FINISHED);

//   log("SM #%d is working: waiting for it to end\n", sm);
//   lkMailboxPrint("lkWaitSM", sm);
  
  do
  {
    lkMailboxFlushSM(false, sm);
    lkMailboxSync();
  }
  while (lkHFromDevice(sm) == THREAD_WORKING);
  
  log("Done.\n");  
}


/* Order to the kernel to exit and wait for its termination.
 * Formerly known as 'dispose'
 */
void lkDispose(dim3 blknum)
{
  int wg = blknum.x;
  log("There are %d SMs\n", wg);
//   return;
  for (int sm = 0; sm < wg; sm++)
  {
//     lkMailboxPrint("lkDispose", sm);
//     log("Stopping SM #%d\n", sm);
    lkHToDevice(sm) = THREAD_EXIT;
  }
  
//   log("Writing mailbox to device..\n");
  lkMailboxFlush(true);
  // No need to perform lkMailboxSync here, as we do cudaDeviceSynchronize right after.
  
//   log("Waiting for persistent LK thread to shut down...\n");
  checkCudaErrors(cudaDeviceSynchronize());
  log("Persistent LK thread stopped.\n");
  
  checkCudaErrors(cudaGetLastError());

//   log("Resetting Mailbox..\n");
  lkMailboxFree();
  
//   log("Resetting device..\n");
  checkCudaErrors(cudaDeviceReset());
  
  log("Done.\n");
} // lkDispose
