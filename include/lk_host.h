#ifndef LIGHT_HOST
#define LIGHT_HOST

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void parse_cmdline(int, char**, dim3*, dim3*, int*);

#endif /* LIGHT_HOST */
