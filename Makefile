#
# Makefile for LightKer
#
# Compiler: nvcc
#

NVCC=nvcc

NVCC_OPTS = -O2 # -Xcompiler -Wall
NVCC_OPTS += $(param)

CUDA_INCLUDEPATH=/opt/cuda/include

APPFILES = apps/example/example.cu
COREFILES = core/light_kernel.cu core/light_host.cu
COREHEAD = core/light_host.h

light_kernel: $(COREFILES) $(COREHEAD) $(APPFILES)
	$(NVCC) -o light_kernel $(COREFILES) $(APPFILES) -L $(NVCC_OPTS)

.PHONY: clean
clean:
	rm -f *.o light_kernel
