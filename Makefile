#
# Makefile for LightKer
#
# Compiler: nvcc
#

NVCC=nvcc

NVCC_OPTS = -O2 # -Xcompiler -Wall
NVCC_OPTS += $(param)

CUDA_INCLUDEPATH=/opt/cuda/include

APPDIR = apps/isKindOf

APPFILES = $(APPDIR)/app.cu $(APPDIR)/data.h
COREFILES = core/light_host.cu
COREHEAD = core/light_kernel.cu core/light_host.h
HEAD = head/head.h head/timer.h head/utils.h

light_kernel: $(COREFILES) $(COREHEAD) $(APPFILES) $(HEAD)
	$(NVCC) -Ihead -I$(APPDIR) -o light_kernel $(COREFILES) -L $(NVCC_OPTS)

.PHONY: clean
clean:
	rm -f *.o light_kernel
