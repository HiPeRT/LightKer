#
# Makefile for LightKer
#
# Compiler: nvcc
#

NVCC=nvcc

param	= -DDEBUG
# param	+= -DDEBUGSEARCH

NVCC_OPTS = -O2 # -Xcompiler -Wall
NVCC_OPTS += $(param)

CUDA_INCLUDEPATH=/opt/cuda/include
LK_HOME=${HOME}/workspace/LightKer

# APPDIR = ${LK_HOME}/apps/isKindOf
APPDIR = ${LK_HOME}/apps/example

COREDIR = ${LK_HOME}/core
INCLUDEDIR = ${LK_HOME}/include

# APPFILES = $(APPDIR)/app.cu $(APPDIR)/data.h
# COREFILES = ${COREDIR}/lk_host.cu ${COREDIR}/lk_device.cu
COREFILES = ${COREDIR}/lk_main.cu
HEAD = ${COREDIR}/head.h ${COREDIR}/timer.h ${COREDIR}/utils.h

light_kernel: $(COREFILES) $(COREHEAD) $(APPFILES) $(HEAD)
	$(NVCC) -I${INCLUDEDIR} -I$(APPDIR) -o light_kernel $(COREFILES) -L $(NVCC_OPTS)

.PHONY: clean
clean:
	rm -f *.o light_kernel
	
all: light_kernel

run:
	./light_kernel
