#
# Makefile for LightKer
#
# Compiler: nvcc
#

NVCC = nvcc
NVCC_OPTS = -O2 # -Xcompiler -Wall
NVCC_OPTS += $(param)

all: light_kernel

APP = example

light_kernel:
	$(MAKE) -C core light_kernel
	$(MAKE) -C apps/$(APP)
	$(NVCC) -o light_kernel core/light_host.o apps/$(APP)/$(APP).o -L $(NVCC_OPTS)

clean:
	make -C core clean
	make -C apps/$(APP) clean
	rm -f *.o light_kernel
