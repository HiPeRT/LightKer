#
# Makefile for LightKer
#
# Compiler: nvcc
#

all: light_kernel

light_kernel:
	$(MAKE) -C core light_kernel

clean:
	make -C core clean
