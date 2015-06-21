NVCC=nvcc

NVCC_OPTS = -O2 # -Xcompiler -Wall
NVCC_OPTS += $(param)

CUDA_INCLUDEPATH=/opt/cuda/include

light_kernel: light_kernel.cu light_host.cu light_host.h head.h utils.h Makefile
	$(NVCC) -o light_kernel light_host.cu -L $(NVCC_OPTS)

.PHONY: clean
clean:
	rm -f *.o light_kernel
