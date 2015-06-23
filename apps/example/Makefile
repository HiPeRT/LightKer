NVCC=nvcc

NVCC_OPTS = -O2 # -Xcompiler -Wall
NVCC_OPTS += $(param)

CUDA_INCLUDEPATH=/opt/cuda/include

light_kernel: example.cu
	$(NVCC) -dc example.cu -L $(NVCC_OPTS)

.PHONY: clean
clean:
	rm -f *.o
