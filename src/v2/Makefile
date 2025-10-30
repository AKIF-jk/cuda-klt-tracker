######################################################################
# Compilers
CC = gcc
NVCC = nvcc

######################################################################
# Flags
FLAG1 = -DNDEBUG
CFLAGS = $(FLAG1)
# NVCC flags: architecture, C++ std for device code, and pass -Wno-register to host compiler
NVCCFLAGS = -arch=sm_86 -std=c++11 -Xcompiler -Wno-register

######################################################################
# Common (non-convolution) sources
COMMON_ARCH = error.c pnmio.c pyramid.c selectGoodFeatures.c \
              storeFeatures.c klt.c klt_util.c writeFeatures.c trackFeatures.c

COMMON_OBJS = $(COMMON_ARCH:.c=.o)

######################################################################
# CPU-only
CPU_SRCS = $(COMMON_ARCH) convolve.c 
CPU_OBJS = $(COMMON_OBJS) convolve_cpu.o

######################################################################
# GPU-only 
GPU_SRCS = $(COMMON_ARCH) convolve.cu 
GPU_OBJS = $(COMMON_OBJS) convolve_gpu.o 

EXAMPLES = example3.c
LIBDIRS = -L/usr/local/lib -L/usr/lib

.SUFFIXES: .c .o .cu

.PHONY: all gpu cpu clean

######################################################################
# Default: GPU build
all: gpu

gpu: libklt_gpu.a example3

cpu: CFLAGS += -DCPU_ONLY
cpu: libklt_cpu.a example3_cpu

######################################################################
# Compile common .c sources with gcc
.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

######################################################################
# Explicit rules for the special files 
convolve_cpu.o: convolve.c
	$(CC) -c $(CFLAGS) $< -o $@

trackFeatures_cpu.o: trackFeatures.c
	$(CC) -c $(CFLAGS) $< -o $@

convolve_gpu.o: convolve.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

trackFeatures_gpu.o: trackFeatures.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

######################################################################
# Build GPU static library
libklt_gpu.a: $(GPU_OBJS)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(GPU_OBJS)
	@echo "Built libklt_gpu.a"

######################################################################
# Build CPU static library
libklt_cpu.a: $(CPU_OBJS)
	rm -f libklt_cpu.a
	ar ruv libklt_cpu.a $(CPU_OBJS)
	@echo "Built libklt_cpu.a"

######################################################################
# GPU executable:
example3: libklt_gpu.a
	$(NVCC) -O3 -o example3 $(EXAMPLES) libklt_gpu.a $(LIBDIRS) -lm
	@echo "Built GPU executable: ./example3"

######################################################################
# CPU executable
example3_cpu: libklt_cpu.a
	$(CC) -O3 -o example3_cpu $(EXAMPLES) libklt_cpu.a $(LIBDIRS) -lm
	@echo "Built CPU executable: ./example3_cpu"

######################################################################
clean:
	rm -f *.o *.a example3 example3_cpu *.tar *.tar.gz libklt*.a \
	      feat*.ppm features.ft features.txt gmon.out

