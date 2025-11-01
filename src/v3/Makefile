######################################################################
# Compilers
CC = gcc
NVCC = nvcc

######################################################################
# Flags
FLAG1 = -DNDEBUG
CFLAGS = $(FLAG1) -pg                    # <-- profiling enabled
NVCCFLAGS = -arch=sm_86 -std=c++11 -Xcompiler "-Wno-register -pg"   # <-- profiling for CUDA

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
GPU_SRCS = $(COMMON_ARCH) convolve.cu cudaMem.cu
GPU_OBJS = $(COMMON_OBJS) convolve_gpu.o cudaMem.o

EXAMPLES = example3.c
LIBDIRS = -L/usr/local/lib -L/usr/lib

.SUFFIXES: .c .o .cu

.PHONY: all gpu cpu clean profile

######################################################################
# Default: GPU build
all: gpu

gpu: libklt_gpu.a example3

cpu: CFLAGS += -DCPU_ONLY
cpu: libklt_cpu.a example3_cpu

######################################################################
# Compile common .c sources with profiling
.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

######################################################################
# Explicit rules
convolve_cpu.o: convolve.c
	$(CC) -c $(CFLAGS) $< -o $@

trackFeatures_cpu.o: trackFeatures.c
	$(CC) -c $(CFLAGS) $< -o $@

convolve_gpu.o: convolve.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

cudaMem.o: cudaMem.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@


######################################################################
# Libraries
libklt_gpu.a: $(GPU_OBJS)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(GPU_OBJS)
	@echo "Built libklt_gpu.a"

libklt_cpu.a: $(CPU_OBJS)
	rm -f libklt_cpu.a
	ar ruv libklt_cpu.a $(CPU_OBJS)
	@echo "Built libklt_cpu.a"

######################################################################
# Executables (link with -pg)
example3: libklt_gpu.a
	$(NVCC) -pg -O3 -o example3 $(EXAMPLES) libklt_gpu.a $(LIBDIRS) -lm
	@echo "Built GPU executable: ./example3 (profiling enabled)"

example3_cpu: libklt_cpu.a
	$(CC) -pg -O3 -o example3_cpu $(EXAMPLES) libklt_cpu.a $(LIBDIRS) -lm
	@echo "Built CPU executable: ./example3_cpu (profiling enabled)"

######################################################################
# Automated profiling target
# - Rebuilds
# - Runs the program to create gmon.out
# - Dumps gprof results into profile_report.txt
profile: clean gpu
	./example3
	gprof ./example3 gmon.out > profile_report.txt
	@echo "Profiling complete. Report saved in profile_report.txt"

######################################################################
clean:
	rm -f *.o *.a example3 example3_cpu *.tar *.tar.gz libklt*.a \
	      feat*.ppm features.ft features.txt gmon.out profile_report.txt




