######################################################################
# Compilers
CC = gcc
NVCC = nvcc

######################################################################
# Flags
FLAG1 = -DNDEBUG
CFLAGS = $(FLAG1) -pg                    # <-- profiling enabled
NVCCFLAGS = -arch=sm_86 -std=c++11 -Xcompiler "-Wno-register -pg"   # <-- profiling for CUDA
NVCCFLAGS_T4 = -arch=sm_75 -std=c++11 -Xcompiler "-Wno-register -pg"   # <-- T4 GPU (sm_75)

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
GPU_SRCS = $(COMMON_ARCH) convolve.cu klt_cuda_mem.cu
GPU_OBJS = $(COMMON_OBJS) convolve_gpu.o klt_cuda_mem.o

EXAMPLES = example3.c
LIBDIRS = -L/usr/local/lib -L/usr/lib

.SUFFIXES: .c .o .cu

.PHONY: all gpu cpu clean profile colabGpu

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

cudaMem.o: klt_cuda_mem.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# T4 GPU specific object files
convolve_gpu_t4.o: convolve.cu
	$(NVCC) -c $(NVCCFLAGS_T4) $< -o $@

cudaMem_t4.o: klt_cuda_mem.cu
	$(NVCC) -c $(NVCCFLAGS_T4) $< -o $@


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

# T4 GPU library
GPU_OBJS_T4 = $(COMMON_OBJS) convolve_gpu_t4.o cudaMem_t4.o
libklt_gpu_t4.a: $(GPU_OBJS_T4)
	rm -f libklt_gpu_t4.a
	ar ruv libklt_gpu_t4.a $(GPU_OBJS_T4)
	@echo "Built libklt_gpu_t4.a for T4 GPU (sm_75)"

######################################################################
# Executables (link with -pg)
example3: libklt_gpu.a
	$(NVCC) -pg -O3 -o example3 $(EXAMPLES) libklt_gpu.a $(LIBDIRS) -lm
	@echo "Built GPU executable: ./example3 (profiling enabled)"

example3_cpu: libklt_cpu.a
	$(CC) -pg -O3 -o example3_cpu $(EXAMPLES) libklt_cpu.a $(LIBDIRS) -lm
	@echo "Built CPU executable: ./example3_cpu (profiling enabled)"

######################################################################
# T4 GPU build (for Google Colab and similar T4 environments)
colabGpu: libklt_gpu_t4.a
	$(NVCC) -pg -O3 -o example3 $(EXAMPLES) libklt_gpu_t4.a $(LIBDIRS) -lm
	@echo "Built T4 GPU executable: ./example3 (sm_75, profiling enabled)"

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
	      feat*.ppm features.ft features.txt gmon.out profile_report.txt \
	      *_t4.o libklt_gpu_t4.a




