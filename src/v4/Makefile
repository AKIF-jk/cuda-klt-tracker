######################################################################
# Makefile: Build CPU and OpenACC executables only
# Usage:
#   make           # builds both CPU and OpenACC executables
#   make ACC=pgcc  # override OpenACC compiler
#   make clean     # removes all build artifacts
######################################################################

# Compilers
CC   = gcc
ACC ?= gcc       # OpenACC compiler; override with ACC=pgcc, nvcc, etc.

# Flags
CFLAGS    = -O2 -DNDEBUG
ACCFLAGS  = -fopenacc -O3 -DNDEBUG

# Source files
COMMON_SRCS    = error.c pnmio.c pyramid.c selectGoodFeatures.c \
                 storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
CPU_SRCS       = $(COMMON_SRCS) convolve.c
OPENACC_SRCS   = $(COMMON_SRCS) convolve_openacc.c

# Object/build directories
CPU_BUILD      = build_cpu
ACC_BUILD      = build_acc

# Object files
CPU_OBJS       = $(patsubst %.c,$(CPU_BUILD)/%.o,$(CPU_SRCS))
ACC_OBJS       = $(patsubst %.c,$(ACC_BUILD)/%.o,$(OPENACC_SRCS))

# Executables
EXE_CPU        = example3_cpu
EXE_ACC        = example3_openacc
EXAMPLE        = example3.c

.PHONY: all cpu openacc clean dirs_cpu dirs_acc

# Default target: build both executables
all: cpu openacc
	@echo "=== Build complete. Run ./$(EXE_CPU) or ./$(EXE_ACC) to execute ==="

######################################################################
# Build CPU executable
cpu: $(EXE_CPU)

$(CPU_BUILD)/%.o: %.c | dirs_cpu
	$(CC) -c $(CFLAGS) $< -o $@

libklt_cpu.a: $(CPU_OBJS)
	@echo "Building CPU static library..."
	ar ruv $@ $(CPU_OBJS)
	@echo "Built $@"
	rm -f $(CPU_OBJS)

$(EXE_CPU): libklt_cpu.a $(EXAMPLE)
	@echo "Linking CPU executable..."
	$(CC) -O3 -o $(EXE_CPU) $(EXAMPLE) -L. -lklt_cpu -lm
	@echo "Built CPU executable: ./$(EXE_CPU)"

######################################################################
# Build OpenACC executable
openacc: $(EXE_ACC)

$(ACC_BUILD)/%.o: %.c | dirs_acc
	$(ACC) -c $(ACCFLAGS) $< -o $@

libklt_openacc.a: $(ACC_OBJS)
	@echo "Building OpenACC static library..."
	ar ruv $@ $(ACC_OBJS)
	@echo "Built $@"
	rm -f $(ACC_OBJS)

$(EXE_ACC): libklt_openacc.a $(EXAMPLE)
	@echo "Linking OpenACC executable..."
	$(ACC) $(ACCFLAGS) -O3 -o $(EXE_ACC) $(EXAMPLE) -L. -lklt_openacc -lm
	@echo "Built OpenACC executable: ./$(EXE_ACC)"

######################################################################
# Ensure build directories exist
dirs_cpu:
	@mkdir -p $(CPU_BUILD)

dirs_acc:
	@mkdir -p $(ACC_BUILD)

######################################################################
# Clean
clean:
	@echo "Cleaning build directories, libraries, and executables..."
	rm -rf $(CPU_BUILD) $(ACC_BUILD) *.a $(EXE_CPU) $(EXE_ACC)
	@echo "Clean complete."
