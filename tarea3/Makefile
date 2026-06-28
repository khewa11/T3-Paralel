# ─────────────────────────────────────────────
# Makefile — Tarea 3 CUDA
# Adjust CIMG_PATH if CImg.h is not in the same directory
# ─────────────────────────────────────────────

NVCC ?= nvcc
ARCH ?= -arch=sm_75

# Optional overrides for systems that need a specific host compiler or
# non-system CUDA/lib paths. Examples:
#   make HOST_COMPILER_DIR=$$CONDA_PREFIX/bin
#   make CONDA_PREFIX=$$CONDA_PREFIX
#   make CUDA_HOME=/usr/local/cuda-12.4
HOST_COMPILER_DIR ?=
CONDA_PREFIX ?=
CUDA_HOME ?=

BASE_CFLAGS = -O2 -std=c++14 -allow-unsupported-compiler
INCLUDES = -I.
LIBDIRS =
LIBS = -lpng -lz -ljpeg

ifneq ($(strip $(HOST_COMPILER_DIR)),)
BASE_CFLAGS += --compiler-bindir $(HOST_COMPILER_DIR)
endif

ifneq ($(strip $(CONDA_PREFIX)),)
INCLUDES += -I$(CONDA_PREFIX)/include
LIBDIRS += -L$(CONDA_PREFIX)/lib
endif

ifneq ($(strip $(CUDA_HOME)),)
INCLUDES += -I$(CUDA_HOME)/include
LIBDIRS += -L$(CUDA_HOME)/lib64
endif

# Set your GPU architecture: sm_75 = Turing, sm_86 = Ampere, sm_89 = Ada
all: experiment1 experiment2

experiment1: experiment1.cu
	$(NVCC) $(BASE_CFLAGS) $(ARCH) $(INCLUDES) -o experiment1 experiment1.cu $(LIBDIRS) $(LIBS)

experiment2: experiment2.cu
	$(NVCC) $(BASE_CFLAGS) $(ARCH) $(INCLUDES) -o experiment2 experiment2.cu $(LIBDIRS) $(LIBS)

clean:
	rm -f experiment1 experiment2

# ── Quick benchmark (adjust dataset path and image count) ──────────────
DATASET = ./dataset/DIV2K_valid_LR_bicubic/X4
IMAGES  = 100
TILE    = 16

bench1:
	./experiment1 $(DATASET) $(IMAGES) $(TILE)

# Run experiment2 with S = 1, 2, 4, 8, 16
bench2:
	@echo "=== S=1  ===" && ./experiment2 $(DATASET) $(IMAGES) 1  $(TILE)
	@echo "=== S=2  ===" && ./experiment2 $(DATASET) $(IMAGES) 2  $(TILE)
	@echo "=== S=4  ===" && ./experiment2 $(DATASET) $(IMAGES) 4  $(TILE)
	@echo "=== S=8  ===" && ./experiment2 $(DATASET) $(IMAGES) 8  $(TILE)
	@echo "=== S=16 ===" && ./experiment2 $(DATASET) $(IMAGES) 16 $(TILE)

# ── Nsight Systems profiling ───────────────────────────────────────────
profile1:
	nsys profile --stats=true -o exp1_profile ./experiment1 $(DATASET) $(IMAGES) $(TILE)

profile2:
	nsys profile --stats=true -o exp2_profile ./experiment2 $(DATASET) $(IMAGES) 4 $(TILE)
