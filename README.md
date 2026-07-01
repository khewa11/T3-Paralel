# Tarea 3 — CUDA Image Processing
## Setup, Compilation & Key Concepts

---

## 1. Requirements

- CUDA Toolkit ≥ 11.0
- `CImg.h` (header-only, place in same directory)
- `libpng`, `libjpeg`, `zlib` for image I/O
  ```bash
  sudo apt install libpng-dev libjpeg-dev zlib1g-dev
  ```

---

## 2. Dataset Setup (DIV2K)

Download the validation LR bicubic ×4 set:
```
https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
```
Unzip and point the programs at the folder:
```bash
unzip DIV2K_valid_LR_bicubic_X4.zip -d ./dataset
# Images will be named 0801.png … 0900.png
```

---

## 3. Compilation

Edit `Makefile` → set `ARCH` to match your GPU:
| GPU series     | arch flag   |
|---------------|-------------|
| GTX 16xx / RTX 20xx (Turing) | `-arch=sm_75` |
| RTX 30xx (Ampere) | `-arch=sm_86` |
| RTX 40xx (Ada)    | `-arch=sm_89` |

```bash
make all
```

If you prefer an auto-configuring wrapper, use:
```bash
./build.sh
./build.sh --arch sm_89
./build.sh --arch sm_89 bench2
```

If you need a non-default host compiler or non-system libraries, pass them as
`make` variables instead of hardcoding paths in the file:
```bash
make HOST_COMPILER_DIR=$CONDA_PREFIX/bin CONDA_PREFIX=$CONDA_PREFIX
make CUDA_HOME=/usr/local/cuda-12.4 ARCH=-arch=sm_89
```

On another PC, if the system `gcc/g++` is compatible with your CUDA version,
you can usually just run `make all` after installing the required packages.

---

## 4. Running

```bash
# Experiment 1 — single stream
./experiment1 ./dataset 100 16

# Experiment 2 — 4 concurrent streams
./experiment2 ./dataset 100 4 16

# Full benchmark sweep (S = 1,2,4,8,16)
make bench2 DATASET=./dataset IMAGES=100
```

Optional resize arguments let you test other resolutions without changing the
source:
```bash
./experiment1 ./dataset 100 16 96 72
./experiment2 ./dataset 100 4 16 96 72
```

---

## 5. Architecture Overview

### Experiment 1 — Traditional (Stream 0)

```
Host                         Device (GPU)
────────────────────────────────────────────────────────
[Load all images]
[cudaMemcpy H→D] ──────────────────────────────────────▶ [d_dataset]
                                                           │
                                                           ▼
                                                    [computeMeanKernel]
                                                           │
                                                           ▼
                                                    [centerDataKernel]
                                                           │
                                                           ▼
                                                    [covarianceTiledKernel]
                                                           │
[cudaMemcpy D→H] ◀─────────────────────────────────────── [d_cov]
```

**Steps:**
1. `computeMeanKernel` — each thread handles one component j, loops over all m images.
2. `centerDataKernel` — 2D grid: subtracts µ_j from every (k, j) element.
3. `covarianceTiledKernel` — tiled SGEMM: C = (1/m) * V_bar^T * V_bar.

### Experiment 2 — Multi-Stream (Overlap)

```
Time ──────────────────────────────────────────────────────▶

Stream 0: [copy batch 0]──[center+cov batch 0]
Stream 1:         [copy batch 1]──[center+cov batch 1]
Stream 2:                 [copy batch 2]──[center+cov batch 2]
Stream 3:                         [copy batch 3]──[center+cov batch 3]
                                                                │
                                              atomicAdd accumulates into d_cov
```

**Key ideas:**
- **Pinned Memory** (`cudaMallocHost`): the CPU RAM is page-locked so the DMA engine can copy it without CPU involvement → enables true overlap.
- **`cudaMemcpyAsync`**: returns immediately on the host, copy runs in background on the stream.
- **Incremental covariance**: C = Σ_batches (1/m) * V_bar_batch^T * V_bar_batch (associativity of addition).
- **`atomicAdd`** on d_cov: safe accumulation from multiple streams.

---

## 6. Memory Sizing

The covariance matrix C is n×n floats. For a 128×96×3 image:
- n = 36,864
- C = 36864² × 4 bytes ≈ **5.4 GB** — does NOT fit in most GPUs!

**Solutions used in this code:**
- Resize images to a smaller resolution (e.g. 64×48 → n=9216 → C=340 MB).
- Process in patches/blocks.
- Use `float16` (half precision) if supported.

Add preprocessing in the image loading loop:
```cpp
img.resize(64, 48, 1, 3);  // before flattenImage(...)
```

---

## 7. Profiling with Nsight Systems

```bash
# Generate timeline
nsys profile --stats=true -o exp2_report ./experiment2 ./dataset 100 4 16

# Open in GUI
nsys-ui exp2_report.nsys-rep
```

Look for: `NvtxRange`, `CUDA HtoD`, `CUDA kernel` timeline lanes to verify overlap.

---

## 8. Expected Speedup

| Streams (S) | Expected behavior |
|-------------|------------------|
| 1           | Baseline — no overlap |
| 2           | ~1.3–1.6× (overlap kicks in) |
| 4           | ~1.5–2× (diminishing returns start) |
| 8           | Plateau — PCIe saturated |
| 16          | No gain or slight regression (overhead) |

Saturation happens when the PCIe bus bandwidth is fully utilized; adding more streams doesn't create more bandwidth, it just creates more scheduling overhead.
