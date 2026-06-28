/*
 * experiment1.cu
 * Tarea 3 - Sistemas Distribuidos y Paralelos
 * Experimento 1: Implementación Tradicional en CUDA (Stream 0)
 *
 * Compile:
 *   nvcc -O2 -o experiment1 experiment1.cu -lcuda
 *   (If using CImg with PNG: add -lpng -lz)
 *
 * Run:
 *   ./experiment1 <dataset_dir> <num_images> <tile_size>
 *   Example: ./experiment1 ./dataset 100 16
 */

#define cimg_display 0
#include "CImg.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

using namespace cimg_library;

// ─────────────────────────────────────────────
// Macro for CUDA error checking
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


// ─────────────────────────────────────────────
// KERNEL 1: Compute the mean vector µ
//
// Each thread computes the mean for one component j
// by summing over all m images: µ_j = (1/m) * Σ v^(k)_j
//
// d_dataset shape: [m × n]  (row k = flattened image k)
// d_mean    shape: [n]
// ─────────────────────────────────────────────
__global__ void computeMeanKernel(const float* __restrict__ d_dataset,
                                   float* __restrict__ d_mean,
                                   int m,   // number of images
                                   int n)   // flattened image size
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // component index
    if (j >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < m; k++) {
        sum += d_dataset[(long long)k * n + j];
    }
    d_mean[j] = sum / (float)m;
}


// ─────────────────────────────────────────────
// KERNEL 2: Center the dataset (subtract mean)
//
// Each thread subtracts µ_j from every image k at component j.
// After this, d_dataset holds the centered matrix V_bar.
// ─────────────────────────────────────────────
__global__ void centerDataKernel(float* __restrict__ d_dataset,
                                  const float* __restrict__ d_mean,
                                  int m,
                                  int n)
{
    // 2D grid: x → component j, y → image k
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n || k >= m) return;

    d_dataset[(long long)k * n + j] -= d_mean[j];
}


// ─────────────────────────────────────────────
// KERNEL 3: Covariance matrix via Tiled SGEMM
//
// C = (1/m) * V_bar^T * V_bar
//
// C is n×n; V_bar is m×n.
// We compute C[j][j'] = (1/m) * Σ_k V_bar[k][j] * V_bar[k][j']
//
// Tiling: each block computes a TILE×TILE sub-block of C.
// Shared memory tiles hold strips of V_bar columns.
// ─────────────────────────────────────────────
template <int TILE>
__global__ void covarianceTiledKernel(const float* __restrict__ d_centered,
                                       float* __restrict__ d_cov,
                                       int m,
                                       int n)
{
    // Shared memory tiles for two column-strips of V_bar
    __shared__ float tileA[TILE][TILE];  // strip for row-dimension j
    __shared__ float tileB[TILE][TILE];  // strip for row-dimension j'

    int j  = blockIdx.x * TILE + threadIdx.x;  // output col index
    int jp = blockIdx.y * TILE + threadIdx.y;  // output row index

    float acc = 0.0f;

    // Iterate over tiles in the m (images) dimension
    for (int t = 0; t < (m + TILE - 1) / TILE; t++) {
        int k = t * TILE + threadIdx.y;   // image index for tileA
        int k2 = t * TILE + threadIdx.x;  // image index for tileB

        // Load tileA: V_bar[k][j]
        tileA[threadIdx.y][threadIdx.x] =
            (k < m && j < n) ? d_centered[(long long)k * n + j] : 0.0f;

        // Load tileB: V_bar[k2][jp]
        tileB[threadIdx.x][threadIdx.y] =
            (k2 < m && jp < n) ? d_centered[(long long)k2 * n + jp] : 0.0f;

        __syncthreads();

        // Accumulate dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            acc += tileA[i][threadIdx.x] * tileB[i][threadIdx.y];
        }
        __syncthreads();
    }

    // Write result (only valid cells)
    if (j < n && jp < n) {
        d_cov[(long long)jp * n + j] = acc / (float)m;
    }
}


// ─────────────────────────────────────────────
// Helper: convert CImg (unsigned char) to float row in h_dataset
// CImg stores pixels as [channel][y][x]; we flatten to [R...G...B...]
// ─────────────────────────────────────────────
void flattenImage(const CImg<unsigned char>& img, float* dest, int n)
{
    int w = img.width(), h = img.height(), c = img.spectrum();
    int idx = 0;
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                dest[idx++] = (float)img(x, y, 0, ch) / 255.0f;
}


// ─────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // ── Parse arguments ──────────────────────
    const char* dataset_dir = (argc > 1) ? argv[1] : "./dataset";
    int num_images           = (argc > 2) ? atoi(argv[2]) : 100;
    int TILE_SIZE            = (argc > 3) ? atoi(argv[3]) : 16;
    int resize_width         = (argc > 4) ? atoi(argv[4]) : 64;
    int resize_height        = (argc > 5) ? atoi(argv[5]) : 48;

    std::cout << "=== Experiment 1: Traditional CUDA (Stream 0) ===\n";
    std::cout << "Dataset: " << dataset_dir
              << "  |  Images: " << num_images
              << "  |  Tile: " << TILE_SIZE
              << "  |  Resize: " << resize_width << "x" << resize_height << "\n\n";

    // ── Print GPU info ───────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name
              << "  VRAM: " << prop.totalGlobalMem / (1024*1024) << " MB\n\n";

    // ── Step 1: Load images into host memory ─
    int width = 0, height = 0, channels = 0, n = 0;

    // Primera imagen para obtener dimensiones
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s/%04dx4.png", dataset_dir, 801);
        CImg<unsigned char> probe;
        probe.load(buf);
        width    = resize_width;
        height   = resize_height;
        channels = 3;
        n        = width * height * channels;
    }

    std::cout << "Image size: " << width << "×" << height
              << "×" << channels << "  →  n=" << n << "\n";

    long long total_floats = (long long)num_images * n;
    size_t dataset_bytes   = total_floats * sizeof(float);
    std::cout << "Dataset size: " << dataset_bytes / (1024*1024) << " MB\n\n";

    // Allocate host dataset (regular pageable memory for Experiment 1)
    std::vector<float> h_dataset(total_floats);

    // Load and flatten all images
    std::cout << "Loading images...\n";
    for (int k = 0; k < num_images; k++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s/%04dx4.png", dataset_dir, 801 + k);
        CImg<unsigned char> img;
        img.load(buf);
        img.resize(resize_width, resize_height, 1, 3);
        flattenImage(img, h_dataset.data() + (long long)k * n, n);
        if ((k + 1) % 10 == 0)
            std::cout << "  Loaded " << (k+1) << "/" << num_images << "\r" << std::flush;
    }
    std::cout << "\nDone loading.\n\n";

    // ── Step 2: Allocate device memory ───────
    float *d_dataset, *d_mean, *d_cov;
    size_t cov_bytes = (long long)n * n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_dataset, dataset_bytes));
    CUDA_CHECK(cudaMalloc(&d_mean,    n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cov,     cov_bytes));
    CUDA_CHECK(cudaMemset(d_cov, 0,   cov_bytes));

    // ── Step 3: Timed copy Host → Device ─────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(d_dataset, h_dataset.data(),
                           dataset_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_h2d = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev_start, ev_stop));
    std::cout << "[Time] H→D copy:          " << ms_h2d << " ms\n";

    // ── Step 4: Kernel — Compute mean ────────
    int threads_1d = 256;
    int blocks_mean = (n + threads_1d - 1) / threads_1d;

    CUDA_CHECK(cudaEventRecord(ev_start));

    computeMeanKernel<<<blocks_mean, threads_1d>>>(d_dataset, d_mean, num_images, n);
    CUDA_CHECK(cudaGetLastError());

    // ── Step 5: Kernel — Center data ─────────
    // 2D grid: x-dim covers n (components), y-dim covers m (images)
    dim3 blockCenter(16, 16);
    dim3 gridCenter((n + 15) / 16, (num_images + 15) / 16);

    centerDataKernel<<<gridCenter, blockCenter>>>(d_dataset, d_mean, num_images, n);
    CUDA_CHECK(cudaGetLastError());

    // ── Step 6: Kernel — Covariance (Tiled) ──
    // We dispatch based on the TILE_SIZE argument
    dim3 blockCov(TILE_SIZE, TILE_SIZE);
    dim3 gridCov((n + TILE_SIZE - 1) / TILE_SIZE,
                 (n + TILE_SIZE - 1) / TILE_SIZE);

    if (TILE_SIZE == 16) {
        covarianceTiledKernel<16><<<gridCov, blockCov>>>(d_dataset, d_cov, num_images, n);
    } else if (TILE_SIZE == 32) {
        covarianceTiledKernel<32><<<gridCov, blockCov>>>(d_dataset, d_cov, num_images, n);
    } else {
        covarianceTiledKernel<8><<<gridCov, blockCov>>>(d_dataset, d_cov, num_images, n);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_compute = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_compute, ev_start, ev_stop));
    std::cout << "[Time] Compute (kernels):  " << ms_compute << " ms\n";

    // ── Step 7: Timed copy Device → Host ─────
    std::vector<float> h_cov((long long)n * n);

    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(h_cov.data(), d_cov, cov_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_start, ev_stop));
    std::cout << "[Time] D→H copy:           " << ms_d2h << " ms\n";

    // ── Summary ───────────────────────────────
    std::cout << "\n--- Summary ---\n";
    std::cout << "  H→D transfer:  " << ms_h2d     << " ms\n";
    std::cout << "  Compute:       " << ms_compute  << " ms\n";
    std::cout << "  D→H transfer:  " << ms_d2h      << " ms\n";
    std::cout << "  Total:         " << (ms_h2d + ms_compute + ms_d2h) << " ms\n";

    // Quick sanity check: print top-left 3×3 of covariance
    std::cout << "\nC[0..2][0..2] (sanity check):\n";
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++)
            printf("  %10.6f", h_cov[(long long)r * n + c]);
        printf("\n");
    }

    // ── Cleanup ───────────────────────────────
    CUDA_CHECK(cudaFree(d_dataset));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_cov));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return 0;
}
