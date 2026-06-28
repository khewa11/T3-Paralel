/*
 * experiment2.cu
 * Tarea 3 - Sistemas Distribuidos y Paralelos
 * Experimento 2: Orquestación con CUDA Streams concurrentes
 *
 * Strategy:
 *   - Split the m images into S batches (one per stream).
 *   - While batch s is being processed on the GPU, batch s+1 is being
 *     copied over PCIe (latency hiding / overlap).
 *   - Covariance is accumulated incrementally:
 *       C += (1/m) * V_bar_batch^T * V_bar_batch
 *     This works because matrix addition is associative.
 *
 * Compile:
 *   nvcc -O2 -o experiment2 experiment2.cu
 *
 * Run:
 *   ./experiment2 <dataset_dir> <num_images> <num_streams> <tile_size>
 *   Example: ./experiment2 ./dataset 100 4 16
 */

#define cimg_display 0
#include "CImg.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

using namespace cimg_library;

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
// KERNEL 1: Accumulate partial mean for a batch
//
// d_partial_mean[j] += Σ_{k in batch} v^(k)_j
// We divide by m (total) only once at the end.
// ─────────────────────────────────────────────
__global__ void accumulateMeanKernel(const float* __restrict__ d_batch,
                                      float* __restrict__ d_partial_mean,
                                      int batch_size,   // images in this batch
                                      int n)            // flattened image size
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < batch_size; k++)
        sum += d_batch[(long long)k * n + j];

    // Atomic add because multiple streams may update concurrently
    atomicAdd(&d_partial_mean[j], sum);
}


// ─────────────────────────────────────────────
// KERNEL 2: Finalize mean (divide by total m)
// ─────────────────────────────────────────────
__global__ void finalizeMeanKernel(float* __restrict__ d_mean,
                                    int n,
                                    int m)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    d_mean[j] /= (float)m;
}


// ─────────────────────────────────────────────
// KERNEL 3: Center a batch in-place
// ─────────────────────────────────────────────
__global__ void centerBatchKernel(float* __restrict__ d_batch,
                                   const float* __restrict__ d_mean,
                                   int batch_size,
                                   int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n || k >= batch_size) return;
    d_batch[(long long)k * n + j] -= d_mean[j];
}


// ─────────────────────────────────────────────
// KERNEL 4: Accumulate covariance for one batch
//
// d_cov[jp][j] += Σ_{k in batch} d_batch[k][j] * d_batch[k][jp]
//
// Uses tiling for memory access efficiency.
// ─────────────────────────────────────────────
template <int TILE>
__global__ void accumulateCovKernel(const float* __restrict__ d_batch,
                                     float* __restrict__ d_cov,
                                     int batch_size,
                                     int n,
                                     float inv_m)   // 1/total_m for scaling
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int j  = blockIdx.x * TILE + threadIdx.x;
    int jp = blockIdx.y * TILE + threadIdx.y;

    float acc = 0.0f;

    for (int t = 0; t < (batch_size + TILE - 1) / TILE; t++) {
        int k  = t * TILE + threadIdx.y;
        int k2 = t * TILE + threadIdx.x;

        tileA[threadIdx.y][threadIdx.x] =
            (k < batch_size && j < n) ? d_batch[(long long)k * n + j] : 0.0f;
        tileB[threadIdx.x][threadIdx.y] =
            (k2 < batch_size && jp < n) ? d_batch[(long long)k2 * n + jp] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++)
            acc += tileA[i][threadIdx.x] * tileB[i][threadIdx.y];

        __syncthreads();
    }

    if (j < n && jp < n)
        atomicAdd(&d_cov[(long long)jp * n + j], acc * inv_m);
}


// ─────────────────────────────────────────────
// Helper: flatten image to float array
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
    int S                    = (argc > 3) ? atoi(argv[3]) : 4;   // num streams
    int TILE_SIZE            = (argc > 4) ? atoi(argv[4]) : 16;
    int resize_width         = (argc > 5) ? atoi(argv[5]) : 64;
    int resize_height        = (argc > 6) ? atoi(argv[6]) : 48;

    std::cout << "=== Experiment 2: CUDA Streams (S=" << S << ") ===\n";
    std::cout << "Dataset: " << dataset_dir
              << "  |  Images: " << num_images
              << "  |  Streams: " << S
              << "  |  Tile: " << TILE_SIZE
              << "  |  Resize: " << resize_width << "x" << resize_height << "\n\n";

    // ── GPU info ─────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name
              << "  VRAM: " << prop.totalGlobalMem / (1024*1024) << " MB\n\n";

    // ── Step 1: Load images (dimensions) ─────
    int width = 0, height = 0, channels = 0, n = 0;
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s/%04dx4.png", dataset_dir, 801);
        CImg<unsigned char> probe;
        probe.load(buf);
        probe.resize(resize_width, resize_height, 1, 3);
        width    = resize_width;
        height   = resize_height;
        channels = 3;
        n        = width * height * channels;  // 9216
    }
    std::cout << "Image size: " << width << "×" << height
              << "×" << channels << "  →  n=" << n << "\n";

    // ── Step 2: Determine batch sizes ────────
    // Distribute images as evenly as possible across S streams
    int base_batch = num_images / S;
    int remainder  = num_images % S;
    // batch s has size: base_batch + (s < remainder ? 1 : 0)

    // ── Step 3: Allocate PINNED host memory ──
    // One pinned buffer per stream to allow concurrent async copies
    long long dataset_bytes_total = (long long)num_images * n * sizeof(float);
    std::cout << "Total dataset: " << dataset_bytes_total / (1024*1024) << " MB\n\n";

    // We allocate one pinned host buffer per stream, sized for the largest batch
    int max_batch = base_batch + (remainder > 0 ? 1 : 0);
    size_t batch_bytes = (long long)max_batch * n * sizeof(float);

    std::vector<float*> h_batches(S);
    for (int s = 0; s < S; s++)
        CUDA_CHECK(cudaMallocHost(&h_batches[s], batch_bytes));

    // ── Step 4: Allocate device memory ───────
    // One device buffer per stream (ping-pong style)
    std::vector<float*> d_batches(S);
    for (int s = 0; s < S; s++)
        CUDA_CHECK(cudaMalloc(&d_batches[s], batch_bytes));

    float *d_mean, *d_partial_mean, *d_cov;
    CUDA_CHECK(cudaMalloc(&d_mean,         n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_mean, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cov,          (long long)n * n * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_partial_mean, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cov,          0, (long long)n * n * sizeof(float)));

    // ── Step 5: Create CUDA streams ──────────
    std::vector<cudaStream_t> streams(S);
    for (int s = 0; s < S; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // ── Timing events ────────────────────────
    cudaEvent_t ev_total_start, ev_total_stop;
    CUDA_CHECK(cudaEventCreate(&ev_total_start));
    CUDA_CHECK(cudaEventCreate(&ev_total_stop));

    // ── PASS 1: Load images into pinned buffers & compute partial means ──
    //
    // Pattern (with S=2, s=0 and s=1):
    //   t=0: load images for s=0 on CPU → async copy s=0 → launch mean kernel s=0
    //   t=1: load images for s=1 on CPU (CPU overlaps with GPU computing s=0)
    //        → async copy s=1 → launch mean kernel s=1
    //
    // With S streams the copy of batch s+1 overlaps with computation of batch s.

    std::cout << "Pass 1: Computing mean with " << S << " streams...\n";
    CUDA_CHECK(cudaEventRecord(ev_total_start));

    int threads_1d = 256;
    int blocks_n   = (n + threads_1d - 1) / threads_1d;

    int img_offset = 0;
    for (int s = 0; s < S; s++) {
        int bs = base_batch + (s < remainder ? 1 : 0);  // this batch size

        // ── CPU: load & flatten images into pinned buffer ──
        for (int k = 0; k < bs; k++) {
            int global_k = img_offset + k;
            char buf[256];
            snprintf(buf, sizeof(buf), "%s/%04dx4.png", dataset_dir, 801 + global_k);
            CImg<unsigned char> img;
            img.load(buf);
            if (img.width() != width || img.height() != height)
                img.resize(width, height, 1, channels);
            flattenImage(img, h_batches[s] + (long long)k * n, n);
        }

        // ── Async H→D copy for this batch ──
        CUDA_CHECK(cudaMemcpyAsync(d_batches[s],
                                   h_batches[s],
                                   (long long)bs * n * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   streams[s]));

        // ── Launch mean accumulation kernel ──
        accumulateMeanKernel<<<blocks_n, threads_1d, 0, streams[s]>>>(
            d_batches[s], d_partial_mean, bs, n);
        CUDA_CHECK(cudaGetLastError());

        img_offset += bs;
    }

    // Sync all streams before finalizing mean
    for (int s = 0; s < S; s++)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    // Divide accumulated sum by total m
    finalizeMeanKernel<<<blocks_n, threads_1d>>>(d_partial_mean, n, num_images);
    CUDA_CHECK(cudaMemcpy(d_mean, d_partial_mean, n * sizeof(float),
                           cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── PASS 2: Center + accumulate covariance ──────────────────────────
    //
    // Key overlap strategy:
    //   - While stream s-1 is running its centering + covariance kernel,
    //     stream s is already copying its data from host.
    //   - This hides PCIe transfer latency behind GPU compute.

    std::cout << "Pass 2: Computing covariance with overlap...\n";

    float inv_m = 1.0f / (float)num_images;
    dim3 blockCov(TILE_SIZE, TILE_SIZE);
    dim3 gridCov((n + TILE_SIZE - 1) / TILE_SIZE,
                 (n + TILE_SIZE - 1) / TILE_SIZE);

    img_offset = 0;
    for (int s = 0; s < S; s++) {
        int bs = base_batch + (s < remainder ? 1 : 0);

        // The pinned buffers already hold the data from Pass 1.
        // Just re-issue the async copy (or we can reuse the same data already on device).
        // Since Pass 1 already copied to device, we can skip re-copy and use d_batches[s].
        // But to demonstrate the stream overlap pattern correctly, we re-issue:
        CUDA_CHECK(cudaMemcpyAsync(d_batches[s],
                                   h_batches[s],
                                   (long long)bs * n * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   streams[s]));

        // Center this batch in-place (subtract global mean)
        dim3 blockCenter(16, 16);
        dim3 gridCenter((n + 15) / 16, (bs + 15) / 16);
        centerBatchKernel<<<gridCenter, blockCenter, 0, streams[s]>>>(
            d_batches[s], d_mean, bs, n);
        CUDA_CHECK(cudaGetLastError());

        // Accumulate covariance: C += (1/m) * V_bar_batch^T * V_bar_batch
        if (TILE_SIZE == 16) {
            accumulateCovKernel<16><<<gridCov, blockCov, 0, streams[s]>>>(
                d_batches[s], d_cov, bs, n, inv_m);
        } else if (TILE_SIZE == 32) {
            accumulateCovKernel<32><<<gridCov, blockCov, 0, streams[s]>>>(
                d_batches[s], d_cov, bs, n, inv_m);
        } else {
            accumulateCovKernel<8><<<gridCov, blockCov, 0, streams[s]>>>(
                d_batches[s], d_cov, bs, n, inv_m);
        }
        CUDA_CHECK(cudaGetLastError());

        img_offset += bs;
    }

    // Wait for all streams to finish
    for (int s = 0; s < S; s++)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    CUDA_CHECK(cudaEventRecord(ev_total_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_total_stop));

    float ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev_total_start, ev_total_stop));
    std::cout << "[Time] Total (compute + transfers): " << ms_total << " ms\n";

    // ── Step 6: Copy result back to host ─────
    cudaEvent_t ev_d2h_start, ev_d2h_stop;
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));

    size_t cov_bytes = (long long)n * n * sizeof(float);
    std::vector<float> h_cov((long long)n * n);

    CUDA_CHECK(cudaEventRecord(ev_d2h_start));
    CUDA_CHECK(cudaMemcpy(h_cov.data(), d_cov, cov_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));

    float ms_d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev_d2h_start, ev_d2h_stop));
    std::cout << "[Time] D→H copy (result):           " << ms_d2h << " ms\n";

    // ── Summary ───────────────────────────────
    std::cout << "\n--- Summary (S=" << S << " streams) ---\n";
    std::cout << "  Total GPU time:   " << ms_total << " ms\n";
    std::cout << "  D→H result copy:  " << ms_d2h   << " ms\n";
    std::cout << "  Grand total:      " << (ms_total + ms_d2h) << " ms\n";

    // Sanity check
    std::cout << "\nC[0..2][0..2] (sanity check):\n";
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++)
            printf("  %10.6f", h_cov[(long long)r * n + c]);
        printf("\n");
    }

    // ── Cleanup ───────────────────────────────
    for (int s = 0; s < S; s++) {
        CUDA_CHECK(cudaFreeHost(h_batches[s]));
        CUDA_CHECK(cudaFree(d_batches[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_partial_mean));
    CUDA_CHECK(cudaFree(d_cov));
    CUDA_CHECK(cudaEventDestroy(ev_total_start));
    CUDA_CHECK(cudaEventDestroy(ev_total_stop));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_start));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_stop));

    return 0;
}
