#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 1024

enum ReductionMode { Max, Min };

__global__ void histogram(const float* const luminance, const float minLuminance, const float lumRange,
    const size_t numBins, unsigned int* const output, const size_t N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;
    int bin = min((int) numBins - 1, (int)((luminance[id] - minLuminance) / lumRange * numBins));
    atomicAdd(&output[bin], 1);
}

__global__ void reduction(const float* const input, float* output, const size_t N, ReductionMode mode) {
    __shared__ float values[BLOCK_SIZE];
    int mult = mode == ReductionMode::Max ? 1 : -1;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= N) values[threadIdx.x] = -INFINITY;
    else values[threadIdx.x] = input[id] * mult;
    __syncthreads();
    //blockDim must be multiple of 2
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s)
            values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) output[blockIdx.x] = values[0] * mult;
}

__global__ void scan(unsigned int* histogram, int numBins) {
    extern __shared__ int values[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;
    int offset = 1;
    int ai = threadId;
    int bi = threadId + numBins / 2;
    values[ai] = histogram[id];
    values[bi] = histogram[id + numBins / 2];
    for (int i = numBins / 2; i > 0; i /= 2) {
        __syncthreads();
        if (threadId < i) {
            ai = offset * (2 * threadId + 1) - 1;
            bi = offset * (2 * threadId + 2) - 1;
            values[bi] += values[ai];
        }
        offset *= 2;    
    }
    if (threadId == 0) values[numBins - 1] = 0;
    for (int i = 1; i < numBins; i *= 2) {
        offset /= 2;
        __syncthreads();
        if (threadId < i) {
            ai = offset * (2 * threadId + 1) - 1;
            bi = offset * (2 * threadId + 2) - 1;
            int temp = values[ai];
            values[ai] = values[bi];
            values[bi] += temp;
        }
    }
    __syncthreads();
    histogram[id] = values[threadId];
    histogram[id + numBins / 2] = values[threadId + numBins / 2];
}

void calculate_cdf(const float* const d_logLuminance,
    unsigned int* const d_cdf,
    float& min_logLum,
    float& max_logLum,
    const size_t numRows,
    const size_t numCols,
    const size_t numBins)
{
    size_t totalSize = numRows * numCols;
    //1) Find luminance max and min values
    //Max init:
    float* d_input_max;
    float* d_output_max;
    cudaMalloc(&d_input_max, sizeof(float) * totalSize);
    cudaMalloc(&d_output_max, sizeof(float) * totalSize);
    cudaMemcpy(d_input_max, d_logLuminance, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
    //Min init:
    float* d_input_min;
    float* d_output_min;
    cudaMalloc(&d_input_min, sizeof(float) * totalSize);
    cudaMalloc(&d_output_min, sizeof(float) * totalSize);
    cudaMemcpy(d_input_min, d_logLuminance, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
    //Kernels:
    int gridSize = (totalSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int currentSize = totalSize;
    while (currentSize > 1) {
        reduction << <gridSize, BLOCK_SIZE >> > (d_input_max, d_output_max, currentSize, ReductionMode::Max);
        reduction << <gridSize, BLOCK_SIZE >> > (d_input_min, d_output_min, currentSize, ReductionMode::Min);
        std::swap(d_input_max, d_output_max);
        std::swap(d_input_min, d_output_min);
        currentSize = gridSize;
        gridSize = (gridSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    //Results are copied to the CPU:
    cudaMemcpy(&max_logLum, d_input_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_logLum, d_input_min, sizeof(float), cudaMemcpyDeviceToHost);
    
    //2)Determine the range to be represented
    float lumRange = max_logLum - min_logLum;

    //3)Generate a histogram using the following formula:
    //  bin = (Lum[i] - lumMin) / lumRange * numBins
    gridSize = (totalSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogram << <gridSize, BLOCK_SIZE >> > (d_logLuminance, min_logLum, lumRange, numBins, d_cdf, totalSize);

    //4)Perform an exclusive scan to obtain the cumulative distribution function (CDF) 
    scan << <1, numBins / 2, numBins * sizeof(int) >> > (d_cdf, numBins);
    cudaDeviceSynchronize();

    //Memory release
    cudaFree(d_input_max);
    cudaFree(d_output_max);
    cudaFree(d_input_min);
    cudaFree(d_output_min);
}
