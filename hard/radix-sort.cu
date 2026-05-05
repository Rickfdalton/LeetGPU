#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define BLOCK_DIM 1024

using namespace std;

__global__ void segmented_scan_kernel(unsigned int* input_d, unsigned int* output_d ,unsigned int* partials_sums_d,int N, int blocks);
__global__ void stitch_scan(unsigned int* input, unsigned int* scan_of_partial_sums_d, int N, const int blocks);
__global__ void bit_extraction_kernel(unsigned int* input, unsigned int* output, int N, int bit_idx);
__global__ void prepare_output_kernel(unsigned int* input, unsigned int* scan, unsigned int* output, int N, int bit_idx);

void kogge_stone_inclusive_scan(unsigned int* input, unsigned int* output ,int N);

__global__ void segmented_scan_kernel(unsigned int* input_d, unsigned int* output_d ,unsigned int* partials_sums_d,int N, int blocks){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int block_s[BLOCK_DIM];
    __shared__ unsigned int temp[BLOCK_DIM];

    if(i < N) output_d[i] = input_d[i];
    __syncthreads();

    block_s[threadIdx.x] = (i < N) ? input_d[i] : 0;
    __syncthreads();

    for(int stride = 1; stride < BLOCK_DIM; stride *= 2){
        if(threadIdx.x >= stride)
            temp[threadIdx.x] = block_s[threadIdx.x] + block_s[threadIdx.x - stride];
        else
            temp[threadIdx.x] = block_s[threadIdx.x];

        __syncthreads();
        block_s[threadIdx.x] = temp[threadIdx.x];
        __syncthreads();
    }

    if(threadIdx.x == blockDim.x - 1 && i < N)
        partials_sums_d[blockIdx.x] = block_s[threadIdx.x];

    if(i < N)
        output_d[i] = block_s[threadIdx.x];
}


__global__ void stitch_scan(unsigned int* input, unsigned int* scan_of_partial_sums_d, int N, const int blocks){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int to_add;

    if(threadIdx.x == 0){
        if(blockIdx.x == 0)
            to_add = 0;
        else
            to_add = scan_of_partial_sums_d[blockIdx.x - 1];
    }

    __syncthreads();

    if(i < N){
        input[i] += to_add;
    }
}


void kogge_stone_inclusive_scan(unsigned int* input, unsigned int* output ,int N){

    int threads_per_block = BLOCK_DIM;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    unsigned int *partials_sums_d, *input_d, *output_d, *scan_of_partial_sums_d;

    cudaMalloc((void**)&partials_sums_d, blocks_per_grid * sizeof(unsigned int));
    cudaMalloc((void**)&input_d, N * sizeof(unsigned int));
    cudaMalloc((void**)&output_d, N * sizeof(unsigned int));
    cudaMalloc((void**)&scan_of_partial_sums_d, blocks_per_grid * sizeof(unsigned int));

    cudaMemcpy(input_d, input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Step 1: block-level scan
    segmented_scan_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_d, output_d, partials_sums_d, N, blocks_per_grid);
    cudaDeviceSynchronize();

    // Step 2: recursive scan of partial sums (FIXED PART)
    if (blocks_per_grid > 1) {
        kogge_stone_inclusive_scan(partials_sums_d, scan_of_partial_sums_d, blocks_per_grid);
    } else {
        cudaMemcpy(scan_of_partial_sums_d, partials_sums_d,
                   sizeof(unsigned int),
                   cudaMemcpyDeviceToDevice);
    }

    // Step 3: stitch
    stitch_scan<<<blocks_per_grid, threads_per_block>>>(
        output_d, scan_of_partial_sums_d, N, blocks_per_grid);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(partials_sums_d);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(scan_of_partial_sums_d);
}

__global__ void bit_extraction_kernel(unsigned int* input, unsigned int* output, int N, int bit_idx){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N){
        unsigned int bit = (input[i] >> bit_idx) & 1u;
        output[i] = bit;
    }
}

__global__ void prepare_output_kernel(unsigned int* input, unsigned int* scan, unsigned int* output, int N, int bit_idx){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < N){

        unsigned int bit = (input[i] >> bit_idx) & 1u;

        unsigned int no_ones_left = (i == 0) ? 0 : scan[i - 1];

        unsigned int out_zero = i - no_ones_left;
        unsigned int out_one  = N - scan[N - 1] + no_ones_left;

        if(bit == 1u){
            output[out_one] = input[i];
        } else {
            output[out_zero] = input[i];
        }
    }
}

extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {

    unsigned int* host_input = new unsigned int[N];
    cudaMemcpy(host_input, input, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int max = host_input[0];
    for(int i = 1; i < N; i++){
        if(host_input[i] > max){
            max = host_input[i];
        }
    }

    unsigned int bit_width = 0;
    while(max > 0){
        bit_width++;
        max >>= 1;
    }

    unsigned int *input_d, *output_d, *bit_store_d, *bit_scan_d;

    cudaMalloc((void**)&input_d, N * sizeof(unsigned int));
    cudaMalloc((void**)&output_d, N * sizeof(unsigned int));
    cudaMalloc((void**)&bit_store_d, N * sizeof(unsigned int));
    cudaMalloc((void**)&bit_scan_d, N * sizeof(unsigned int));

    cudaMemcpy(input_d, input, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMemset(bit_store_d, 0, N * sizeof(unsigned int));
    cudaMemset(bit_scan_d, 0, N * sizeof(unsigned int));

    int threads_per_block = BLOCK_DIM;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    for(int iter = 0; iter < bit_width; iter++){

        bit_extraction_kernel<<<blocks_per_grid, threads_per_block>>>(input_d, bit_store_d, N, iter);
        cudaDeviceSynchronize();

        kogge_stone_inclusive_scan(bit_store_d, bit_scan_d, N);
        cudaDeviceSynchronize();

        prepare_output_kernel<<<blocks_per_grid, threads_per_block>>>(input_d, bit_scan_d, output_d, N, iter);
        cudaDeviceSynchronize();

        unsigned int* tmp = input_d;
        input_d = output_d;
        output_d = tmp;
    }

    cudaMemcpy(output, input_d, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(bit_store_d);
    cudaFree(bit_scan_d);
}