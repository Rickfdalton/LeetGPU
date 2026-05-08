
#include <cuda_runtime.h>
#define BLOCK_DIM 256

__global__ void count_kernel(const int* input, int* count, int N, int K){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int local_count;
    if (threadIdx.x ==0){
        local_count=0;
    }
    __syncthreads();

    if (i < N && input[i] == K){
        atomicAdd(&local_count,1);
    }
    __syncthreads();

    if (threadIdx.x ==0){
        atomicAdd(count,local_count);
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threads_per_blocks = BLOCK_DIM;
    int blocks_per_grid = (N+threads_per_blocks-1)/threads_per_blocks;
    count_kernel<<<blocks_per_grid,threads_per_blocks, sizeof(int)>>>(input, output, N, K);
}