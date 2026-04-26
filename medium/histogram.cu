#include <cuda_runtime.h>
#define BLOCK_DIM 256

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int local_i = threadIdx.x;

    extern __shared__ int private_bins[];
    if(local_i<num_bins){
        private_bins[local_i]=0;
    }
    __syncthreads();

    if (i < N){
        int val = input[i];
        atomicAdd(&private_bins[val],1);
    }
    __syncthreads();

    if (local_i<num_bins){
        atomicAdd(&histogram[local_i],private_bins[local_i]);
    }

}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threads_per_blocks = BLOCK_DIM;
    int blocks_per_grid = (N+threads_per_blocks-1)/threads_per_blocks;

    histogram_kernel<<<blocks_per_grid,threads_per_blocks, num_bins* sizeof(int)>>>(input, histogram, N, num_bins);


}
