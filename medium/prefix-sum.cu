#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__global__ void prefix_sum_kernel(const float* input, float* output, float* sum_partial_scans_d, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float block_s[BLOCK_DIM];
    __shared__ float temp_s[BLOCK_DIM];

    if(i<N){
        block_s[threadIdx.x]= input[i];
    }else{
        block_s[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for(int stride=1 ; stride<BLOCK_DIM; stride*=2){
        if(threadIdx.x >= stride){
            temp_s[threadIdx.x ]= block_s[threadIdx.x] + block_s[threadIdx.x - stride];
        }else{
            temp_s[threadIdx.x ]= block_s[threadIdx.x];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            block_s[threadIdx.x ]= temp_s[threadIdx.x];
        }
        __syncthreads();

    }

    if(i<N){
        output[i]= block_s[threadIdx.x];
    }

    if(threadIdx.x == 0){
        sum_partial_scans_d[blockIdx.x] =block_s[BLOCK_DIM -1] ;
    }

}


__global__ void add_kernel(float* output, float* scan_sum_partial_scan_d, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float to_add;
    to_add=0.0f;
    if(blockIdx.x >0 && threadIdx.x==0){
        to_add= scan_sum_partial_scan_d[blockIdx.x-1];
    }
    __syncthreads();

    if(i<N){
        output[i]+=to_add;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads_per_block = BLOCK_DIM;
    int blocks_per_grid = (N+threads_per_block-1 )/threads_per_block;

    float* sum_partial_scans_d;
    float* scan_sum_partial_scan_d;
    float* sum_d;
    cudaMalloc((void**)&sum_partial_scans_d, blocks_per_grid*sizeof(float));
    cudaMalloc((void**)&scan_sum_partial_scan_d, blocks_per_grid*sizeof(float));
    cudaMalloc((void**)&sum_d,sizeof(float));

    prefix_sum_kernel<<<blocks_per_grid,threads_per_block>>>(input, output,sum_partial_scans_d,N);
    cudaDeviceSynchronize();
    prefix_sum_kernel<<<1,threads_per_block>>>(sum_partial_scans_d,scan_sum_partial_scan_d,sum_d,blocks_per_grid);
    cudaDeviceSynchronize();
    add_kernel<<<blocks_per_grid,threads_per_block>>>(output,scan_sum_partial_scan_d, N);

    cudaFree(sum_partial_scans_d);
    cudaFree(scan_sum_partial_scan_d);
    cudaFree(sum_d);

}
