#include <cuda_runtime.h>
#define BLOCK_SIZE 16

__global__ void element_wise_product_kernel(const float* A,const float* B, float* C, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N){
        C[i]=A[i]*B[i];
    }
}

__global__ void reduction_kernel(float* C, float* result, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    //shared memory load
    __shared__ float C_s[BLOCK_SIZE];
    if(i < N){
        C_s[threadIdx.x]= C[i];
    }else{
        C_s[threadIdx.x]= 0.0f;
    }
    __syncthreads();

    for(int stride = 1; stride< BLOCK_SIZE; stride*=2){
        if((threadIdx.x+stride< BLOCK_SIZE)&&((threadIdx.x)%(stride*2)==0)){
            C_s[threadIdx.x]+=C_s[threadIdx.x+stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) atomicAdd(result, C_s[0]);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (N+threads_per_block -1)/threads_per_block;

    float* C_d;//stores element wise product
    cudaMalloc((void**)&C_d, N*sizeof(float));
    cudaMemset(C_d,0.0f, N*sizeof(float));

    //call element-wise kernel
    element_wise_product_kernel<<<blocks_per_grid,threads_per_block>>>(A,B,C_d,N);
    cudaDeviceSynchronize();
    //call reduction kernel
    reduction_kernel<<<blocks_per_grid,threads_per_block>>>(C_d,result,N);

    cudaFree(C_d);
}
