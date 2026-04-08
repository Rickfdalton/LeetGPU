#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__global__ void reduction_kernel(const float* input_d, float* output, int N){
    __shared__ float input_block_s[BLOCK_DIM*2];

    //load to shared mem
    if(2*blockDim.x* blockIdx.x + 2*threadIdx.x < N){
        input_block_s[2*threadIdx.x]=input_d[2*blockDim.x* blockIdx.x + 2*threadIdx.x];
    }else{
        input_block_s[2*threadIdx.x ]=0.0f;
    }
    if(2*blockDim.x* blockIdx.x + 2*threadIdx.x +1 < N){
        input_block_s[2*threadIdx.x +1]=input_d[2*blockDim.x* blockIdx.x + 2*threadIdx.x +1];
    }else{
        input_block_s[2*threadIdx.x +1]=0.0f;
    }

    //compute
    for(int stride=1; stride< 2*BLOCK_DIM; stride*=2){
        int idx = 2*stride*threadIdx.x ;
        if(idx + stride< 2*BLOCK_DIM){
            input_block_s[idx]+=input_block_s[idx +stride ];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(output,input_block_s[0]); 
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK_DIM;
    int elementsPerBlock = 2*threadsPerBlock;
    int blocksPerGrid = (N+elementsPerBlock -1)/elementsPerBlock;
    cudaMemset(output, 0, sizeof(float));
    reduction_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,output,N);

}
