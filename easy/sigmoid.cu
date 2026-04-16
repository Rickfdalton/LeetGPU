#include <cuda_runtime.h>
#include <math.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N){
        float x= input[i];
        output[i]= x/(1.0 + __expf(-x));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

/*
need to optimize further
*/