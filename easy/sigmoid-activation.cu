#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N){
        float x= X[i];
        Y[i]= (1+tanh(x/2.0f))/2.0f;
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}

/*
tanh fomula is optimized
*/