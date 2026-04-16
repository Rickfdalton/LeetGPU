#include <cuda_runtime.h>
#include <math.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N ) output[i] =  max(lo, min(input[i], hi)); 
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
