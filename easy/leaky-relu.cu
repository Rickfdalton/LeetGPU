#include <cuda_runtime.h>
#include <math.h>
#define ALPHA 0.01

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<N){
        output[idx] = input[idx];
        // float abs_x = fabsf(x);
        // output[idx] = abs_x + (1+ALPHA)*(x-abs_x)/2.0f  ;
        if(input[idx]<=0){
            output[idx]=ALPHA*output[idx];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

/*
notes, we can define a function in abs(x) and x but it doesnt overshadow the control divergence 
*/